from __future__ import annotations

# Ablation-only copy of train_controlled_latefusion.py.
# The original training script is intentionally left unchanged.

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor

from datasets.base import sample_records
from datasets.mmimdb import MMIMDbDataset
from models.controlled_late_fusion import ControlledLateFusionModel, _restore_rng_state
from utils.adaptive_calibration import calibrate_branch_matched_rho_f
from utils.controlled_metrics import (
    GEOMETRY_KEYS,
    compute_controlled_metrics,
    mean_dict,
    primary_metric_name,
    select_primary_metric,
)
from utils.io_utils import append_jsonl, ensure_dir, save_json, save_text, str2bool
from utils.logging_utils import setup_logger
from utils.optim import compute_grad_norm
from utils.seed import build_dataloader_generator, seed_worker, set_seed
from utils.time_utils import build_elapsed_record


HATEFUL_MEMES_SPLIT_FILENAME_MAP = {
    "train": "train.jsonl",
    "dev_seen": "dev_seen.jsonl",
    "dev_unseen": "dev_unseen.jsonl",
    "test_seen": "test_seen.jsonl",
    "test_unseen": "test_unseen.jsonl",
}


def tensor_item(value: torch.Tensor | float | int) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def resolve_device(device_arg: str) -> torch.device:
    device_text = str(device_arg).strip().lower()
    if device_text == "cpu":
        return torch.device("cpu")
    if device_text.startswith("cuda"):
        return torch.device(device_text if torch.cuda.is_available() else "cpu")
    if device_text.isdigit():
        if torch.cuda.is_available():
            return torch.device(f"cuda:{int(device_text)}")
        return torch.device("cpu")
    return torch.device(device_text)


def maybe_autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def sanitize_label_name(label: str) -> str:
    text = str(label).strip()
    safe = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-"}:
            safe.append(ch)
        elif ch in {" ", "/", "\\", ":"}:
            safe.append("_")
    out = "".join(safe).strip("_")
    return out or "label"


def hateful_memes_text(split_value: str) -> str:
    key = str(split_value).strip().lower()
    return HATEFUL_MEMES_SPLIT_FILENAME_MAP.get(key, split_value)


def resolve_hateful_memes_split_path(root: str, split_value: str) -> str:
    candidate = hateful_memes_text(split_value)
    candidates = []
    if os.path.isabs(candidate):
        candidates.append(candidate)
    candidates.append(os.path.join(root, candidate))
    candidates.append(candidate)
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Hateful Memes split file not found for split={split_value}. "
        f"Tried: {candidates}"
    )


def resolve_hateful_memes_image_path(root: str, image_value: str) -> str:
    image_value = str(image_value or "").strip()
    if not image_value:
        raise FileNotFoundError("Hateful Memes record is missing the 'img' field.")

    candidates: List[str] = []
    if os.path.isabs(image_value):
        candidates.append(image_value)
    candidates.append(os.path.join(root, image_value))

    basename = os.path.basename(image_value)
    if basename:
        candidates.append(os.path.join(root, "img", basename))

    seen = set()
    deduped = []
    for path in candidates:
        normed = os.path.normpath(path)
        if normed in seen:
            continue
        deduped.append(path)
        seen.add(normed)

    for path in deduped:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Could not resolve Hateful Memes image path for img={image_value}. "
        f"Tried: {deduped}"
    )


class ControlledHatefulMemesDataset(Dataset):
    task_name = "hateful_memes"

    def __init__(
        self,
        root: str,
        split: str,
        *,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Hateful Memes root directory does not exist: {root}")
        self.root = root
        self.split = split
        self.split_path = resolve_hateful_memes_split_path(root, split)
        self.records: List[Dict[str, Any]] = []
        self.missing_records: List[Dict[str, Any]] = []

        with open(self.split_path, "r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                text = line.strip()
                if not text:
                    continue
                item = json.loads(text)
                try:
                    image_path = resolve_hateful_memes_image_path(root, item.get("img", ""))
                except FileNotFoundError as exc:
                    self.missing_records.append(
                        {
                            "line_index": line_index,
                            "id": item.get("id"),
                            "img": item.get("img", ""),
                            "error": str(exc),
                        }
                    )
                    continue
                if item.get("label") is None:
                    raise ValueError(
                        f"Hateful Memes split {self.split_path} contains a record without label. "
                        f"Controlled training/evaluation requires labeled data."
                    )
                self.records.append(
                    {
                        "id": item.get("id", line_index),
                        "text": str(item.get("text", "")),
                        "label": int(item.get("label", 0)) if item.get("label") is not None else 0,
                        "image_path": image_path,
                    }
                )

        self.records = sample_records(self.records, max_samples=max_samples, seed=seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return dict(self.records[index])

    def summary(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "split_path": self.split_path,
            "num_samples": len(self.records),
            "num_missing_skipped": len(self.missing_records),
        }


class ControlledLateFusionCollator:
    def __init__(
        self,
        *,
        dataset: str,
        image_processor,
        tokenizer,
        max_text_length: int,
    ) -> None:
        self.dataset = str(dataset).strip().lower()
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_text_length = int(max_text_length)

    def _build_text(self, sample: Dict[str, Any]) -> str:
        if self.dataset == "hateful_memes":
            return str(sample.get("text", "")).strip()
        title = str(sample.get("title", "")).strip()
        plot = str(sample.get("plot", "")).strip()
        if title and plot:
            return f"{title}\n\n{plot}"
        return title or plot

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images: List[Image.Image] = []
        texts: List[str] = []
        sample_ids: List[str] = []
        image_paths: List[str] = []
        labels: List[Any] = []

        for sample in batch:
            sample_ids.append(str(sample.get("id")))
            image_path = str(sample.get("image_path", ""))
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file does not exist: {image_path}")
            image_paths.append(image_path)
            images.append(Image.open(image_path).convert("RGB"))
            texts.append(self._build_text(sample))
            if self.dataset == "hateful_memes":
                labels.append(int(sample.get("label", 0)))
            else:
                labels.append([float(x) for x in sample.get("labels", [])])

        vision_inputs = self.image_processor(images=images, return_tensors="pt")
        model_max_length = int(getattr(self.tokenizer, "model_max_length", self.max_text_length) or self.max_text_length)
        if model_max_length <= 0 or model_max_length > 100000:
            model_max_length = self.max_text_length
        text_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=min(self.max_text_length, model_max_length),
            return_tensors="pt",
        )

        if self.dataset == "hateful_memes":
            label_tensor = torch.tensor(labels, dtype=torch.float32)
        else:
            label_tensor = torch.tensor(labels, dtype=torch.float32)

        return {
            "dataset_name": self.dataset,
            "sample_ids": sample_ids,
            "texts": texts,
            "image_paths": image_paths,
            "pixel_values": vision_inputs["pixel_values"],
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs.get("attention_mask"),
            "labels": label_tensor,
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["hateful_memes", "mmimdb"])
    parser.add_argument("--fusion_type", type=str, required=True, choices=["concat_mlp", "gated_mlp"])
    parser.add_argument("--method", type=str, required=True, choices=["vanilla", "sam", "cofla", "vanilla_lora", "sam_lora", "esam_lora", "msam_lora", "masam_lora", "dgl_lora"])
    parser.add_argument("--exp_name", type=str, default="", help="Optional experiment name. If empty, the script builds one automatically.")
    parser.add_argument("--clip_path", type=str, default="./external_models/clip-vit-base-patch32")
    parser.add_argument("--roberta_path", type=str, default="./external_models/roberta-base")
    parser.add_argument("--hm_root", type=str, default="./data/hateful_memes")
    parser.add_argument("--mmimdb_root", type=str, default="./data/mmimdb")
    parser.add_argument("--result_root", type=str, default="./outputs/controlled_latefusion")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"], help="Optimizer used for controlled ablation training. Use sgd to run the SGD breadth check.")
    parser.add_argument("--sgd_momentum", type=float, default=0.9, help="Momentum for torch.optim.SGD when --optimizer sgd.")
    parser.add_argument("--sgd_nesterov", type=str2bool, default=False, help="Whether to use Nesterov momentum for torch.optim.SGD.")
    parser.add_argument("--branch_tuning_mode", type=str, default="frozen", choices=["frozen", "branch_lora_full_fusion"], help="Controlled adaptation scope: 'frozen' keeps unimodal branches frozen; 'branch_lora_full_fusion' trains LoRA adapters in visual/text branches plus the full fusion block and classifier.")
    parser.add_argument("--branch_lora_lr", type=float, default=None, help="Optional LR for branch LoRA params. Defaults to --learning_rate when omitted.")
    parser.add_argument("--branch_lora_r", type=int, default=8)
    parser.add_argument("--branch_lora_alpha", type=int, default=16)
    parser.add_argument("--branch_lora_dropout", type=float, default=0.05)
    parser.add_argument("--branch_lora_target_modules", type=str, default="auto", help="Comma-separated LoRA target module names for controlled branches, or 'auto'.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--fusion_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rho_v", type=float, default=1e-3)
    parser.add_argument("--rho_t", type=float, default=1e-3)
    parser.add_argument("--rho_f", type=float, default=1e-2)
    parser.add_argument("--auto_rho_f", type=str2bool, default=None)
    parser.add_argument("--rho_f_calib_batches", type=int, default=20)
    parser.add_argument("--rho_f_calib_stat", type=str, default="median", choices=["median", "mean"])
    parser.add_argument("--eval_rho_v", type=float, default=None, help="Evaluation-only visual perturbation radius. Defaults to --rho_v when omitted.")
    parser.add_argument("--eval_rho_t", type=float, default=None, help="Evaluation-only text perturbation radius. Defaults to --rho_t when omitted.")
    parser.add_argument("--eval_rho_f", type=float, default=None, help="Evaluation-only fusion perturbation radius. Defaults to --rho_f when omitted.")
    parser.add_argument("--geometry_metric", type=str, default="fast_fcf", choices=["fast_fcf", "exact_fcf", "both"], help="Geometry evaluation protocol. FAST-FCF is the default; exact FCF is optional diagnostic only.")
    parser.add_argument("--cofla_train_geometry", type=str, default="exact_fcf", choices=["exact_fcf", "fast_fcf"], help="COFLA training geometry objective. exact_fcf uses the original three-probe COFLA-E; fast_fcf uses one-probe fusion-exact/branch-calibrated COFLA-F.")
    parser.add_argument("--cofla_f_ema_mu", type=float, default=0.9, help="Deprecated compatibility flag. One-probe COFLA-F does not use EMA.")
    parser.add_argument(
        "--cofla_ablation_mode",
        type=str,
        default="full",
        choices=[
            "base",
            "always_on",
            "gate_no_proj",
            "gate_with_proj",
            "branch_proj_only",
            "fusion_proj_only",
            "full",
        ],
        help=(
            "Ablation mode used only when --method cofla. "
            "base: train with L0 only; "
            "always_on: train with L_pert only; "
            "gate_no_proj: FCF-gated robust objective without projection; "
            "gate_with_proj/full: FCF-gated robust objective with both branch and fusion projections; "
            "branch_proj_only/fusion_proj_only: enable only one safeguard for finer analysis."
        ),
    )
    parser.add_argument("--sam_rho", type=float, default=0.05)
    parser.add_argument("--alpha_v", type=float, default=0.5)
    parser.add_argument("--alpha_t", type=float, default=0.5)
    parser.add_argument("--esam_keep_ratio", type=float, default=0.75, help="ESAM SDS keep ratio gamma; selects samples with the largest perturbation-induced loss increase.")
    parser.add_argument("--esam_swp_prob", type=float, default=0.6, help="ESAM SWP probability beta for stochastic weight perturbation.")
    parser.add_argument("--msam_shapley_eps", type=float, default=1e-8, help="Numerical floor for normalizing two-modality Shapley contributions in M-SAM.")
    parser.add_argument("--masam_aps_alpha", type=float, default=0.5, help="MASAM APS balance between convergence-speed decay and gradient alignment.")
    parser.add_argument("--masam_ma_beta", type=float, default=0.9, help="MASAM EMA factor for unimodal-loss moving averages.")
    parser.add_argument("--masam_rho_min_scale", type=float, default=0.5, help="Deprecated compatibility option; the corrected MASAM uses MDPS cosine scaling instead.")
    parser.add_argument("--masam_rho_max_scale", type=float, default=2.0, help="Deprecated compatibility option; the corrected MASAM uses MDPS cosine scaling instead.")
    parser.add_argument("--dgl_correction_strength", type=float, default=1.0, help="Deprecated compatibility option; corrected DGL replaces encoder gradients with unimodal modality-dropout gradients.")
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--fp16", type=str2bool, default=False)
    parser.add_argument("--local_files_only", type=str2bool, default=True)
    parser.add_argument("--val_compute_geometry", type=str2bool, default=True)
    parser.add_argument("--test_compute_geometry", type=str2bool, default=True)
    parser.add_argument("--max_text_length", type=int, default=512)

    parser.add_argument("--hm_train_split", type=str, default="train.jsonl")
    parser.add_argument("--hm_val_split", type=str, default="dev_seen.jsonl")
    parser.add_argument("--hm_test_split", type=str, default="dev_unseen.jsonl")

    parser.add_argument("--mmimdb_train_split", type=str, default="train")
    parser.add_argument("--mmimdb_val_split", type=str, default="dev")
    parser.add_argument("--mmimdb_test_split", type=str, default="test")
    parser.add_argument("--mmimdb_split_file", type=str, default="split.json")
    parser.add_argument("--mmimdb_label_names", type=str, default="")
    parser.add_argument("--mmimdb_threshold", type=float, default=0.5)
    return parser


def encoder_setup_for_dataset(dataset: str) -> Tuple[str, str]:
    dataset_key = str(dataset).strip().lower()
    if dataset_key == "hateful_memes":
        return "clip", "clip"
    if dataset_key == "mmimdb":
        return "clip", "roberta"
    raise ValueError(f"Unsupported dataset: {dataset}")


def text_encoder_type_for_dataset(dataset: str) -> str:
    _, text_branch = encoder_setup_for_dataset(dataset)
    return "clip_text" if text_branch == "clip" else "roberta"


def build_experiment_name(args) -> str:
    visual_branch, text_branch = encoder_setup_for_dataset(args.dataset)
    scope = str(getattr(args, "branch_tuning_mode", "frozen")).strip().lower()
    scope_tag = "branch_lora_fullfusion" if scope == "branch_lora_full_fusion" else "frozen_branch"
    return f"{args.dataset}_{visual_branch}_{text_branch}_{args.fusion_type}_{args.method}_{scope_tag}_seed{args.seed}"


def build_experiment_dirs(result_root: str, exp_name: str) -> Dict[str, str]:
    exp_dir = ensure_dir(os.path.join(result_root, exp_name))
    dirs = {
        "exp_dir": exp_dir,
        "logs": ensure_dir(os.path.join(exp_dir, "logs")),
        "metrics": ensure_dir(os.path.join(exp_dir, "metrics")),
        "predictions": ensure_dir(os.path.join(exp_dir, "predictions")),
        "checkpoints": ensure_dir(os.path.join(exp_dir, "checkpoints")),
        "best_checkpoint": ensure_dir(os.path.join(exp_dir, "checkpoints", "best")),
        "latest_checkpoint": ensure_dir(os.path.join(exp_dir, "checkpoints", "latest")),
    }
    return dirs


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def infer_mmimdb_label_names(args) -> List[str]:
    label_names = MMIMDbDataset.parse_label_names(args.mmimdb_label_names)
    if label_names:
        return label_names
    label_names = MMIMDbDataset.infer_label_names(args.mmimdb_root, split_file=args.mmimdb_split_file)
    if not label_names:
        raise ValueError(
            f"Could not infer MM-IMDb label names from root={args.mmimdb_root} split_file={args.mmimdb_split_file}"
        )
    return label_names


def build_dataloaders(args, image_processor, tokenizer, logger):
    if args.dataset == "hateful_memes":
        train_dataset = ControlledHatefulMemesDataset(
            args.hm_root,
            args.hm_train_split,
            max_samples=args.max_train_samples,
            seed=args.seed,
        )
        val_dataset = ControlledHatefulMemesDataset(
            args.hm_root,
            args.hm_val_split,
            max_samples=args.max_val_samples,
            seed=args.seed,
        )
        test_dataset = ControlledHatefulMemesDataset(
            args.hm_root,
            args.hm_test_split,
            max_samples=args.max_test_samples,
            seed=args.seed,
        )
        label_names = ["hateful"]
    else:
        label_names = infer_mmimdb_label_names(args)
        train_dataset = MMIMDbDataset(
            args.mmimdb_root,
            args.mmimdb_train_split,
            max_samples=args.max_train_samples,
            seed=args.seed,
            label_names=label_names,
            split_file=args.mmimdb_split_file,
        )
        val_dataset = MMIMDbDataset(
            args.mmimdb_root,
            args.mmimdb_val_split,
            max_samples=args.max_val_samples,
            seed=args.seed,
            label_names=label_names,
            split_file=args.mmimdb_split_file,
        )
        test_dataset = MMIMDbDataset(
            args.mmimdb_root,
            args.mmimdb_test_split,
            max_samples=args.max_test_samples,
            seed=args.seed,
            label_names=label_names,
            split_file=args.mmimdb_split_file,
        )

    logger.info("Train dataset summary: %s", json.dumps(train_dataset.summary(), ensure_ascii=False))
    logger.info("Val dataset summary: %s", json.dumps(val_dataset.summary(), ensure_ascii=False))
    logger.info("Test dataset summary: %s", json.dumps(test_dataset.summary(), ensure_ascii=False))

    collator = ControlledLateFusionCollator(
        dataset=args.dataset,
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_text_length=args.max_text_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        worker_init_fn=seed_worker,
        generator=build_dataloader_generator(args.seed),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        worker_init_fn=seed_worker,
        generator=build_dataloader_generator(args.seed + 1),
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        worker_init_fn=seed_worker,
        generator=build_dataloader_generator(args.seed + 2),
        persistent_workers=args.num_workers > 0,
    )

    return label_names, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def build_calibration_loader(train_loader, args):
    return DataLoader(
        train_loader.dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=train_loader.collate_fn,
        worker_init_fn=seed_worker,
        generator=build_dataloader_generator(args.seed + 197),
        persistent_workers=args.num_workers > 0,
    )


def rho_f_calibration_payload(args) -> Dict[str, Any]:
    return {
        "auto_rho_f": bool(getattr(args, "auto_rho_f", False)),
        "rho_f_before_calibration": float(getattr(args, "rho_f_before_calibration", args.rho_f)),
        "rho_f_after_calibration": float(getattr(args, "rho_f_after_calibration", args.rho_f)),
        "rho_f_calib_requested_batches": int(getattr(args, "rho_f_calib_requested_batches", getattr(args, "rho_f_calib_batches", 0))),
        "rho_f_calib_count": int(getattr(args, "rho_f_calib_count", 0)),
        "rho_f_calib_mean": getattr(args, "rho_f_calib_mean", None),
        "rho_f_calib_median": getattr(args, "rho_f_calib_median", None),
        "rho_f_calib_min": getattr(args, "rho_f_calib_min", None),
        "rho_f_calib_max": getattr(args, "rho_f_calib_max", None),
        "rho_f_calib_stat": getattr(args, "rho_f_calib_stat", "median"),
        "rho_f_calib_values": list(getattr(args, "rho_f_calib_values", [])),
    }


def maybe_run_rho_f_calibration(
    model: ControlledLateFusionModel,
    train_loader: DataLoader,
    args,
    *,
    device: torch.device,
    dirs: Dict[str, str],
    logger,
) -> Dict[str, Any]:
    calibration_info: Dict[str, Any]
    if bool(getattr(args, "auto_rho_f", False)):
        model.train()
        calibration_loader = build_calibration_loader(train_loader, args)

        def _prepare_batch(raw_batch):
            return move_batch_to_device(raw_batch, device)

        def _compute_stats(batch):
            with torch.enable_grad():
                with maybe_autocast(device, args.fp16):
                    return model.compute_linear_geometry_stats(
                        batch,
                        rho_v=args.rho_v,
                        rho_t=args.rho_t,
                        retain_graph=False,
                    )

        calibration_info = calibrate_branch_matched_rho_f(
            dataloader=calibration_loader,
            prepare_batch_fn=_prepare_batch,
            compute_stats_fn=_compute_stats,
            rho_f_before=float(args.rho_f),
            num_batches=int(args.rho_f_calib_batches),
            stat=str(args.rho_f_calib_stat),
            logger=logger,
        )
        args.rho_f = float(calibration_info["rho_f_after_calibration"])
    else:
        calibration_info = {
            "rho_f_before_calibration": float(args.rho_f),
            "rho_f_after_calibration": float(args.rho_f),
            "rho_f_calib_requested_batches": int(getattr(args, "rho_f_calib_batches", 0)),
            "rho_f_calib_count": 0,
            "rho_f_calib_mean": None,
            "rho_f_calib_median": None,
            "rho_f_calib_min": None,
            "rho_f_calib_max": None,
            "rho_f_calib_stat": str(getattr(args, "rho_f_calib_stat", "median")),
            "rho_f_calib_values": [],
        }

    for key, value in calibration_info.items():
        setattr(args, key, value)

    logger.info(
        "[AdaptiveCalibration] rho_f before=%s, after=%s, calib_batches=%s, valid=%s, stat=%s",
        calibration_info["rho_f_before_calibration"],
        calibration_info["rho_f_after_calibration"],
        calibration_info["rho_f_calib_requested_batches"],
        calibration_info["rho_f_calib_count"],
        calibration_info["rho_f_calib_stat"],
    )
    append_jsonl(
        {"type": "adaptive_calibration", **rho_f_calibration_payload(args)},
        os.path.join(dirs["metrics"], "metrics.jsonl"),
    )
    return calibration_info


def build_model_and_tokenizers(args, device: torch.device):
    if not os.path.exists(args.clip_path):
        raise FileNotFoundError(f"CLIP path does not exist: {args.clip_path}")
    if args.dataset == "mmimdb" and not os.path.exists(args.roberta_path):
        raise FileNotFoundError(f"RoBERTa path does not exist: {args.roberta_path}")

    image_processor = CLIPImageProcessor.from_pretrained(
        args.clip_path,
        local_files_only=args.local_files_only,
    )
    tokenizer_path = args.clip_path if args.dataset == "hateful_memes" else args.roberta_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=args.local_files_only,
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    label_names = infer_mmimdb_label_names(args) if args.dataset == "mmimdb" else ["hateful"]
    model = ControlledLateFusionModel(
        dataset_type=args.dataset,
        text_encoder_type=text_encoder_type_for_dataset(args.dataset),
        fusion_type=args.fusion_type,
        clip_path=args.clip_path,
        roberta_path=args.roberta_path if args.dataset == "mmimdb" else None,
        fusion_dim=args.fusion_dim,
        dropout=args.dropout,
        num_labels=len(label_names) if args.dataset == "mmimdb" else 1,
        freeze_branches=str(getattr(args, "branch_tuning_mode", "frozen")).strip().lower() == "frozen",
        branch_tuning_mode=args.branch_tuning_mode,
        branch_lora_r=args.branch_lora_r,
        branch_lora_alpha=args.branch_lora_alpha,
        branch_lora_dropout=args.branch_lora_dropout,
        branch_lora_target_modules=args.branch_lora_target_modules,
        local_files_only=args.local_files_only,
        eps=args.eps,
    )
    model.to(device)
    return model, image_processor, tokenizer


def geometry_tensor_logs(geom: Dict[str, torch.Tensor]) -> Dict[str, float]:
    logs = {}
    for key in GEOMETRY_KEYS:
        if key in geom:
            logs[key] = tensor_item(geom[key])
    return logs


def _differentiable_grad_norm(
    grads: Sequence[Optional[torch.Tensor]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
) -> torch.Tensor:
    """Differentiable L2 norm for FAST-FCF training gradients."""
    sq: Optional[torch.Tensor] = None
    for grad in grads:
        if grad is None:
            continue
        term = grad.float().pow(2).sum()
        sq = term if sq is None else sq + term.to(device=sq.device)
    if sq is None:
        return torch.zeros((), device=device, dtype=dtype)
    return torch.sqrt(sq.to(device=device) + float(eps)).to(device=device, dtype=dtype)


def compute_fast_train_geometry_from_probe(
    model: ControlledLateFusionModel,
    batch: Dict[str, Any],
    probe: Dict[str, object],
    *,
    rho_v: float,
    rho_t: float,
    rho_f: float,
    eps: float,
    retain_graph: bool = True,
) -> Dict[str, Any]:
    """One-probe fusion-exact COFLA-F training geometry.

    COFLA-F keeps the finite-step fusion sharpness S_f=[L_f^+-L_0]_+ for
    projection and uses a detached branch-side first-order calibration as the
    denominator.  It therefore avoids differentiating through gradient-norm
    sharpness while reducing COFLA-E's three probes to one fusion probe.
    """
    z_v = probe["z_v"]
    z_t = probe["z_t"]
    loss0 = probe["loss"]
    device = loss0.device
    dtype = loss0.dtype

    terms = model._compute_linear_geometry_terms(
        batch,
        probe,
        rho_v=rho_v,
        rho_t=rho_t,
        rho_f=rho_f,
        retain_graph=retain_graph,
    )
    gv_norm = terms["gv_norm"].to(device=device, dtype=dtype)
    gt_norm = terms["gt_norm"].to(device=device, dtype=dtype)
    gf_norm = terms["gf_norm"].to(device=device, dtype=dtype)
    delta_f = terms["delta_f"]

    rng_state = probe.get("rng_state")
    if model.training and rng_state is not None:
        _restore_rng_state(rng_state, device)
    loss_f = model.forward_from_activations(
        z_v,
        z_t,
        labels=batch["labels"],
        fusion_param_overrides=delta_f if delta_f else None,
    ).loss
    if loss_f is None:
        raise RuntimeError("COFLA-F one-probe fusion forward failed to produce a loss.")

    s_v = (float(rho_v) * gv_norm).detach()
    s_t = (float(rho_t) * gt_norm).detach()
    s_branch = terms.get("branch_linear_proxy", 0.5 * (s_v + s_t)).to(device=device, dtype=dtype).detach()
    s_f = (loss_f - loss0).clamp_min(0.0)
    eps_tensor = torch.tensor(float(eps), device=device, dtype=dtype)
    fcf = (s_f + eps_tensor) / (s_branch + eps_tensor)
    rfcf = torch.log(fcf.clamp_min(float(eps)))
    r_lin_fcf = torch.log(((float(rho_f) * gf_norm) + eps_tensor) / (s_branch + eps_tensor))

    return {
        "loss": loss0,
        "loss_f_plus": loss_f,
        "s_v": s_v,
        "s_t": s_t,
        "s_branch": s_branch,
        "s_f": s_f,
        "fcf": fcf,
        "rfcf": rfcf,
        "r_lin_fcf": r_lin_fcf,
        "grad_zv_norm": gv_norm,
        "grad_zt_norm": gt_norm,
        "grad_fusion_norm": gf_norm,
        # Backward-compatible aliases for geometry logging.
        "fast_s_v": s_v,
        "fast_s_t": s_t,
        "fast_s_branch": s_branch,
        "fast_s_f": s_f,
        "fast_fcf": fcf.detach(),
        "fast_rfcf": rfcf.detach(),
        "fast_grad_zv_norm": gv_norm.detach(),
        "fast_grad_zt_norm": gt_norm.detach(),
        "fast_grad_fusion_norm": gf_norm.detach(),
    }

def bounded_positive_fcf_gate(value: torch.Tensor) -> torch.Tensor:
    """Parameter-free bounded gate used by the robust COFLA objective.

    The gate is activated only for positive branch--fusion sharpness mismatch and
    is detached so that it controls the objective strength without becoming an
    additional optimization target.  No manually tuned loss weight or threshold
    is introduced.
    """
    positive = torch.relu(value.float())
    gate = positive / (1.0 + positive)
    return gate.detach().to(device=value.device, dtype=value.dtype)




def grad_map_from_grads(
    named_params: Sequence[Tuple[str, torch.nn.Parameter]],
    grads: Sequence[Optional[torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Convert autograd.grad outputs into a name -> detached gradient map.

    Several baseline implementations compute gradients with ``allow_unused=True``.
    This helper drops missing gradients and stores detached tensors so the maps
    can be safely reused after the source graph is released.
    """
    grad_map: Dict[str, torch.Tensor] = {}
    for (name, param), grad in zip(named_params, grads):
        if grad is None:
            continue
        grad_map[name] = grad.detach().to(device=param.device, dtype=param.dtype)
    return grad_map


def merge_grad_maps_sum(*grad_maps: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Merge multiple gradient maps by summing tensors with the same name."""
    merged: Dict[str, torch.Tensor] = {}
    for grad_map in grad_maps:
        if not grad_map:
            continue
        for name, grad in grad_map.items():
            if grad is None:
                continue
            if name in merged:
                merged[name] = merged[name] + grad.to(device=merged[name].device, dtype=merged[name].dtype)
            else:
                merged[name] = grad.detach().clone()
    return merged


def dot_and_norm_stats(
    named_params: Sequence[Tuple[str, torch.nn.Parameter]],
    grad_a: Dict[str, torch.Tensor],
    grad_b: Dict[str, torch.Tensor],
) -> Tuple[float, float, float]:
    """Return dot(grad_a, grad_b), ||grad_a||^2, and ||grad_b||^2.

    The returned values are Python floats so they can be logged and used for
    projection coefficients without keeping autograd graphs alive.
    """
    dot_value = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0
    for name, _ in named_params:
        ga = grad_a.get(name)
        gb = grad_b.get(name)
        if ga is not None:
            ga_f = ga.detach().float()
            norm_a_sq += float(ga_f.pow(2).sum().cpu().item())
        if gb is not None:
            gb_f = gb.detach().float()
            norm_b_sq += float(gb_f.pow(2).sum().cpu().item())
        if ga is not None and gb is not None:
            dot_value += float((ga.detach().float() * gb.detach().float()).sum().cpu().item())
    return dot_value, norm_a_sq, norm_b_sq


def cosine_between_grad_maps(
    named_params: Sequence[Tuple[str, torch.nn.Parameter]],
    grad_a: Dict[str, torch.Tensor],
    grad_b: Dict[str, torch.Tensor],
    eps: float = 1e-12,
) -> float:
    """Cosine similarity between two named-gradient maps."""
    dot_value, norm_a_sq, norm_b_sq = dot_and_norm_stats(named_params, grad_a, grad_b)
    denom = math.sqrt(max(norm_a_sq, 0.0)) * math.sqrt(max(norm_b_sq, 0.0)) + float(eps)
    if denom <= float(eps):
        return 0.0
    return float(dot_value / denom)


def build_param_perturbation(
    named_params: Sequence[Tuple[str, torch.nn.Parameter]],
    grads: Sequence[Optional[torch.Tensor]],
    *,
    radius: float,
    device: torch.device,
    eps: float = 1e-12,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Build a SAM-style parameter perturbation from precomputed gradients.

    The perturbation is ``radius * grad / ||grad||`` over the provided parameter
    subset.  The function only constructs the perturbation; applying and
    restoring it is handled by the caller.
    """
    grad_sq = 0.0
    for grad in grads:
        if grad is not None:
            grad_sq += float(grad.detach().float().pow(2).sum().cpu().item())
    grad_norm_value = math.sqrt(max(grad_sq, 0.0))
    grad_norm = torch.tensor(grad_norm_value, device=device, dtype=torch.float32)
    if float(radius) <= 0.0 or grad_norm_value <= float(eps):
        return {}, grad_norm

    scale = float(radius) / (grad_norm_value + float(eps))
    perturb: Dict[str, torch.Tensor] = {}
    for (name, param), grad in zip(named_params, grads):
        if grad is None:
            continue
        perturb[name] = grad.detach().to(device=param.device, dtype=param.dtype) * scale
    return perturb, grad_norm


def build_esam_swp_perturbation(
    named_params: Sequence[Tuple[str, torch.nn.Parameter]],
    grads: Sequence[Optional[torch.Tensor]],
    *,
    radius: float,
    swp_prob: float,
    device: torch.device,
    eps: float = 1e-12,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, int, float]:
    """Build ESAM stochastic-weight-perturbation (SWP) for a parameter subset.

    Each tensor is selected with probability ``swp_prob``.  Following the ESAM
    idea, the selected perturbation is rescaled by ``1 / swp_prob`` so that the
    expected perturbation direction remains comparable to standard SAM.  At
    least one tensor with a valid gradient is selected when possible.
    """
    beta = min(1.0, max(float(swp_prob), float(eps)))
    candidates: List[Tuple[Tuple[str, torch.nn.Parameter], torch.Tensor]] = []
    for item, grad in zip(named_params, grads):
        if grad is not None:
            candidates.append((item, grad))

    selected: List[Tuple[Tuple[str, torch.nn.Parameter], torch.Tensor]] = []
    for item, grad in candidates:
        if torch.rand((), device=device).item() <= beta:
            selected.append((item, grad))
    if not selected and candidates:
        # Keep ESAM numerically usable even when the Bernoulli mask is empty.
        random_index = int(torch.randint(low=0, high=len(candidates), size=(), device=device).item())
        selected.append(candidates[random_index])

    grad_sq = 0.0
    for _, grad in selected:
        grad_sq += float(grad.detach().float().pow(2).sum().cpu().item())
    grad_norm_value = math.sqrt(max(grad_sq, 0.0))
    grad_norm = torch.tensor(grad_norm_value, device=device, dtype=torch.float32)
    if float(radius) <= 0.0 or grad_norm_value <= float(eps):
        return {}, grad_norm, 0, beta

    scale = float(radius) / (beta * (grad_norm_value + float(eps)))
    perturb: Dict[str, torch.Tensor] = {}
    for (name, param), grad in selected:
        perturb[name] = grad.detach().to(device=param.device, dtype=param.dtype) * scale
    return perturb, grad_norm, len(perturb), beta


def split_branch_named_parameters(
    named_params: Sequence[Tuple[str, torch.nn.Parameter]],
) -> Tuple[List[Tuple[str, torch.nn.Parameter]], List[Tuple[str, torch.nn.Parameter]]]:
    """Split branch LoRA parameters into visual and textual groups.

    The controlled model has used slightly different naming schemes across
    versions, so this matcher accepts common visual/text aliases instead of
    relying on a single exact prefix.
    """
    visual_keywords = (
        "visual", "vision", "image", "clip_vision", "vision_model", "visual_encoder", "image_encoder",
    )
    text_keywords = (
        "text", "txt", "language", "lang", "roberta", "bert", "clip_text", "text_encoder", "text_model",
    )
    visual: List[Tuple[str, torch.nn.Parameter]] = []
    text: List[Tuple[str, torch.nn.Parameter]] = []
    unmatched: List[Tuple[str, torch.nn.Parameter]] = []

    for name, param in named_params:
        lower = name.lower()
        is_visual = any(key in lower for key in visual_keywords)
        is_text = any(key in lower for key in text_keywords)
        if is_visual and not is_text:
            visual.append((name, param))
        elif is_text and not is_visual:
            text.append((name, param))
        elif is_visual and is_text:
            # Prefer explicit branch prefixes when both keyword sets appear.
            if lower.startswith(("visual", "vision", "image")):
                visual.append((name, param))
            elif lower.startswith(("text", "txt", "language", "roberta", "bert")):
                text.append((name, param))
            else:
                unmatched.append((name, param))
        else:
            unmatched.append((name, param))

    # Conservative fallback for older controlled models whose LoRA names did not
    # include branch identifiers: split unmatched tensors into two halves so the
    # baseline code can still run, while preserving explicitly matched tensors.
    if unmatched:
        if not visual and not text:
            midpoint = len(unmatched) // 2
            visual.extend(unmatched[:midpoint])
            text.extend(unmatched[midpoint:])
        elif not visual:
            visual.extend(unmatched)
        elif not text:
            text.extend(unmatched)
    return visual, text


def compute_controlled_per_sample_losses(
    model: ControlledLateFusionModel,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Return per-sample task losses for the controlled datasets.

    Hateful Memes is binary classification with a single logit.  MM-IMDb is
    multi-label classification, so label-wise BCE is averaged per sample.
    """
    dataset_key = str(getattr(model, "dataset_type", "")).strip().lower()
    if dataset_key == "hateful_memes":
        logits_flat = logits.squeeze(-1)
        label_tensor = labels.to(device=logits.device, dtype=logits.dtype).view_as(logits_flat)
        return F.binary_cross_entropy_with_logits(logits_flat, label_tensor, reduction="none")

    label_tensor = labels.to(device=logits.device, dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, label_tensor, reduction="none")
    if loss.ndim <= 1:
        return loss.view(-1)
    return loss.mean(dim=tuple(range(1, loss.ndim)))

def build_projection_correction_from_objective(
    *,
    named_params: Sequence[Tuple[str, torch.nn.Parameter]],
    loss_task: torch.Tensor,
    objective: torch.Tensor,
    objective_value: float,
    eps: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """Minimum projection correction used by COFLA.

    The task gradient is left unchanged unless a task update would increase the
    specified flatness/stability objective in the first-order approximation.
    This keeps the update rule weight-free and avoids turning COFLA into a
    manually weighted auxiliary-loss method.
    """
    params = [param for _, param in named_params]
    if not params:
        return {}, {
            "alpha_star": 0.0,
            "projection_active": 0.0,
            "projection_dot": 0.0,
            "projection_norm_task": 0.0,
            "projection_norm_obj": 0.0,
        }
    grads_task = torch.autograd.grad(
        outputs=loss_task,
        inputs=params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    grads_obj = torch.autograd.grad(
        outputs=objective,
        inputs=params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    grad_task_map = grad_map_from_grads(named_params, grads_task)
    grad_obj_map = grad_map_from_grads(named_params, grads_obj)
    dot, norm_task_sq, norm_obj_sq = dot_and_norm_stats(named_params, grad_task_map, grad_obj_map)
    alpha_star = 0.0
    if objective_value > 0.0 and norm_obj_sq > 0.0 and math.isfinite(dot):
        alpha_star = max(0.0, -dot / (norm_obj_sq + float(eps)))
    correction: Dict[str, torch.Tensor] = {}
    if alpha_star > 0.0 and math.isfinite(alpha_star):
        for name, param in named_params:
            grad_obj = grad_obj_map.get(name)
            if grad_obj is not None:
                correction[name] = (grad_obj * alpha_star).to(device=param.device, dtype=param.dtype)
    return correction, {
        "alpha_star": float(alpha_star),
        "projection_active": 1.0 if correction else 0.0,
        "projection_dot": float(dot),
        "projection_norm_task": math.sqrt(max(float(norm_task_sq), 0.0)),
        "projection_norm_obj": math.sqrt(max(float(norm_obj_sq), 0.0)),
    }


def build_projection_correction_from_direction(
    *,
    named_params: Sequence[Tuple[str, torch.nn.Parameter]],
    loss_task: torch.Tensor,
    direction_map: Dict[str, torch.Tensor],
    gate_value: float,
    eps: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    params = [param for _, param in named_params]
    if not params or not direction_map:
        return {}, {
            "alpha_star": 0.0,
            "projection_active": 0.0,
            "projection_dot": 0.0,
            "projection_norm_task": 0.0,
            "projection_norm_obj": 0.0,
        }
    grads_task = torch.autograd.grad(
        outputs=loss_task,
        inputs=params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    grad_task_map = grad_map_from_grads(named_params, grads_task)
    dot, norm_task_sq, norm_dir_sq = dot_and_norm_stats(named_params, grad_task_map, direction_map)
    alpha_star = 0.0
    if float(gate_value) > 0.0 and norm_dir_sq > 0.0 and math.isfinite(dot):
        alpha_star = max(0.0, -dot / (norm_dir_sq + float(eps)))
    correction: Dict[str, torch.Tensor] = {}
    scale = float(gate_value) * float(alpha_star)
    if scale > 0.0 and math.isfinite(scale):
        for name, param in named_params:
            direction = direction_map.get(name)
            if direction is not None:
                correction[name] = direction.detach().to(device=param.device, dtype=param.dtype) * scale
    return correction, {
        "alpha_star": float(alpha_star),
        "projection_active": 1.0 if correction else 0.0,
        "projection_dot": float(dot),
        "projection_norm_task": math.sqrt(max(float(norm_task_sq), 0.0)),
        "projection_norm_obj": math.sqrt(max(float(norm_dir_sq), 0.0)),
    }

def compute_modality_dropout_losses(
    model: ControlledLateFusionModel,
    z_v: torch.Tensor,
    z_t: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return visual-only, text-only, and empty-modality losses via modality dropout.

    This follows the DGL/MASAM family of baselines: no extra unimodal heads are
    introduced; instead, the same fusion module/head is reused while the missing
    modality representation is set to zero.
    """
    z_v_zero = torch.zeros_like(z_v).detach()
    z_t_zero = torch.zeros_like(z_t).detach()
    loss_v = model.forward_from_activations(z_v, z_t_zero, labels=labels).loss
    loss_t = model.forward_from_activations(z_v_zero, z_t, labels=labels).loss
    loss_empty = model.forward_from_activations(z_v_zero, z_t_zero, labels=labels).loss
    if loss_v is None or loss_t is None or loss_empty is None:
        raise RuntimeError("Modality-dropout losses failed to produce scalar losses.")
    return loss_v, loss_t, loss_empty


def compute_two_modality_shapley_weights(
    *,
    loss_fuse: torch.Tensor,
    loss_v: torch.Tensor,
    loss_t: torch.Tensor,
    loss_empty: torch.Tensor,
    eps: float,
) -> Tuple[float, float, float, float]:
    """Two-modality Shapley contribution weights using negative loss as utility.

    For utility U(S)=-L(S), phi_v = 0.5[(U(v)-U(empty)) + (U(v,t)-U(t))].
    Negative contributions are floored to zero before normalization for stable
    loss weighting in minibatch training.
    """
    lf = tensor_item(loss_fuse)
    lv = tensor_item(loss_v)
    lt = tensor_item(loss_t)
    le = tensor_item(loss_empty)
    phi_v = 0.5 * ((le - lv) + (lt - lf))
    phi_t = 0.5 * ((le - lt) + (lv - lf))
    pv = max(0.0, float(phi_v))
    pt = max(0.0, float(phi_t))
    denom = pv + pt + float(eps)
    if denom <= float(eps):
        return 0.5, 0.5, float(phi_v), float(phi_t)
    return pv / denom, pt / denom, float(phi_v), float(phi_t)


def compute_grad_alignment(
    named_params: Sequence[Tuple[str, torch.nn.Parameter]],
    grad_a: Dict[str, torch.Tensor],
    grad_b: Dict[str, torch.Tensor],
    eps: float,
) -> float:
    return cosine_between_grad_maps(named_params, grad_a, grad_b, eps=eps)


def run_train_step(
    model: ControlledLateFusionModel,
    batch: Dict[str, Any],
    args,
    method_state: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    method = str(args.method).strip().lower()

    if method in {"vanilla", "vanilla_lora"}:
        outputs = model.forward(batch, branch_requires_grad=False)
        if outputs.loss is None:
            raise RuntimeError(f"{method} step failed to produce a task loss.")
        total_loss = outputs.loss
        logs = {
            "task_loss": tensor_item(total_loss),
            "train_loss": tensor_item(total_loss),
        }
        logs["train_loss"] = tensor_item(total_loss)
        if method == "vanilla_lora":
            logs["controlled_note"] = 1.0
        return total_loss, logs

    if method in {"sam", "sam_lora", "esam_lora"}:
        # Branch-only SAM/ESAM baselines.
        #
        # Baseline definition used in the paper/code comparison:
        #   - SAM/ESAM perturbations are applied only to unimodal branch LoRA
        #     adapters.
        #   - Only branch LoRA gradients are replaced by the corresponding
        #     adversarial gradients.
        #   - The explicit fusion module and classifier/head are updated by the
        #     clean task loss, so these baselines do not receive implicit
        #     fusion-side robust optimization.
        #
        # This keeps SAM-LoRA/ESAM-LoRA as branch-side sharpness-aware baselines
        # and leaves COFLA as the only method that explicitly optimizes the
        # branch--fusion robust objective and fusion projection safeguard.
        base_outputs = model.forward(batch, branch_requires_grad=False)
        if base_outputs.loss is None:
            raise RuntimeError(f"{method} step failed to produce a base loss.")
        base_loss = base_outputs.loss
        logs: Dict[str, float] = {
            "task_loss": tensor_item(base_loss),
        }

        perturb_named = model.get_branch_lora_named_parameters()
        perturb_params = [param for _, param in perturb_named]
        # retain_graph=True is required because the training loop later calls
        # base_loss.backward() to obtain clean gradients for fusion/head.
        grads = torch.autograd.grad(
            outputs=base_loss,
            inputs=perturb_params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        ) if perturb_named else []

        if method == "esam_lora":
            perturb, grad_norm_value, swp_selected_count, swp_beta = build_esam_swp_perturbation(
                perturb_named,
                grads,
                radius=float(args.sam_rho),
                swp_prob=float(getattr(args, "esam_swp_prob", 0.6)),
                device=device,
                eps=args.eps,
            )
        else:
            perturb, grad_norm_value = build_param_perturbation(
                perturb_named,
                grads,
                radius=float(args.sam_rho),
                device=device,
                eps=args.eps,
            )
            swp_selected_count = len(perturb)
            swp_beta = 1.0

        if perturb:
            model._apply_param_perturb(perturb, sign=1)
        try:
            adv_out = model.forward(batch, branch_requires_grad=False)
            if adv_out.loss is None:
                raise RuntimeError(f"{method} step failed to produce adversarial loss.")

            if method == "esam_lora":
                base_per_sample = compute_controlled_per_sample_losses(model, base_outputs.logits, batch["labels"])
                adv_per_sample = compute_controlled_per_sample_losses(model, adv_out.logits, batch["labels"])
                # SDS selects the samples whose loss increases most after the
                # SWP perturbation, not merely the clean hard samples.
                sharpness_increase = (adv_per_sample - base_per_sample).detach()
                keep_ratio = max(0.0, min(1.0, float(args.esam_keep_ratio)))
                batch_size = int(base_per_sample.size(0))
                keep_count = batch_size if batch_size <= 1 else max(1, int(math.ceil(batch_size * keep_ratio)))
                if keep_count >= batch_size:
                    adv_branch_loss = adv_per_sample.mean()
                else:
                    selected = torch.topk(sharpness_increase, k=keep_count, largest=True).indices
                    adv_branch_loss = adv_per_sample.index_select(0, selected).mean()
                logs["esam_selected_count"] = int(keep_count)
                logs["esam_selected_ratio"] = float(keep_count / max(1, batch_size))
                logs["esam_swp_beta"] = float(swp_beta)
                logs["esam_swp_selected_tensors"] = float(swp_selected_count)
                logs["esam_mean_sharpness_increase"] = tensor_item(sharpness_increase.mean())
                logs["esam_base_loss"] = tensor_item(base_loss)
                logs["esam_adv_loss"] = tensor_item(adv_branch_loss)
                logs["esam_grad_norm"] = tensor_item(grad_norm_value)
                logs["esam_perturb_scope"] = 1.0 if perturb else 0.0
            else:
                adv_branch_loss = adv_out.loss
                prefix = "sam"
                logs[f"{prefix}_base_loss"] = tensor_item(base_loss)
                logs[f"{prefix}_adv_loss"] = tensor_item(adv_branch_loss)
                logs[f"{prefix}_grad_norm"] = tensor_item(grad_norm_value)
                logs[f"{prefix}_perturb_scope"] = 1.0 if perturb else 0.0

            adv_branch_grads = torch.autograd.grad(
                outputs=adv_branch_loss,
                inputs=perturb_params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            ) if perturb_named else []
            method_state["sam_branch_grad_map"] = grad_map_from_grads(perturb_named, adv_branch_grads)
            method_state["sam_branch_only_update"] = True
        finally:
            # Restore immediately: the returned loss is the clean loss, and the
            # optimizer step must be applied from the clean parameter values.
            if perturb:
                model._apply_param_perturb(perturb, sign=-1)

        # Return the clean task loss.  The main training loop will use this loss
        # to update fusion/head normally; branch LoRA gradients are replaced in
        # apply_post_backward_adjustments() with the SAM/ESAM adversarial branch
        # gradients computed above.
        total_loss = base_loss
        logs["sam_branch_grad_replacement_pending"] = float(len(method_state.get("sam_branch_grad_map", {}) or {}))
        logs["train_loss"] = tensor_item(total_loss)
        return total_loss, logs

    if method == "masam_lora":
        # Corrected MASAM-style implementation for the controlled late-fusion
        # setting.  It uses modality-dropout unimodal losses, APS
        # (convergence-speed decay + fusion/unimodal gradient alignment) to
        # select the dominant modality, and MDPS cosine-scaled SAM
        # perturbation only on the selected modality's branch-LoRA tensors.
        base_outputs = model.forward(batch, branch_requires_grad=False)
        if base_outputs.loss is None:
            raise RuntimeError("masam_lora step failed to produce a base loss.")
        base_loss = base_outputs.loss
        z_v, z_t = base_outputs.z_v, base_outputs.z_t
        loss_v_uni, loss_t_uni, _ = compute_modality_dropout_losses(model, z_v, z_t, batch["labels"])

        branch_all = model.get_branch_lora_named_parameters()
        branch_v, branch_t = split_branch_named_parameters(branch_all)
        params_v = [param for _, param in branch_v]
        params_t = [param for _, param in branch_t]

        grads_fuse_v = torch.autograd.grad(
            outputs=base_loss,
            inputs=params_v,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        ) if params_v else []
        grads_fuse_t = torch.autograd.grad(
            outputs=base_loss,
            inputs=params_t,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        ) if params_t else []
        grads_uni_v = torch.autograd.grad(
            outputs=loss_v_uni,
            inputs=params_v,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        ) if params_v else []
        grads_uni_t = torch.autograd.grad(
            outputs=loss_t_uni,
            inputs=params_t,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        ) if params_t else []

        fuse_v_map = grad_map_from_grads(branch_v, grads_fuse_v)
        fuse_t_map = grad_map_from_grads(branch_t, grads_fuse_t)
        uni_v_map = grad_map_from_grads(branch_v, grads_uni_v)
        uni_t_map = grad_map_from_grads(branch_t, grads_uni_t)
        gamma_v = compute_grad_alignment(branch_v, fuse_v_map, uni_v_map, eps=args.eps)
        gamma_t = compute_grad_alignment(branch_t, fuse_t_map, uni_t_map, eps=args.eps)

        beta = max(0.0, min(0.9999, float(getattr(args, "masam_ma_beta", 0.9))))
        alpha = max(0.0, min(1.0, float(getattr(args, "masam_aps_alpha", 0.5))))
        lv = tensor_item(loss_v_uni)
        lt = tensor_item(loss_t_uni)
        if "masam_ma_v" not in method_state:
            method_state["masam_ma_v"] = lv
            method_state["masam_ma_t"] = lt
            method_state["masam_prev_v"] = lv
            method_state["masam_prev_t"] = lt
        ma_v = beta * float(method_state["masam_ma_v"]) + (1.0 - beta) * lv
        ma_t = beta * float(method_state["masam_ma_t"]) + (1.0 - beta) * lt
        decay_v = max(0.0, float(method_state["masam_prev_v"]) - ma_v)
        decay_t = max(0.0, float(method_state["masam_prev_t"]) - ma_t)
        method_state["masam_ma_v"] = ma_v
        method_state["masam_ma_t"] = ma_t
        method_state["masam_prev_v"] = lv
        method_state["masam_prev_t"] = lt

        aps_v = alpha * decay_v + (1.0 - alpha) * gamma_v
        aps_t = alpha * decay_t + (1.0 - alpha) * gamma_t
        dominant = "visual" if aps_v >= aps_t else "text"

        if dominant == "visual":
            dominant_named = branch_v
            dominant_fuse_grads = grads_fuse_v
            gamma = max(0.0, gamma_v)
        else:
            dominant_named = branch_t
            dominant_fuse_grads = grads_fuse_t
            gamma = max(0.0, gamma_t)

        perturb, grad_norm_value = build_param_perturbation(
            dominant_named,
            dominant_fuse_grads,
            radius=float(args.sam_rho) * float(gamma),
            device=device,
            eps=args.eps,
        )
        if perturb:
            model._apply_param_perturb(perturb, sign=1)
        try:
            adv_out = model.forward(batch, branch_requires_grad=False)
            if adv_out.loss is None:
                raise RuntimeError("masam_lora perturbed forward failed to produce a loss.")
            params_dom = [param for _, param in dominant_named]
            adv_grads_dom = torch.autograd.grad(
                outputs=adv_out.loss,
                inputs=params_dom,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            ) if params_dom else []
        finally:
            if perturb:
                model._apply_param_perturb(perturb, sign=-1)

        # Eq. (12): dominant modality uses perturbed fusion gradient + unimodal
        # gradient.  Eq. (13): non-dominant modality uses current fusion
        # gradient + unimodal gradient.
        if dominant == "visual":
            dom_map = grad_map_from_grads(branch_v, adv_grads_dom)
            branch_grad_map = merge_grad_maps_sum(dom_map, uni_v_map, fuse_t_map, uni_t_map)
        else:
            dom_map = grad_map_from_grads(branch_t, adv_grads_dom)
            branch_grad_map = merge_grad_maps_sum(fuse_v_map, uni_v_map, dom_map, uni_t_map)
        method_state["masam_branch_grad_map"] = branch_grad_map
        method_state["masam_dominant_modality"] = dominant

        # Base loss updates fusion/head; branch gradients are replaced after
        # backward with MASAM's modality-specific update directions.
        total_loss = base_loss
        logs = {
            "task_loss": tensor_item(base_loss),
            "masam_loss_v_uni": lv,
            "masam_loss_t_uni": lt,
            "masam_gamma_v": float(gamma_v),
            "masam_gamma_t": float(gamma_t),
            "masam_decay_v": float(decay_v),
            "masam_decay_t": float(decay_t),
            "masam_aps_v": float(aps_v),
            "masam_aps_t": float(aps_t),
            "masam_dominant_is_visual": 1.0 if dominant == "visual" else 0.0,
            "masam_mdps_gamma": float(gamma),
            "masam_adaptive_rho": float(args.sam_rho) * float(gamma),
            "masam_grad_norm": tensor_item(grad_norm_value),
            "masam_perturb_scope": 1.0 if perturb else 0.0,
            "masam_adv_loss": tensor_item(adv_out.loss),
            "train_loss": tensor_item(total_loss),
        }
        return total_loss, logs

    if method == "cofla":
        ablation_mode = str(getattr(args, "cofla_ablation_mode", "full")).strip().lower()
        cofla_train_geometry = str(getattr(args, "cofla_train_geometry", "exact_fcf")).strip().lower()
        valid_ablation_modes = {
            "base",
            "always_on",
            "gate_no_proj",
            "gate_with_proj",
            "branch_proj_only",
            "fusion_proj_only",
            "full",
        }
        if ablation_mode not in valid_ablation_modes:
            raise ValueError(f"Unsupported COFLA ablation mode: {ablation_mode}")

        # Pure L0 baseline: skip geometry probes, gating, and both safeguards.
        if ablation_mode == "base":
            outputs = model.forward(batch, branch_requires_grad=False)
            if outputs.loss is None:
                raise RuntimeError("COFLA base ablation failed to produce a task loss.")
            total_loss = outputs.loss
            method_state["cofla_branch_projection_grad_map"] = {}
            method_state["cofla_projection_grad_map"] = {}
            method_state["cofla_alpha_star"] = 0.0
            method_state["cofla_projection_active"] = False
            method_state["cofla_j_bf"] = 0.0
            method_state["cofla_ablation_mode"] = ablation_mode
            logs = {
                "task_loss": tensor_item(total_loss),
                "train_loss": tensor_item(total_loss),
                "cofla_gate": 0.0,
                "cofla_raw_gate": 0.0,
                "cofla_j_bf": 0.0,
                "cofla_perturbed_loss": tensor_item(total_loss),
                "cofla_robust_loss": tensor_item(total_loss),
                "cofla_robust_delta": 0.0,
                "cofla_use_base_loss": 1.0,
                "cofla_use_perturbed_only": 0.0,
                "cofla_use_gate": 0.0,
                "cofla_branch_projection_enabled": 0.0,
                "cofla_fusion_projection_enabled": 0.0,
                "cofla_projection_active": 0.0,
                "cofla_branch_v_projection_active": 0.0,
                "cofla_branch_t_projection_active": 0.0,
                "cofla_branch_projection_params": 0.0,
                "cofla_train_uses_fast_fcf": 1.0 if cofla_train_geometry == "fast_fcf" else 0.0,
            }
            return total_loss, logs

        probe = model.compute_geometry_probe(batch)

        zero_projection_stats = {
            "alpha_star": 0.0,
            "projection_active": 0.0,
            "projection_dot": 0.0,
            "projection_norm_task": 0.0,
            "projection_norm_obj": 0.0,
        }
        branch_correction_map: Dict[str, torch.Tensor] = {}
        fusion_corr: Dict[str, torch.Tensor] = {}
        stats_v = dict(zero_projection_stats)
        stats_t = dict(zero_projection_stats)
        fusion_stats = dict(zero_projection_stats)

        if cofla_train_geometry == "fast_fcf":
            # COFLA-F: one finite-step fusion probe with a detached first-order
            # branch reference. Branch projection is intentionally disabled.
            geom = compute_fast_train_geometry_from_probe(
                model,
                batch,
                probe,
                rho_v=args.rho_v,
                rho_t=args.rho_t,
                rho_f=args.rho_f,
                eps=args.eps,
                retain_graph=True,
            )
            base_loss = geom["loss"]
            perturbed_loss = geom["loss_f_plus"]
            logs = geometry_tensor_logs(geom)
            logs["cofla_train_uses_fast_fcf"] = 1.0
            logs["cofla_f_no_second_order"] = 1.0
            logs["cofla_f_one_probe_fusion_exact"] = 1.0
            logs["cofla_f_branch_calibrated"] = 1.0
        else:
            # COFLA-E: exact visual, textual, and fusion finite-step probes.
            geom = model.compute_geometry_metrics_from_probe(
                batch,
                probe,
                rho_v=args.rho_v,
                rho_t=args.rho_t,
                rho_f=args.rho_f,
                retain_graph=True,
            )
            base_loss = geom["loss"]
            perturbed_loss = (
                geom["loss_v_plus"] + geom["loss_t_plus"] + geom["loss_f_plus"]
            ) / 3.0
            logs = geometry_tensor_logs(geom)
            logs["cofla_train_uses_fast_fcf"] = 0.0

        j_bf = torch.relu(geom["rfcf"])
        raw_gate = bounded_positive_fcf_gate(geom["rfcf"])
        robust_loss = (1.0 - raw_gate) * base_loss + raw_gate * perturbed_loss

        if ablation_mode == "always_on":
            # L_pert only: no learned gate and no projection safeguards.
            total_loss = perturbed_loss
            effective_gate = torch.ones_like(raw_gate)
        else:
            # gate_no_proj, gate_with_proj, branch_proj_only,
            # fusion_proj_only, and full all use the gated robust objective.
            total_loss = robust_loss
            effective_gate = raw_gate

        request_branch_projection = ablation_mode in {
            "gate_with_proj",
            "branch_proj_only",
            "full",
        }
        request_fusion_projection = ablation_mode in {
            "gate_with_proj",
            "fusion_proj_only",
            "full",
        }

        # COFLA-F deliberately omits branch projection; COFLA-E applies it only
        # when the selected ablation mode enables the branch safeguard.
        use_branch_projection = request_branch_projection and cofla_train_geometry == "exact_fcf"
        use_fusion_projection = request_fusion_projection

        if use_branch_projection:
            branch_named = model.get_branch_lora_named_parameters()
            visual_named, text_named = split_branch_named_parameters(branch_named)
            corr_v, stats_v = build_projection_correction_from_objective(
                named_params=visual_named,
                loss_task=total_loss,
                objective=geom["s_v"],
                objective_value=tensor_item(geom["s_v"]),
                eps=args.eps,
            )
            corr_t, stats_t = build_projection_correction_from_objective(
                named_params=text_named,
                loss_task=total_loss,
                objective=geom["s_t"],
                objective_value=tensor_item(geom["s_t"]),
                eps=args.eps,
            )
            branch_correction_map.update(corr_v)
            branch_correction_map.update(corr_t)

        if use_fusion_projection:
            fusion_named = model.get_fusion_named_parameters()
            fusion_corr, fusion_stats = build_projection_correction_from_objective(
                named_params=fusion_named,
                loss_task=total_loss,
                objective=j_bf,
                objective_value=tensor_item(j_bf),
                eps=args.eps,
            )

        method_state["cofla_branch_projection_grad_map"] = branch_correction_map
        method_state["cofla_projection_grad_map"] = fusion_corr
        method_state["cofla_alpha_star"] = float(fusion_stats["alpha_star"])
        method_state["cofla_projection_active"] = bool(fusion_corr)
        method_state["cofla_j_bf"] = tensor_item(j_bf)
        method_state["cofla_ablation_mode"] = ablation_mode

        logs["task_loss"] = tensor_item(base_loss)
        logs["cofla_gate"] = tensor_item(effective_gate)
        logs["cofla_raw_gate"] = tensor_item(raw_gate)
        logs["cofla_perturbed_loss"] = tensor_item(perturbed_loss)
        logs["cofla_robust_loss"] = tensor_item(robust_loss)
        logs["cofla_robust_delta"] = tensor_item(perturbed_loss - base_loss)
        logs["cofla_use_base_loss"] = 0.0
        logs["cofla_use_perturbed_only"] = 1.0 if ablation_mode == "always_on" else 0.0
        logs["cofla_use_gate"] = 0.0 if ablation_mode == "always_on" else 1.0
        logs["cofla_branch_projection_enabled"] = 1.0 if use_branch_projection else 0.0
        logs["cofla_fusion_projection_enabled"] = 1.0 if use_fusion_projection else 0.0
        logs["cofla_branch_projection_suppressed_for_fast"] = (
            1.0 if request_branch_projection and cofla_train_geometry == "fast_fcf" else 0.0
        )
        logs["loss_align"] = tensor_item(j_bf)
        logs["loss_fuse"] = tensor_item(geom["s_f"])
        logs["cofla_j_bf"] = tensor_item(j_bf)
        logs["cofla_alpha_star"] = float(fusion_stats["alpha_star"])
        logs["cofla_projection_active"] = 1.0 if fusion_corr else 0.0
        logs["cofla_projection_dot"] = float(fusion_stats["projection_dot"])
        logs["cofla_projection_norm_task"] = float(fusion_stats["projection_norm_task"])
        logs["cofla_projection_norm_bf"] = float(fusion_stats["projection_norm_obj"])
        logs["cofla_branch_v_alpha_star"] = float(stats_v["alpha_star"])
        logs["cofla_branch_t_alpha_star"] = float(stats_t["alpha_star"])
        logs["cofla_branch_v_projection_active"] = float(stats_v["projection_active"])
        logs["cofla_branch_t_projection_active"] = float(stats_t["projection_active"])
        logs["cofla_branch_projection_params"] = float(len(branch_correction_map))
        logs["train_loss"] = tensor_item(total_loss)
        return total_loss, logs

    if method == "msam_lora":
        # Corrected M-SAM-style implementation.  We estimate two-modality
        # Shapley contributions with modality dropout, use them as loss weights,
        # select the dominant modality, and perform SAM perturbation only on the
        # dominant branch-LoRA parameters.
        base_outputs = model.forward(batch, branch_requires_grad=False)
        if base_outputs.loss is None:
            raise RuntimeError("msam_lora step failed to produce a base loss.")
        base_loss = base_outputs.loss
        z_v, z_t = base_outputs.z_v, base_outputs.z_t
        loss_v_uni, loss_t_uni, loss_empty = compute_modality_dropout_losses(model, z_v, z_t, batch["labels"])
        w_v, w_t, phi_v, phi_t = compute_two_modality_shapley_weights(
            loss_fuse=base_loss,
            loss_v=loss_v_uni,
            loss_t=loss_t_uni,
            loss_empty=loss_empty,
            eps=float(getattr(args, "msam_shapley_eps", 1e-8)),
        )
        dominant = "visual" if w_v >= w_t else "text"
        branch_v, branch_t = split_branch_named_parameters(model.get_branch_lora_named_parameters())
        dominant_named = branch_v if dominant == "visual" else branch_t
        modulated_loss = base_loss + w_v * loss_v_uni + w_t * loss_t_uni
        grads_dom = torch.autograd.grad(
            outputs=modulated_loss,
            inputs=[param for _, param in dominant_named],
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        ) if dominant_named else []
        perturb, grad_norm_value = build_param_perturbation(
            dominant_named,
            grads_dom,
            radius=float(args.sam_rho),
            device=device,
            eps=args.eps,
        )
        if perturb:
            model._apply_param_perturb(perturb, sign=1)
        try:
            adv_outputs = model.forward(batch, branch_requires_grad=False)
            if adv_outputs.loss is None:
                raise RuntimeError("msam_lora perturbed forward failed to produce a loss.")
            adv_loss_v_uni, adv_loss_t_uni, _ = compute_modality_dropout_losses(
                model, adv_outputs.z_v, adv_outputs.z_t, batch["labels"]
            )
            total_loss = adv_outputs.loss + w_v * adv_loss_v_uni + w_t * adv_loss_t_uni
        except Exception:
            if perturb:
                model._apply_param_perturb(perturb, sign=-1)
            raise
        method_state["pending_param_perturb"] = perturb
        method_state["pending_param_perturb_scope"] = f"msam_dominant_{dominant}"
        logs = {
            "task_loss": tensor_item(base_loss),
            "msam_phi_v": float(phi_v),
            "msam_phi_t": float(phi_t),
            "msam_w_v": float(w_v),
            "msam_w_t": float(w_t),
            "msam_dominant_is_visual": 1.0 if dominant == "visual" else 0.0,
            "msam_loss_v_uni": tensor_item(loss_v_uni),
            "msam_loss_t_uni": tensor_item(loss_t_uni),
            "msam_loss_empty": tensor_item(loss_empty),
            "msam_grad_norm": tensor_item(grad_norm_value),
            "msam_adv_loss": tensor_item(total_loss),
            "train_loss": tensor_item(total_loss),
        }
        return total_loss, logs

    if method == "dgl_lora":
        # Corrected DGL implementation.  The multimodal/fusion loss is computed
        # on detached branch features, so it updates the fusion module/head but
        # does not propagate to branch encoders.  Branch LoRA gradients are
        # provided by modality-dropout unimodal losses and are installed after
        # backward, while unimodal gradients to fusion/head are not used.
        z_v, z_t = model.prepare_branch_activations(batch, requires_grad=False)
        labels = batch["labels"]
        detached_out = model.forward_from_activations(z_v.detach(), z_t.detach(), labels=labels)
        if detached_out.loss is None:
            raise RuntimeError("dgl_lora detached multimodal forward failed to produce a loss.")
        loss_multi_detached = detached_out.loss
        loss_v_uni, loss_t_uni, _ = compute_modality_dropout_losses(model, z_v, z_t, labels)
        branch_v, branch_t = split_branch_named_parameters(model.get_branch_lora_named_parameters())
        grads_uni_v = torch.autograd.grad(
            outputs=loss_v_uni,
            inputs=[param for _, param in branch_v],
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        ) if branch_v else []
        grads_uni_t = torch.autograd.grad(
            outputs=loss_t_uni,
            inputs=[param for _, param in branch_t],
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        ) if branch_t else []
        uni_v_map = grad_map_from_grads(branch_v, grads_uni_v)
        uni_t_map = grad_map_from_grads(branch_t, grads_uni_t)
        method_state["dgl_branch_grad_map"] = merge_grad_maps_sum(uni_v_map, uni_t_map)
        total_loss = loss_multi_detached
        logs = {
            "task_loss": tensor_item(loss_multi_detached),
            "dgl_multimodal_detached_loss": tensor_item(loss_multi_detached),
            "dgl_unimodal_v_loss": tensor_item(loss_v_uni),
            "dgl_unimodal_t_loss": tensor_item(loss_t_uni),
            "dgl_branch_param_count": float(len(branch_v) + len(branch_t)),
            "train_loss": tensor_item(total_loss),
        }
        return total_loss, logs

    raise ValueError(f"Unsupported method: {args.method}")

def apply_post_backward_adjustments(
    model: ControlledLateFusionModel,
    args,
    method_state: Dict[str, Any],
) -> Dict[str, float]:
    logs: Dict[str, float] = {}
    method = str(args.method).strip().lower()

    # Restore any branch-side parameter perturbation used by SAM/ESAM/MASAM
    # baselines after gradients have been computed at the perturbed point.
    pending = method_state.get("pending_param_perturb", {}) or {}
    if pending:
        model._apply_param_perturb(pending, sign=-1)
        logs["restored_param_perturb"] = float(len(pending))
        logs["restored_param_perturb_scope"] = str(method_state.get("pending_param_perturb_scope", ""))
        method_state["pending_param_perturb"] = {}
        method_state["pending_param_perturb_scope"] = ""

    if method == "cofla":
        branch_correction_map = method_state.get("cofla_branch_projection_grad_map", {}) or {}
        branch_applied = 0
        for name, param in model.get_branch_lora_named_parameters():
            correction = branch_correction_map.get(name)
            if correction is None:
                continue
            correction = correction.to(device=param.device, dtype=param.dtype)
            if param.grad is None:
                param.grad = correction.clone()
            else:
                param.grad.add_(correction)
            branch_applied += 1

        correction_map = method_state.get("cofla_projection_grad_map", {}) or {}
        applied = 0
        for name, param in model.get_fusion_named_parameters():
            correction = correction_map.get(name)
            if correction is None:
                continue
            correction = correction.to(device=param.device, dtype=param.dtype)
            if param.grad is None:
                param.grad = correction.clone()
            else:
                param.grad.add_(correction)
            applied += 1
        logs["cofla_alpha_star"] = float(method_state.get("cofla_alpha_star", 0.0))
        logs["cofla_projection_active"] = 1.0 if method_state.get("cofla_projection_active", False) else 0.0
        logs["cofla_projection_params"] = float(applied)
        logs["cofla_branch_projection_params"] = float(branch_applied)
        logs["cofla_j_bf"] = float(method_state.get("cofla_j_bf", 0.0))
        method_state["cofla_projection_grad_map"] = {}
        method_state["cofla_branch_projection_grad_map"] = {}

    if method in {"sam", "sam_lora", "esam_lora"}:
        # SAM/ESAM are branch-only baselines: after clean backward has produced
        # gradients for all trainable parameters, replace only branch LoRA
        # gradients with their adversarial branch gradients.  Fusion/head keep
        # their clean-loss gradients.
        branch_grad_map = method_state.get("sam_branch_grad_map", {}) or {}
        replaced = 0
        for name, param in model.get_branch_lora_named_parameters():
            corrected = branch_grad_map.get(name)
            if corrected is None:
                continue
            corrected = corrected.to(device=param.device, dtype=param.dtype)
            param.grad = corrected.clone()
            replaced += 1
        logs["sam_replaced_branch_grad_count"] = float(replaced)
        logs["sam_branch_only_update"] = 1.0 if method_state.get("sam_branch_only_update", False) else 0.0
        method_state["sam_branch_grad_map"] = {}
        method_state["sam_branch_only_update"] = False

    if method == "masam_lora":
        masam_grad_map = method_state.get("masam_branch_grad_map", {}) or {}
        replaced = 0
        for name, param in model.get_branch_lora_named_parameters():
            corrected = masam_grad_map.get(name)
            if corrected is None:
                continue
            corrected = corrected.to(device=param.device, dtype=param.dtype)
            param.grad = corrected.clone()
            replaced += 1
        logs["masam_replaced_branch_grad_count"] = float(replaced)
        logs["masam_dominant_modality"] = str(method_state.get("masam_dominant_modality", ""))
        method_state["masam_branch_grad_map"] = {}

    if method == "dgl_lora":
        branch_grad_map = method_state.get("dgl_branch_grad_map", {}) or {}
        replaced = 0
        for name, param in model.get_branch_lora_named_parameters():
            corrected = branch_grad_map.get(name)
            if corrected is None:
                continue
            corrected = corrected.to(device=param.device, dtype=param.dtype)
            # DGL replaces multimodal encoder gradients with unimodal
            # modality-dropout gradients.  Fusion/head gradients already come
            # from the detached multimodal loss.
            param.grad = corrected.clone()
            replaced += 1
        logs["dgl_replaced_branch_grad_count"] = float(replaced)
        method_state["dgl_branch_grad_map"] = {}
    return logs

def flatten_for_logging(record: Dict[str, Any]) -> Dict[str, float | int | str]:
    flat: Dict[str, float | int | str] = {}
    for key, value in record.items():
        if isinstance(value, float):
            flat[key] = value
        elif isinstance(value, int):
            flat[key] = value
        else:
            flat[key] = str(value)
    return flat


def save_trainable_state(model: ControlledLateFusionModel, path: str, metadata: Dict[str, Any]) -> None:
    torch.save(model.export_trainable_state(metadata), path)


def load_trainable_state(model: ControlledLateFusionModel, path: str) -> Dict[str, Any]:
    state = torch.load(path, map_location="cpu")
    return model.load_trainable_state(state, strict=False)


def save_predictions_csv(
    path: str,
    *,
    dataset: str,
    prediction_rows: Sequence[Dict[str, Any]],
    label_names: Sequence[str],
) -> None:
    ensure_dir(os.path.dirname(path))
    dataset_key = str(dataset).strip().lower()

    if dataset_key == "hateful_memes":
        fieldnames = ["id", "label", "pred", "score"]
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in prediction_rows:
                writer.writerow(
                    {
                        "id": row["id"],
                        "label": int(row["label"]),
                        "pred": int(row["pred"]),
                        "score": float(row["score"]),
                    }
                )
        return

    sanitized = [sanitize_label_name(name) for name in label_names]
    fieldnames = ["id"]
    for name in sanitized:
        fieldnames.extend([f"label__{name}", f"pred__{name}", f"score__{name}"])

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in prediction_rows:
            out = {"id": row["id"]}
            labels = row["labels"]
            preds = row["preds"]
            scores = row["scores"]
            for name, label_value, pred_value, score_value in zip(sanitized, labels, preds, scores):
                out[f"label__{name}"] = int(label_value)
                out[f"pred__{name}"] = int(pred_value)
                out[f"score__{name}"] = float(score_value)
            writer.writerow(out)


def save_geometry_text(path: str, metrics: Dict[str, float]) -> None:
    lines = []
    for key in GEOMETRY_KEYS:
        value = metrics.get(key, float("nan"))
        if isinstance(value, float) and math.isnan(value):
            lines.append(f"{key}: nan")
        else:
            lines.append(f"{key}: {float(value):.6f}")
    save_text("\n".join(lines) + "\n", path)


def evaluate(
    model: ControlledLateFusionModel,
    dataloader: DataLoader,
    args,
    *,
    device: torch.device,
    split: str,
    compute_geometry: bool,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    model.eval()
    labels: List[Any] = []
    probs: List[Any] = []
    prediction_rows: List[Dict[str, Any]] = []
    geometry_records: List[Dict[str, float]] = []

    for raw_batch in tqdm(dataloader, desc=f"eval:{split}"):
        batch = move_batch_to_device(raw_batch, device)
        with torch.no_grad():
            with maybe_autocast(device, args.fp16):
                outputs = model(batch)
        batch_probs, batch_preds = model.predict_from_logits(outputs.logits, threshold=args.mmimdb_threshold)
        if args.dataset == "hateful_memes":
            batch_labels = batch["labels"].detach().cpu().to(torch.int64).tolist()
            probs_list = batch_probs.detach().cpu().tolist()
            preds_list = batch_preds.detach().cpu().tolist()
            labels.extend(batch_labels)
            probs.extend(probs_list)
            for sample_id, label_value, pred_value, score_value in zip(
                batch["sample_ids"], batch_labels, preds_list, probs_list
            ):
                prediction_rows.append(
                    {
                        "id": sample_id,
                        "label": int(label_value),
                        "pred": int(pred_value),
                        "score": float(score_value),
                    }
                )
        else:
            batch_labels = batch["labels"].detach().cpu().to(torch.int64).tolist()
            probs_list = batch_probs.detach().cpu().tolist()
            preds_list = batch_preds.detach().cpu().tolist()
            labels.extend(batch_labels)
            probs.extend(probs_list)
            for sample_id, label_value, pred_value, score_value in zip(
                batch["sample_ids"], batch_labels, preds_list, probs_list
            ):
                prediction_rows.append(
                    {
                        "id": sample_id,
                        "labels": [int(x) for x in label_value],
                        "preds": [int(x) for x in pred_value],
                        "scores": [float(x) for x in score_value],
                    }
                )

        if compute_geometry:
            with torch.enable_grad():
                with maybe_autocast(device, args.fp16):
                    eval_rho_v = args.rho_v if args.eval_rho_v is None else args.eval_rho_v
                    eval_rho_t = args.rho_t if args.eval_rho_t is None else args.eval_rho_t
                    eval_rho_f = args.rho_f if args.eval_rho_f is None else args.eval_rho_f
                    geometry_metric = str(getattr(args, "geometry_metric", "fast_fcf")).strip().lower()
                    geom: Dict[str, torch.Tensor] = {}
                    if geometry_metric in {"fast_fcf", "both"}:
                        geom.update(
                            model.compute_fast_fcf_metrics(
                                batch,
                                rho_v=eval_rho_v,
                                rho_t=eval_rho_t,
                                rho_f=eval_rho_f,
                                retain_graph=False,
                            )
                        )
                    if geometry_metric in {"exact_fcf", "both"}:
                        geom.update(
                            model.compute_geometry_metrics(
                                batch,
                                rho_v=eval_rho_v,
                                rho_t=eval_rho_t,
                                rho_f=eval_rho_f,
                                retain_graph=False,
                            )
                        )
            geometry_records.append({key: tensor_item(geom[key]) for key in GEOMETRY_KEYS if key in geom})

    metrics = compute_controlled_metrics(
        args.dataset,
        labels,
        probs,
        threshold=args.mmimdb_threshold,
    )
    metrics.update(mean_dict(geometry_records, GEOMETRY_KEYS))
    return metrics, prediction_rows


def build_summary_text(
    *,
    args,
    best_epoch: int,
    best_val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    trainable_report: Dict[str, Any],
    elapsed: Dict[str, Any],
    peak_gpu_memory_mb: float,
) -> str:
    visual_branch, text_branch = encoder_setup_for_dataset(args.dataset)
    lines = [
        f"dataset: {args.dataset}",
        f"visual branch: CLIP-ViT",
        f"text branch: {'CLIP-text' if text_branch == 'clip' else 'RoBERTa-base'}",
        f"encoder setup: {visual_branch}+{text_branch}",
        f"fusion type: {args.fusion_type}",
        f"method: {args.method}",
        f"optimizer: {getattr(args, 'optimizer', 'adamw')}",
        f"branch tuning mode: {getattr(args, 'branch_tuning_mode', 'frozen')}",
        f"branch lora r: {getattr(args, 'branch_lora_r', 0)}",
        f"branch lora alpha: {getattr(args, 'branch_lora_alpha', 0)}",
        f"branch lora dropout: {getattr(args, 'branch_lora_dropout', 0.0)}",
        f"seed: {args.seed}",
        f"best epoch: {best_epoch}",
        f"best validation metric ({primary_metric_name(args.dataset)}): {select_primary_metric(args.dataset, best_val_metrics):.6f}",
        f"test metric ({primary_metric_name(args.dataset)}): {select_primary_metric(args.dataset, test_metrics):.6f}",
    ]

    if args.dataset == "hateful_memes":
        lines.append(f"AUROC: {test_metrics.get('auroc', float('nan')):.6f}" if not math.isnan(test_metrics.get("auroc", float("nan"))) else "AUROC: nan")
        lines.append(f"Accuracy: {test_metrics.get('accuracy', float('nan')):.6f}" if not math.isnan(test_metrics.get("accuracy", float("nan"))) else "Accuracy: nan")
    else:
        lines.append(f"Macro-F1: {test_metrics.get('macro_f1', float('nan')):.6f}")
        lines.append(f"Micro-F1: {test_metrics.get('micro_f1', float('nan')):.6f}")

    for key in ["fast_s_v", "fast_s_t", "fast_s_branch", "fast_s_f", "fast_fcf", "fast_rfcf"]:
        value = best_val_metrics.get(key, float("nan"))
        if isinstance(value, float) and math.isnan(value):
            lines.append(f"mean {key}: nan")
        else:
            lines.append(f"mean {key}: {float(value):.6f}")

    if str(getattr(args, "geometry_metric", "fast_fcf")) in {"exact_fcf", "both"}:
        for key in ["exact_s_v", "exact_s_t", "exact_s_branch", "exact_s_f", "exact_fcf", "exact_rfcf"]:
            value = best_val_metrics.get(key, float("nan"))
            if isinstance(value, float) and math.isnan(value):
                lines.append(f"mean {key}: nan")
            else:
                lines.append(f"mean {key}: {float(value):.6f}")

    lines.extend(
        [
            f"trainable parameter count: {int(trainable_report['trainable_parameters'])}",
            f"total wall-clock time: {elapsed['elapsed_hms']} ({elapsed['elapsed_seconds']}s)",
            f"peak gpu memory mb: {peak_gpu_memory_mb:.2f}" if not math.isnan(peak_gpu_memory_mb) else "peak gpu memory mb: nan",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.auto_rho_f is None:
        args.auto_rho_f = args.method == "cofla"
    if not str(getattr(args, "exp_name", "")).strip():
        args.exp_name = build_experiment_name(args)

    experiment_start = time.perf_counter()
    set_seed(args.seed, deterministic=True)
    device = resolve_device(args.device)
    dirs = build_experiment_dirs(args.result_root, args.exp_name)
    logger = setup_logger(os.path.join(dirs["logs"], "train.log"), name=f"controlled_{args.exp_name}")

    if str(args.method).strip().lower() == "cofla":
        ablation_mode = str(getattr(args, "cofla_ablation_mode", "full")).strip().lower()
        train_geometry = str(getattr(args, "cofla_train_geometry", "exact_fcf")).strip().lower()
        if ablation_mode == "base":
            effective_loss = "L0"
        elif ablation_mode == "always_on":
            effective_loss = "Lpert"
        else:
            effective_loss = "Lrob"
        branch_projection_enabled = (
            ablation_mode in {"gate_with_proj", "branch_proj_only", "full"}
            and train_geometry == "exact_fcf"
        )
        fusion_projection_enabled = ablation_mode in {
            "gate_with_proj", "fusion_proj_only", "full"
        }
        gate_enabled = ablation_mode not in {"base", "always_on"}
        args.cofla_effective_loss = effective_loss
        args.cofla_effective_gate = bool(gate_enabled)
        args.cofla_effective_branch_projection = bool(branch_projection_enabled)
        args.cofla_effective_fusion_projection = bool(fusion_projection_enabled)
        logger.info(
            "[COFLA-Ablation] mode=%s geometry=%s loss=%s gate=%s branch_proj=%s fusion_proj=%s",
            ablation_mode,
            train_geometry,
            effective_loss,
            gate_enabled,
            branch_projection_enabled,
            fusion_projection_enabled,
        )

    model, image_processor, tokenizer = build_model_and_tokenizers(args, device)
    label_names, _, _, _, train_loader, val_loader, test_loader = build_dataloaders(
        args,
        image_processor,
        tokenizer,
        logger,
    )

    if len(train_loader) == 0:
        raise RuntimeError("Training dataloader is empty.")
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    maybe_run_rho_f_calibration(
        model,
        train_loader,
        args,
        device=device,
        dirs=dirs,
        logger=logger,
    )

    optimizer_param_groups = model.build_optimizer_param_groups(
        learning_rate=args.learning_rate,
        branch_lora_lr=args.branch_lora_lr,
        weight_decay=args.weight_decay,
    )
    if not optimizer_param_groups:
        raise RuntimeError("No trainable parameters found for the selected controlled adaptation scope.")
    optimizer_name = str(getattr(args, "optimizer", "adamw")).strip().lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(optimizer_param_groups, lr=args.learning_rate, weight_decay=0.0)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            optimizer_param_groups,
            lr=args.learning_rate,
            momentum=float(getattr(args, "sgd_momentum", 0.9)),
            weight_decay=0.0,
            nesterov=bool(getattr(args, "sgd_nesterov", False)),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    trainable_report = model.describe_trainable_state(preview_limit=50)
    logger.info("Trainable report: %s", json.dumps(trainable_report, ensure_ascii=False))

    config_blob = vars(args).copy()
    config_blob["device_resolved"] = str(device)
    config_blob["label_names"] = label_names
    config_blob["trainable_report"] = trainable_report
    config_blob["adaptive_calibration"] = rho_f_calibration_payload(args)
    save_json(config_blob, os.path.join(dirs["exp_dir"], "config.json"))

    best_val_metric = float("-inf")
    best_val_metrics: Dict[str, float] = {}
    best_epoch = -1
    best_val_predictions: List[Dict[str, Any]] = []
    method_state: Dict[str, Any] = {
        "cofla_branch_projection_grad_map": {},
        "cofla_projection_grad_map": {},
        "cofla_alpha_star": 0.0,
        "cofla_projection_active": False,
        "cofla_j_bf": 0.0,
    }
    global_step = 0

    for epoch in range(args.num_train_epochs):
        model.train()
        epoch_logs: List[Dict[str, float]] = []
        progress = tqdm(train_loader, desc=f"train:{epoch}")
        for raw_batch in progress:
            batch = move_batch_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)

            with maybe_autocast(device, args.fp16):
                total_loss, step_logs = run_train_step(model, batch, args, method_state, device)
            total_loss.backward()
            step_logs.update(apply_post_backward_adjustments(model, args, method_state))

            grad_norm = compute_grad_norm(model.get_trainable_parameters())
            if args.max_grad_norm and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), args.max_grad_norm)
            optimizer.step()

            global_step += 1
            step_logs["lr"] = float(optimizer.param_groups[0]["lr"])
            if len(optimizer.param_groups) > 1:
                for group_index, group in enumerate(optimizer.param_groups):
                    group_name = str(group.get("name", f"group{group_index}"))
                    step_logs[f"lr_{group_name}"] = float(group.get("lr", optimizer.param_groups[0]["lr"]))
            step_logs["grad_norm"] = float(grad_norm)
            step_logs["epoch"] = int(epoch)
            step_logs["global_step"] = int(global_step)
            epoch_logs.append(step_logs)
            append_jsonl({"type": "train", **flatten_for_logging(step_logs)}, os.path.join(dirs["metrics"], "metrics.jsonl"))
            progress.set_postfix({"loss": f"{step_logs['train_loss']:.4f}"})

        mean_train = mean_dict(epoch_logs, list(epoch_logs[0].keys()) if epoch_logs else [])
        logger.info("Epoch %s train summary: %s", epoch, json.dumps(mean_train, ensure_ascii=False))

        val_metrics, val_predictions = evaluate(
            model,
            val_loader,
            args,
            device=device,
            split="val",
            compute_geometry=args.val_compute_geometry,
        )
        append_jsonl(
            {"type": "eval", "split": "val", "epoch": epoch, **flatten_for_logging(val_metrics)},
            os.path.join(dirs["metrics"], "metrics.jsonl"),
        )
        logger.info("Epoch %s val metrics: %s", epoch, json.dumps(val_metrics, ensure_ascii=False))

        latest_state_path = os.path.join(dirs["latest_checkpoint"], "trainable_state.pt")
        save_trainable_state(
            model,
            latest_state_path,
            {"epoch": epoch, "global_step": global_step, "best_val_metric": best_val_metric},
        )

        current_metric = select_primary_metric(args.dataset, val_metrics)
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            best_val_metrics = deepcopy(val_metrics)
            best_epoch = epoch
            best_val_predictions = deepcopy(val_predictions)
            best_state_path = os.path.join(dirs["best_checkpoint"], "trainable_state.pt")
            save_trainable_state(
                model,
                best_state_path,
                {"epoch": epoch, "global_step": global_step, "best_val_metric": best_val_metric},
            )
            save_predictions_csv(
                os.path.join(dirs["predictions"], "val_predictions.csv"),
                dataset=args.dataset,
                prediction_rows=best_val_predictions,
                label_names=label_names,
            )
            save_geometry_text(
                os.path.join(dirs["metrics"], "geometry_eval_val_best.txt"),
                best_val_metrics,
            )

    if best_epoch < 0:
        val_metrics, val_predictions = evaluate(
            model,
            val_loader,
            args,
            device=device,
            split="val",
            compute_geometry=args.val_compute_geometry,
        )
        best_val_metrics = deepcopy(val_metrics)
        best_val_predictions = deepcopy(val_predictions)
        best_epoch = max(args.num_train_epochs - 1, 0)
        save_predictions_csv(
            os.path.join(dirs["predictions"], "val_predictions.csv"),
            dataset=args.dataset,
            prediction_rows=best_val_predictions,
            label_names=label_names,
        )
        save_geometry_text(
            os.path.join(dirs["metrics"], "geometry_eval_val_best.txt"),
            best_val_metrics,
        )
        best_state_path = os.path.join(dirs["best_checkpoint"], "trainable_state.pt")
        save_trainable_state(
            model,
            best_state_path,
            {"epoch": best_epoch, "global_step": global_step, "best_val_metric": select_primary_metric(args.dataset, best_val_metrics)},
        )

    best_state_path = os.path.join(dirs["best_checkpoint"], "trainable_state.pt")
    load_info = load_trainable_state(model, best_state_path)
    logger.info("Loaded best checkpoint: %s", json.dumps(load_info, ensure_ascii=False))

    test_metrics, test_predictions = evaluate(
        model,
        test_loader,
        args,
        device=device,
        split="test",
        compute_geometry=args.test_compute_geometry,
    )
    append_jsonl(
        {"type": "eval", "split": "test", "epoch": best_epoch, **flatten_for_logging(test_metrics)},
        os.path.join(dirs["metrics"], "metrics.jsonl"),
    )
    logger.info("Test metrics: %s", json.dumps(test_metrics, ensure_ascii=False))

    save_predictions_csv(
        os.path.join(dirs["predictions"], "test_predictions.csv"),
        dataset=args.dataset,
        prediction_rows=test_predictions,
        label_names=label_names,
    )

    final_metrics = {
        "dataset": args.dataset,
        "method": args.method,
        "fusion_type": args.fusion_type,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "label_names": label_names,
        "trainable_parameters": int(trainable_report["trainable_parameters"]),
        "primary_metric_name": primary_metric_name(args.dataset),
        "adaptive_calibration": rho_f_calibration_payload(args),
    }
    save_json(final_metrics, os.path.join(dirs["metrics"], "final_metrics.json"))

    elapsed = build_elapsed_record(time.perf_counter() - experiment_start)
    peak_gpu_memory_mb = float("nan")
    if torch.cuda.is_available() and device.type == "cuda":
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)

    summary = build_summary_text(
        args=args,
        best_epoch=best_epoch,
        best_val_metrics=best_val_metrics,
        test_metrics=test_metrics,
        trainable_report=trainable_report,
        elapsed=elapsed,
        peak_gpu_memory_mb=peak_gpu_memory_mb,
    )
    save_text(summary, os.path.join(dirs["exp_dir"], "summary.txt"))
    logger.info("Saved summary to %s", os.path.join(dirs["exp_dir"], "summary.txt"))

    # Remove checkpoint artifacts after all metrics, predictions, and summary files have been saved.
    checkpoint_dir = dirs.get("checkpoints", os.path.join(dirs["exp_dir"], "checkpoints"))
    if os.path.isdir(checkpoint_dir):
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        logger.info("Removed checkpoint directory: %s", checkpoint_dir)


# -----------------------------------------------------------------------------
# VLM compatibility dispatcher
# -----------------------------------------------------------------------------
# This file remains backward-compatible with the original controlled
# CLIP/RoBERTa late-fusion experiments.  When VLM arguments such as
# --model_family/--model_name/--model_path are present, it reuses the project's
# standard train.py data/model/checkpoint pipeline and installs VLM-compatible
# COFLA ablation methods at runtime.  No change to train.py is required.


def _cli_has_option(argv: Sequence[str], option: str) -> bool:
    return any(token == option or token.startswith(option + "=") for token in argv)


def _cli_pop_option(argv: List[str], option: str, default: Optional[str] = None) -> Tuple[List[str], Optional[str]]:
    """Remove one --option value (supports '--x y' and '--x=y')."""
    output: List[str] = []
    value = default
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == option:
            if i + 1 >= len(argv):
                raise ValueError(f"Missing value after {option}")
            value = argv[i + 1]
            i += 2
            continue
        prefix = option + "="
        if token.startswith(prefix):
            value = token[len(prefix):]
            i += 1
            continue
        output.append(token)
        i += 1
    return output, value


def _cli_replace_option(argv: List[str], old: str, new: str) -> List[str]:
    output: List[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == old:
            if i + 1 >= len(argv):
                raise ValueError(f"Missing value after {old}")
            output.extend([new, argv[i + 1]])
            i += 2
            continue
        prefix = old + "="
        if token.startswith(prefix):
            output.append(new + "=" + token[len(prefix):])
            i += 1
            continue
        output.append(token)
        i += 1
    return output


def _looks_like_vlm_cli(argv: Sequence[str]) -> bool:
    vlm_flags = ("--model_family", "--model_name", "--model_path", "--trainable_scope", "--quantization")
    return any(_cli_has_option(argv, flag) for flag in vlm_flags)


def _vlm_scalar(method, value) -> float:
    return method._float(value)


def _vlm_geometry_logs(
    method,
    geom: Dict[str, torch.Tensor],
    *,
    base_loss: torch.Tensor,
    perturbed_loss: torch.Tensor,
    robust_loss: torch.Tensor,
    total_loss: torch.Tensor,
    raw_gate: torch.Tensor,
    effective_gate: torch.Tensor,
    mode: str,
    uses_fast: bool,
    stats_v: Optional[Dict[str, float]] = None,
    stats_t: Optional[Dict[str, float]] = None,
    stats_f: Optional[Dict[str, float]] = None,
    branch_correction_count: int = 0,
    fusion_correction_count: int = 0,
) -> Dict[str, float]:
    zero = {
        "alpha_star": 0.0,
        "projection_active": 0.0,
        "projection_dot": 0.0,
        "projection_norm_task": 0.0,
        "projection_norm_obj": 0.0,
    }
    stats_v = dict(zero if stats_v is None else stats_v)
    stats_t = dict(zero if stats_t is None else stats_t)
    stats_f = dict(zero if stats_f is None else stats_f)
    z = base_loss.new_tensor(0.0)
    rfcf = geom.get("rfcf", z)
    j_bf = torch.relu(rfcf)
    s_f = geom.get("s_f", geom.get("s_f_b", z))
    logs = {
        "train_loss": _vlm_scalar(method, total_loss),
        "task_loss": _vlm_scalar(method, base_loss),
        "loss_align": _vlm_scalar(method, j_bf),
        "loss_fuse": _vlm_scalar(method, s_f),
        "s_v_b": _vlm_scalar(method, geom.get("s_v_b", geom.get("s_v", z))),
        "s_t_b": _vlm_scalar(method, geom.get("s_t_b", geom.get("s_t", z))),
        "s_f_b": _vlm_scalar(method, geom.get("s_f_b", s_f)),
        "s_branch_b": _vlm_scalar(method, geom.get("s_branch_b", geom.get("s_branch", z))),
        "s_v": _vlm_scalar(method, geom.get("s_v", geom.get("s_v_b", z))),
        "s_t": _vlm_scalar(method, geom.get("s_t", geom.get("s_t_b", z))),
        "s_f": _vlm_scalar(method, s_f),
        "s_branch": _vlm_scalar(method, geom.get("s_branch", geom.get("s_branch_b", z))),
        "fcf": _vlm_scalar(method, geom.get("fcf", z)),
        "rfcf": _vlm_scalar(method, rfcf),
        "rlin_fcf": _vlm_scalar(method, geom.get("rlin_fcf", geom.get("r_lin_fcf", rfcf))),
        "gv_norm": _vlm_scalar(method, geom.get("gv_norm", geom.get("grad_zv_norm", z))),
        "gt_norm": _vlm_scalar(method, geom.get("gt_norm", geom.get("grad_zt_norm", z))),
        "gf_norm": _vlm_scalar(method, geom.get("gf_norm", geom.get("grad_fusion_norm", z))),
        "cofla_train_uses_fast_fcf": 1.0 if uses_fast else 0.0,
        "cofla_gate": _vlm_scalar(method, effective_gate),
        "cofla_raw_gate": _vlm_scalar(method, raw_gate),
        "cofla_perturbed_loss": _vlm_scalar(method, perturbed_loss),
        "cofla_robust_loss": _vlm_scalar(method, robust_loss),
        "cofla_robust_delta": _vlm_scalar(method, perturbed_loss - base_loss),
        "cofla_j_bf": _vlm_scalar(method, j_bf),
        "cofla_use_base_loss": 1.0 if mode == "base" else 0.0,
        "cofla_use_perturbed_only": 1.0 if mode == "always_on" else 0.0,
        "cofla_use_gate": 0.0 if mode in {"base", "always_on"} else 1.0,
        "cofla_alpha_star": float(stats_f.get("alpha_star", 0.0)),
        "cofla_projection_active": float(stats_f.get("projection_active", 0.0)),
        "cofla_projection_dot": float(stats_f.get("projection_dot", 0.0)),
        "cofla_projection_norm_task": float(stats_f.get("projection_norm_task", 0.0)),
        "cofla_projection_norm_bf": float(stats_f.get("projection_norm_obj", 0.0)),
        "cofla_projection_params": float(fusion_correction_count),
        "cofla_branch_v_alpha_star": float(stats_v.get("alpha_star", 0.0)),
        "cofla_branch_t_alpha_star": float(stats_t.get("alpha_star", 0.0)),
        "cofla_branch_v_projection_active": float(stats_v.get("projection_active", 0.0)),
        "cofla_branch_t_projection_active": float(stats_t.get("projection_active", 0.0)),
        "cofla_branch_projection_params": float(branch_correction_count),
    }
    return logs


def _install_vlm_ablation_methods(vlm_train, mode: str, geometry: str) -> None:
    """Install the requested ablation under METHOD_MAP['cofla'].

    Keeping the public method name as 'cofla' is intentional: VLM wrappers use
    it to select the same matched-LoRA trainable scope as the full method.
    """
    from methods.base_method import StepOutput
    from methods.cofla import COFLAMethod, bounded_positive_fcf_gate
    from methods.fast_cofla import FastCOFLAMethod
    from methods.vanilla_lora import VanillaLoRAMethod

    mode = str(mode).strip().lower()
    geometry = str(geometry).strip().lower()
    valid_modes = {
        "base", "always_on", "gate_no_proj", "gate_with_proj",
        "branch_proj_only", "fusion_proj_only", "full",
    }
    if mode not in valid_modes:
        raise ValueError(f"Unsupported VLM COFLA ablation mode: {mode}")
    if geometry not in {"exact_fcf", "fast_fcf"}:
        raise ValueError(f"Unsupported VLM COFLA training geometry: {geometry}")

    class VLMExactCOFLAAblation(COFLAMethod):
        name = "cofla"

        def training_step(self, wrapper, batch):
            selected_mode = str(getattr(self.args, "cofla_ablation_mode", mode)).strip().lower()
            if selected_mode == "base":
                outputs, _ = wrapper.compute_base_loss(batch, capture_layer0=False)
                base_loss = outputs.loss.float()
                self._branch_correction_map = {}
                self._fusion_correction_map = {}
                self._alpha_star = 0.0
                self._projection_active = False
                self._j_bf = 0.0
                zero = base_loss.new_tensor(0.0)
                geom = {"fcf": zero, "rfcf": zero, "s_v": zero, "s_t": zero, "s_f": zero, "s_branch": zero}
                logs = _vlm_geometry_logs(
                    self, geom, base_loss=base_loss, perturbed_loss=base_loss,
                    robust_loss=base_loss, total_loss=base_loss,
                    raw_gate=zero, effective_gate=zero, mode=selected_mode,
                    uses_fast=False,
                )
                self._last_logs = dict(logs)
                return StepOutput(loss=base_loss, logs=logs)

            geom, _ = self._compute_training_geometry(wrapper, batch)
            base_loss = geom.get("loss", geom.get("probe_loss"))
            if base_loss is None:
                raise RuntimeError("VLM COFLA ablation requires a scalar clean task loss.")
            base_loss = base_loss.float()
            rfcf = geom["rfcf"]
            j_bf = torch.relu(rfcf)
            raw_gate = bounded_positive_fcf_gate(rfcf)
            perturbed_loss = self._perturbed_loss_from_geom(geom, base_loss)
            robust_loss = (1.0 - raw_gate) * base_loss + raw_gate * perturbed_loss
            if selected_mode == "always_on":
                total_loss = perturbed_loss
                effective_gate = torch.ones_like(raw_gate)
            else:
                total_loss = robust_loss
                effective_gate = raw_gate

            request_branch = selected_mode in {"gate_with_proj", "branch_proj_only", "full"}
            request_fusion = selected_mode in {"gate_with_proj", "fusion_proj_only", "full"}
            self._branch_correction_map = {}
            self._fusion_correction_map = {}
            zero_stats = {
                "alpha_star": 0.0, "projection_active": 0.0,
                "projection_dot": 0.0, "projection_norm_task": 0.0,
                "projection_norm_obj": 0.0,
            }
            stats_v = dict(zero_stats)
            stats_t = dict(zero_stats)
            stats_f = dict(zero_stats)

            if request_branch:
                visual_named, text_named = self._grouped_branch_params(wrapper)
                s_v = geom.get("s_v", geom.get("s_v_b", base_loss.new_tensor(0.0)))
                s_t = geom.get("s_t", geom.get("s_t_b", base_loss.new_tensor(0.0)))
                corr_v, stats_v = self._build_projection_for_objective(
                    named_params=visual_named, loss_task=total_loss,
                    objective=s_v, objective_value=self._float(s_v),
                )
                corr_t, stats_t = self._build_projection_for_objective(
                    named_params=text_named, loss_task=total_loss,
                    objective=s_t, objective_value=self._float(s_t),
                )
                self._branch_correction_map.update(corr_v)
                self._branch_correction_map.update(corr_t)

            if request_fusion:
                fusion_named = self._named_fusion_params(wrapper)
                self._fusion_correction_map, stats_f = self._build_projection_for_objective(
                    named_params=fusion_named, loss_task=total_loss,
                    objective=j_bf, objective_value=self._float(j_bf),
                )

            self._alpha_star = float(stats_f["alpha_star"])
            self._projection_active = bool(self._fusion_correction_map)
            self._j_bf = self._float(j_bf)
            logs = _vlm_geometry_logs(
                self, geom, base_loss=base_loss, perturbed_loss=perturbed_loss,
                robust_loss=robust_loss, total_loss=total_loss,
                raw_gate=raw_gate, effective_gate=effective_gate,
                mode=selected_mode, uses_fast=False,
                stats_v=stats_v, stats_t=stats_t, stats_f=stats_f,
                branch_correction_count=len(self._branch_correction_map),
                fusion_correction_count=len(self._fusion_correction_map),
            )
            self._last_logs = dict(logs)
            return StepOutput(loss=total_loss, logs=logs)

    class VLMFastCOFLAAblation(FastCOFLAMethod):
        name = "cofla"

        def training_step(self, wrapper, batch):
            selected_mode = str(getattr(self.args, "cofla_ablation_mode", mode)).strip().lower()
            if selected_mode == "base":
                outputs, _ = wrapper.compute_base_loss(batch, capture_layer0=False)
                loss = outputs.loss.float()
                self._branch_correction_map = {}
                self._fusion_correction_map = {}
                self._alpha_star = 0.0
                self._projection_active = False
                self._j_bf = 0.0
                zero = loss.new_tensor(0.0)
                geom = {"fcf": zero, "rfcf": zero, "s_v": zero, "s_t": zero, "s_f": zero, "s_branch": zero}
                logs = _vlm_geometry_logs(
                    self, geom, base_loss=loss, perturbed_loss=loss,
                    robust_loss=loss, total_loss=loss,
                    raw_gate=zero, effective_gate=zero, mode=selected_mode,
                    uses_fast=True,
                )
                self._last_logs = dict(logs)
                return StepOutput(loss=loss, logs=logs)

            if selected_mode != "always_on":
                output = super().training_step(wrapper, batch)
                # COFLA-F has only a fusion-side projection.  Disable it for the
                # no-projection and branch-only controls.
                if selected_mode in {"gate_no_proj", "branch_proj_only"}:
                    self._fusion_correction_map = {}
                    self._alpha_star = 0.0
                    self._projection_active = False
                    output.logs["cofla_alpha_star"] = 0.0
                    output.logs["cofla_projection_active"] = 0.0
                    output.logs["cofla_projection_params"] = 0.0
                output.logs["cofla_use_base_loss"] = 0.0
                output.logs["cofla_use_perturbed_only"] = 0.0
                output.logs["cofla_use_gate"] = 1.0
                output.logs["cofla_raw_gate"] = output.logs.get("cofla_gate", 0.0)
                return output

            # FAST-FCF always-on control: reproduce the one-probe COFLA-F
            # geometry, but optimize L_f^+ directly and apply no projection.
            outputs, cache = wrapper.compute_base_loss(batch, capture_layer0=True)
            base_loss = outputs.loss.float()
            hidden = cache.get("hidden")
            if hidden is None:
                raise RuntimeError("Failed to capture layer-0 hidden states for FAST-FCF ablation.")
            eps = self._safe_eps(self.args)
            device = base_loss.device
            dtype = base_loss.dtype
            fusion_named = self._named_fusion_params(wrapper)
            grad_inputs = [hidden] + [param for _, param in fusion_named]
            grads = torch.autograd.grad(
                outputs=base_loss, inputs=grad_inputs, retain_graph=True,
                create_graph=False, allow_unused=True,
            )
            hidden_grad = grads[0] if grads[0] is not None else torch.zeros_like(hidden)
            fusion_grads = list(grads[1:])
            image_mask, text_mask = wrapper._build_branch_masks(batch, hidden)
            if hasattr(wrapper, "masked_frobenius_norm"):
                gv_norm = wrapper.masked_frobenius_norm(hidden_grad.detach(), mask=image_mask, eps=eps).to(device=device, dtype=dtype)
                gt_norm = wrapper.masked_frobenius_norm(hidden_grad.detach(), mask=text_mask, eps=eps).to(device=device, dtype=dtype)
            else:
                gv_norm = self._masked_norm(hidden_grad, image_mask, eps, device=device, dtype=dtype)
                gt_norm = self._masked_norm(hidden_grad, text_mask, eps, device=device, dtype=dtype)
            gf_norm = self._grad_norm_from_list(fusion_grads, eps, device=device, dtype=dtype)
            fast_s_v = float(getattr(self.args, "rho_v", 0.0)) * gv_norm
            fast_s_t = float(getattr(self.args, "rho_t", 0.0)) * gt_norm
            if hasattr(wrapper, "_aggregate_branch_sharpness"):
                branch_calibration = wrapper._aggregate_branch_sharpness(fast_s_v, fast_s_t).detach()
            else:
                branch_calibration = (0.5 * (fast_s_v + fast_s_t)).detach()
            fusion_grad_map = self._grad_map_local(fusion_named, fusion_grads)
            param_perturb = self._build_param_perturb_from_grad_map(
                fusion_named, fusion_grad_map,
                float(getattr(self.args, "rho_f", 0.0)), eps,
            )
            perturbed_outputs, _ = self._forward_training_loss(
                wrapper, batch, capture_layer0=False, hidden_perturb=None,
                param_perturb=param_perturb if param_perturb else None,
            )
            if perturbed_outputs.loss is None:
                raise RuntimeError("FAST-FCF always-on fusion probe did not return a loss.")
            loss_f_plus = perturbed_outputs.loss.float()
            s_f = (loss_f_plus - base_loss).clamp_min(0.0)
            eps_tensor = torch.tensor(float(eps), device=device, dtype=dtype)
            fcf = (s_f + eps_tensor) / (branch_calibration + eps_tensor)
            rfcf = torch.log(fcf.clamp_min(float(eps)))
            raw_gate = bounded_positive_fcf_gate(rfcf)
            robust_loss = (1.0 - raw_gate) * base_loss + raw_gate * loss_f_plus
            self._branch_correction_map = {}
            self._fusion_correction_map = {}
            self._alpha_star = 0.0
            self._projection_active = False
            self._j_bf = self._float(torch.relu(rfcf))
            geom = {
                "s_v": fast_s_v, "s_t": fast_s_t, "s_f": s_f,
                "s_branch": branch_calibration, "fcf": fcf, "rfcf": rfcf,
                "rlin_fcf": rfcf, "gv_norm": gv_norm,
                "gt_norm": gt_norm, "gf_norm": gf_norm,
            }
            logs = _vlm_geometry_logs(
                self, geom, base_loss=base_loss, perturbed_loss=loss_f_plus,
                robust_loss=robust_loss, total_loss=loss_f_plus,
                raw_gate=raw_gate, effective_gate=torch.ones_like(raw_gate),
                mode=selected_mode, uses_fast=True,
            )
            logs["cofla_f_no_second_order"] = 1.0
            logs["cofla_f_one_probe_fusion_exact"] = 1.0
            logs["cofla_f_branch_calibrated"] = 1.0
            self._last_logs = dict(logs)
            return StepOutput(loss=loss_f_plus, logs=logs)

    if mode == "base":
        # Same matched-LoRA scope as COFLA, but clean task loss only.
        vlm_train.METHOD_MAP["cofla"] = VanillaLoRAMethod
    elif geometry == "fast_fcf":
        vlm_train.METHOD_MAP["cofla"] = VLMFastCOFLAAblation
    else:
        vlm_train.METHOD_MAP["cofla"] = VLMExactCOFLAAblation


def _normalize_vlm_argv(argv: Sequence[str], *, default_optimizer: str) -> Tuple[List[str], str, str, bool]:
    args = list(argv)
    args, mode = _cli_pop_option(args, "--cofla_ablation_mode", "full")
    args, geometry = _cli_pop_option(args, "--cofla_train_geometry", "exact_fcf")
    args, nesterov_text = _cli_pop_option(args, "--sgd_nesterov", "false")
    args, sgd_momentum = _cli_pop_option(args, "--sgd_momentum", None)

    # Common aliases from the controlled scripts to train.py's VLM CLI.
    args = _cli_replace_option(args, "--learning_rate", "--lr")
    args = _cli_replace_option(args, "--hm_root", "--data_root")
    args = _cli_replace_option(args, "--max_text_length", "--max_seq_length")
    if sgd_momentum is not None and not _cli_has_option(args, "--momentum"):
        args.extend(["--momentum", str(sgd_momentum)])
    if not _cli_has_option(args, "--optimizer"):
        args.extend(["--optimizer", default_optimizer])
    if not _cli_has_option(args, "--method"):
        args.extend(["--method", "cofla"])

    # The dispatcher currently installs its ablation implementation under the
    # public 'cofla' method key so that wrappers select the matched-LoRA scope.
    normalized: List[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--method" and i + 1 < len(args):
            normalized.extend(["--method", "cofla"])
            i += 2
            continue
        if args[i].startswith("--method="):
            normalized.append("--method=cofla")
            i += 1
            continue
        normalized.append(args[i])
        i += 1

    nesterov = str(nesterov_text).strip().lower() in {"1", "true", "yes", "y", "on"}
    return normalized, str(mode), str(geometry), nesterov


def _run_vlm_ablation(argv: Sequence[str], *, default_optimizer: str) -> None:
    import sys
    import train as vlm_train

    normalized, mode, geometry, use_nesterov = _normalize_vlm_argv(
        argv, default_optimizer=default_optimizer,
    )
    _install_vlm_ablation_methods(vlm_train, mode, geometry)

    original_parse_args = vlm_train.parse_args

    def parse_args_with_ablation_metadata():
        parsed = original_parse_args()
        parsed.cofla_ablation_mode = mode
        parsed.cofla_train_geometry = geometry
        parsed.vlm_ablation_entrypoint = os.path.basename(__file__)
        return parsed

    vlm_train.parse_args = parse_args_with_ablation_metadata

    if use_nesterov:
        original_build_optimizer = vlm_train.build_optimizer

        def build_optimizer_with_nesterov(parsed, model):
            if str(getattr(parsed, "optimizer", "")).lower() != "sgd":
                return original_build_optimizer(parsed, model)
            params = [p for p in model.parameters() if p.requires_grad]
            return torch.optim.SGD(
                params,
                lr=float(parsed.lr),
                momentum=float(getattr(parsed, "momentum", 0.9)),
                weight_decay=float(parsed.weight_decay),
                nesterov=True,
            )

        vlm_train.build_optimizer = build_optimizer_with_nesterov

    sys.argv = [sys.argv[0]] + normalized
    print(
        "[VLM-ABLATION] entrypoint={} mode={} geometry={} optimizer_default={}".format(
            os.path.basename(__file__), mode, geometry, default_optimizer
        ),
        flush=True,
    )
    vlm_train.main()


if __name__ == "__main__":
    if _looks_like_vlm_cli(sys.argv[1:]):
        _run_vlm_ablation(sys.argv[1:], default_optimizer="sgd")
    else:
        main()
