from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
    HatefulMemesCollator,
    HatefulMemesDataset,
    MMIMDbCollator,
    MMIMDbDataset,
    NLVR2Collator,
    NLVR2Dataset,
    ScienceQACollator,
    ScienceQADataset,
)
from methods import (
    COFLAMethod,
    DGLLoRAMethod,
    ESAMLoRAMethod,
    FastCOFLAMethod,
    MASAMLoRAMethod,
    MSAMLoRAMethod,
    SAMLoRAMethod,
    VanillaFTMethod,
    VanillaLoRAMethod,
)
from models import build_model_wrapper
from utils.adaptive_calibration import calibrate_branch_matched_rho_f
from utils.io_utils import append_jsonl, ensure_dir, experiment_dirs, save_json, save_text, str2bool
from utils.logging_utils import setup_logger
from utils.metrics import compute_metrics_by_dataset
from utils.optim import build_optimizer, build_scheduler, compute_grad_norm
from utils.seed import build_dataloader_generator, seed_worker, set_seed
from utils.time_utils import build_elapsed_record


METHOD_MAP = {
    "vanilla_ft": VanillaFTMethod,
    "vanilla_lora": VanillaLoRAMethod,
    "sam_lora": SAMLoRAMethod,
    "esam_lora": ESAMLoRAMethod,
    "msam_lora": MSAMLoRAMethod,
    "masam_lora": MASAMLoRAMethod,
    "dgl_lora": DGLLoRAMethod,
    "cofla": COFLAMethod,
    "fast_cofla": FastCOFLAMethod,
}


def resolve_mmimdb_runtime_args(args) -> None:
    if args.dataset != "mmimdb":
        return
    if str(args.task_format).lower() != "task_head_cls":
        raise ValueError("MMIMDb must use --task_format task_head_cls because it is a multi-label task.")

    label_names = MMIMDbDataset.parse_label_names(getattr(args, "mmimdb_label_names", ""))
    if not label_names:
        label_names = MMIMDbDataset.infer_label_names(args.data_root, split_file=args.mmimdb_split_file)
        args.mmimdb_label_names = ",".join(label_names)
    if not label_names:
        raise ValueError(f"Could not infer MMIMDb label names from {args.data_root}")
    args.num_labels = len(label_names)


def select_primary_metric(dataset: str, metrics: Dict[str, float]) -> float:
    dataset = str(dataset).lower()
    if dataset == "mmimdb":
        return float(metrics.get("macro_f1", metrics.get("micro_f1", 0.0)))
    return float(metrics.get("auroc", metrics.get("accuracy", 0.0)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", type=str, default="cofla", choices=list(METHOD_MAP.keys()))
    p.add_argument("--dataset", type=str, default="hateful_memes", choices=["hateful_memes", "nlvr2", "scienceqa", "mmimdb"])
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--model_family", type=str, default=None)
    p.add_argument("--model_path", type=str, default="./external_models/qwen2_5_vl_3b_instruct")
    p.add_argument("--data_root", type=str, default="./data/hateful_memes")
    p.add_argument("--result_root", type=str, default="./outputs/full_vlm")
    p.add_argument("--exp_name", type=str, required=True)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", type=str2bool, default=True)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adamw8bit", "sgd"])
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "linear", "none"])
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--quantization", type=str, default="none", choices=["none", "8bit", "4bit"])
    p.add_argument("--local_files_only", type=str2bool, default=True)
    p.add_argument("--trust_remote_code", type=str2bool, default=True)
    p.add_argument("--use_flash_attn", type=str2bool, default=True)
    p.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    p.add_argument("--bnb_4bit_use_double_quant", type=str2bool, default=True)
    p.add_argument("--llm_int8_threshold", type=float, default=6.0)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--gradient_checkpointing", type=str2bool, default=True)
    p.add_argument("--gradient_checkpointing_use_reentrant", type=str, default="auto", choices=["auto", "true", "false"])
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--min_pixels", type=int, default=200704)
    p.add_argument("--max_pixels", type=int, default=802816)
    p.add_argument("--max_seq_length", type=int, default=512)

    p.add_argument("--log_every_n_steps", type=int, default=10)
    p.add_argument("--eval_every_n_steps", type=int, default=200)
    p.add_argument("--save_every_n_steps", type=int, default=200)
    p.add_argument("--checkpoint_mode", type=str, default="both", choices=["none", "best", "latest", "both"])
    p.add_argument("--dry_run", type=str2bool, default=False)
    p.add_argument("--preview_limit", type=int, default=80)

    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_val_samples", type=int, default=None)
    p.add_argument("--max_test_samples", type=int, default=None)

    p.add_argument("--task_format", type=str, default="verbalizer_cls", choices=["verbalizer_cls", "task_head_cls"])
    p.add_argument("--num_labels", type=int, default=2)
    p.add_argument("--label_words", type=str, default="Yes,No")
    p.add_argument("--positive_label_word", type=str, default="Yes")
    p.add_argument("--negative_label_word", type=str, default="No")
    p.add_argument("--prompt_template", type=str, default="hateful_memes_yes_no")

    p.add_argument("--hm_train_split", type=str, default="train")
    p.add_argument("--hm_val_split", type=str, default="dev_seen")
    p.add_argument("--hm_test_split", type=str, default="dev_unseen")

    p.add_argument("--nlvr2_variant", type=str, default="balanced", choices=["balanced", "unbalanced"])
    p.add_argument("--nlvr2_train_split", type=str, default="train")
    p.add_argument("--nlvr2_val_split", type=str, default="dev")
    p.add_argument("--nlvr2_test_split", type=str, default="test_public")

    p.add_argument("--scienceqa_train_split", type=str, default="train")
    p.add_argument("--scienceqa_val_split", type=str, default="validation")
    p.add_argument("--scienceqa_test_split", type=str, default="test")
    p.add_argument("--scienceqa_include_hint", type=str2bool, default=True)
    p.add_argument("--scienceqa_image_only", type=str2bool, default=False)

    p.add_argument("--mmimdb_train_split", type=str, default="train")
    p.add_argument("--mmimdb_val_split", type=str, default="dev")
    p.add_argument("--mmimdb_test_split", type=str, default="test")
    p.add_argument("--mmimdb_split_file", type=str, default="split.json")
    p.add_argument("--mmimdb_label_names", type=str, default="")
    p.add_argument("--mmimdb_threshold", type=float, default=0.5)

    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--bias", type=str, default="none")
    p.add_argument("--num_fusion_layers", type=int, default=4)
    p.add_argument("--num_lm_fusion_layers", type=int, default=4)
    p.add_argument("--num_branch_vision_layers", type=int, default=2)
    p.add_argument("--num_branch_lm_layers", type=int, default=2)
    p.add_argument("--trainable_scope", type=str, default="auto")
    p.add_argument("--internvl_max_dynamic_tiles", type=int, default=12)
    p.add_argument("--internvl_use_thumbnail", type=str2bool, default=True)

    p.add_argument("--rho_v", type=float, default=1e-3)
    p.add_argument("--rho_t", type=float, default=1e-3)
    p.add_argument("--rho_f", type=float, default=1e-2)
    p.add_argument("--auto_rho_f", type=str2bool, default=None)
    p.add_argument("--rho_f_calib_batches", type=int, default=20)
    p.add_argument("--rho_f_calib_stat", type=str, default="median", choices=["median", "mean"])
    p.add_argument("--alpha_v", type=float, default=0.5)
    p.add_argument("--alpha_t", type=float, default=0.5)
    p.add_argument("--fcf_eps", type=float, default=1e-8)
    p.add_argument("--cofla_f_ema_mu", type=float, default=0.9, help="Deprecated compatibility flag. One-probe COFLA-F does not use EMA.")
    p.add_argument("--branch_sharpness_aggregation", type=str, default="symmetric", choices=["symmetric", "weighted"])

    p.add_argument("--fast_tau", type=float, default=0.9)
    p.add_argument("--fast_d_min", type=float, default=0.1)
    p.add_argument("--sam_rho", type=float, default=None, help="SAM/ESAM/M-SAM/MASAM branch-LoRA perturbation radius. Defaults to rho_f when omitted to preserve previous VLM runs.")
    p.add_argument("--esam_keep_ratio", type=float, default=0.5)
    p.add_argument("--esam_swp_prob", type=float, default=0.6)
    p.add_argument("--msam_shapley_eps", type=float, default=1e-8)
    p.add_argument("--masam_aps_alpha", type=float, default=0.5)
    p.add_argument("--masam_ma_beta", type=float, default=0.9)
    p.add_argument("--masam_rho_min_scale", type=float, default=0.5)
    p.add_argument("--masam_rho_max_scale", type=float, default=2.0)
    p.add_argument("--dgl_correction_strength", type=float, default=0.5)

    p.add_argument("--val_compute_geometry", type=str2bool, default=False)
    p.add_argument("--test_compute_geometry", type=str2bool, default=False)
    args = p.parse_args()
    if args.sam_rho is None:
        args.sam_rho = args.rho_f
    if args.auto_rho_f is None:
        args.auto_rho_f = args.method in {"cofla", "fast_cofla"}
    return args


def build_datasets(args, wrapper, logger):
    if args.dataset == "hateful_memes":
        train_ds = HatefulMemesDataset(args.data_root, args.hm_train_split, max_samples=args.max_train_samples, seed=args.seed)
        val_ds = HatefulMemesDataset(args.data_root, args.hm_val_split, max_samples=args.max_val_samples, seed=args.seed)
        test_ds = HatefulMemesDataset(args.data_root, args.hm_test_split, max_samples=args.max_test_samples, seed=args.seed)
        collate_train = HatefulMemesCollator(wrapper, split="train")
        collate_val = HatefulMemesCollator(wrapper, split="val")
        collate_test = HatefulMemesCollator(wrapper, split="test")
    elif args.dataset == "nlvr2":
        try:
            train_ds = NLVR2Dataset(
                args.data_root,
                args.nlvr2_train_split,
                max_samples=args.max_train_samples,
                seed=args.seed,
                variant=args.nlvr2_variant,
            )
        except FileNotFoundError as exc:
            if args.max_train_samples is None:
                raise

            fallback_splits = []
            for candidate in [args.nlvr2_val_split, "dev", "validation", args.nlvr2_test_split, "test_public"]:
                candidate = str(candidate).strip()
                if not candidate or candidate == args.nlvr2_train_split or candidate in fallback_splits:
                    continue
                fallback_splits.append(candidate)

            train_ds = None
            chosen_split = None
            for fallback_split in fallback_splits:
                try:
                    train_ds = NLVR2Dataset(
                        args.data_root,
                        fallback_split,
                        max_samples=args.max_train_samples,
                        seed=args.seed,
                        variant=args.nlvr2_variant,
                    )
                    chosen_split = fallback_split
                    break
                except FileNotFoundError:
                    continue

            if train_ds is None:
                raise exc

            logger.warning(
                "NLVR2 train split '%s' is unavailable under %s. "
                "Falling back to split '%s' for this sampled debug run because --max_train_samples=%s. "
                "Train/val overlap may occur.",
                args.nlvr2_train_split,
                args.data_root,
                chosen_split,
                args.max_train_samples,
            )
        val_ds = NLVR2Dataset(
            args.data_root,
            args.nlvr2_val_split,
            max_samples=args.max_val_samples,
            seed=args.seed,
            variant=args.nlvr2_variant,
        )
        test_ds = NLVR2Dataset(
            args.data_root,
            args.nlvr2_test_split,
            max_samples=args.max_test_samples,
            seed=args.seed,
            variant=args.nlvr2_variant,
        )
        collate_train = NLVR2Collator(wrapper, split="train", data_root=args.data_root)
        collate_val = NLVR2Collator(wrapper, split="val", data_root=args.data_root)
        collate_test = NLVR2Collator(wrapper, split="test", data_root=args.data_root)
    elif args.dataset == "scienceqa":
        train_ds = ScienceQADataset(
            args.data_root,
            args.scienceqa_train_split,
            max_samples=args.max_train_samples,
            seed=args.seed,
            image_only=args.scienceqa_image_only,
        )
        val_ds = ScienceQADataset(
            args.data_root,
            args.scienceqa_val_split,
            max_samples=args.max_val_samples,
            seed=args.seed,
            image_only=args.scienceqa_image_only,
        )
        test_ds = ScienceQADataset(
            args.data_root,
            args.scienceqa_test_split,
            max_samples=args.max_test_samples,
            seed=args.seed,
            image_only=args.scienceqa_image_only,
        )
        collate_train = ScienceQACollator(wrapper, split="train", data_root=args.data_root, include_hint=args.scienceqa_include_hint)
        collate_val = ScienceQACollator(wrapper, split="val", data_root=args.data_root, include_hint=args.scienceqa_include_hint)
        collate_test = ScienceQACollator(wrapper, split="test", data_root=args.data_root, include_hint=args.scienceqa_include_hint)
    elif args.dataset == "mmimdb":
        label_names = MMIMDbDataset.parse_label_names(args.mmimdb_label_names)
        train_ds = MMIMDbDataset(
            args.data_root,
            args.mmimdb_train_split,
            max_samples=args.max_train_samples,
            seed=args.seed,
            label_names=label_names,
            split_file=args.mmimdb_split_file,
        )
        val_ds = MMIMDbDataset(
            args.data_root,
            args.mmimdb_val_split,
            max_samples=args.max_val_samples,
            seed=args.seed,
            label_names=label_names,
            split_file=args.mmimdb_split_file,
        )
        test_ds = MMIMDbDataset(
            args.data_root,
            args.mmimdb_test_split,
            max_samples=args.max_test_samples,
            seed=args.seed,
            label_names=label_names,
            split_file=args.mmimdb_split_file,
        )
        collate_train = MMIMDbCollator(wrapper, split="train", data_root=args.data_root)
        collate_val = MMIMDbCollator(wrapper, split="val", data_root=args.data_root)
        collate_test = MMIMDbCollator(wrapper, split="test", data_root=args.data_root)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    logger.info(f"Train dataset summary: {train_ds.summary()}")
    logger.info(f"Val dataset summary: {val_ds.summary()}")
    logger.info(f"Test dataset summary: {test_ds.summary()}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_train,
        worker_init_fn=seed_worker,
        generator=build_dataloader_generator(args.seed),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_val,
        worker_init_fn=seed_worker,
        generator=build_dataloader_generator(args.seed + 1),
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_test,
        worker_init_fn=seed_worker,
        generator=build_dataloader_generator(args.seed + 2),
        persistent_workers=args.num_workers > 0,
    )
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def build_calibration_loader(train_loader, args):
    return DataLoader(
        train_loader.dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=train_loader.collate_fn,
        worker_init_fn=seed_worker,
        generator=build_dataloader_generator(args.seed + 97),
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


def maybe_run_rho_f_calibration(args, wrapper, train_loader, dirs, logger) -> Dict[str, Any]:
    calibration_info: Dict[str, Any]
    if bool(getattr(args, "auto_rho_f", False)):
        wrapper.train()
        calibration_loader = build_calibration_loader(train_loader, args)

        def _prepare_batch(raw_batch):
            return wrapper.move_batch_to_device(raw_batch)

        def _compute_stats(batch):
            with torch.enable_grad():
                return wrapper.compute_linear_geometry_stats(batch, retain_graph=False)

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


def evaluate(wrapper, method, dataloader, args, split: str, logger=None):
    wrapper.eval()
    labels: List[Any] = []
    preds: List[Any] = []
    metric_scores: List[Any] = []
    preds_out: List[Dict[str, Any]] = []
    geo_sums = defaultdict(float)
    geo_count = 0
    geometry_disabled = False

    for raw_batch in tqdm(dataloader, desc=f"eval:{split}"):
        batch = wrapper.move_batch_to_device(raw_batch)
        with torch.no_grad():
            prompt_scores = wrapper.compute_prompt_scores(batch)
        batch_probs = prompt_scores["probs"].detach().cpu().tolist()
        batch_preds = prompt_scores["preds"].detach().cpu().tolist()
        batch_labels = batch["labels"].detach().cpu().tolist()
        sample_ids = batch["sample_ids"]

        labels.extend(batch_labels)
        preds.extend(batch_preds)
        metric_scores.extend(batch_probs)
        for sid, lab, pred, prob in zip(sample_ids, batch_labels, batch_preds, batch_probs):
            record = {"id": sid}
            if isinstance(lab, list):
                record["labels"] = [int(x) for x in lab]
            else:
                record["label"] = int(lab)
            if isinstance(pred, list):
                record["preds"] = [int(x) for x in pred]
            else:
                record["pred"] = int(pred)
            if isinstance(prob, list):
                record["scores"] = [float(x) for x in prob]
            else:
                record["prob"] = float(prob)
            preds_out.append(record)
        del prompt_scores

        need_geo = ((split == "val" and args.val_compute_geometry) or (split == "test" and args.test_compute_geometry)) and not geometry_disabled
        if need_geo:
            try:
                geom = method.validation_geometry(wrapper, batch)
            except torch.cuda.OutOfMemoryError as exc:
                geometry_disabled = True
                if logger is not None:
                    logger.warning(
                        "Skipping geometry metrics for split=%s after CUDA OOM: %s. "
                        "Use --val_compute_geometry false or reduce --max_pixels for multi-image evaluation.",
                        split,
                        exc,
                    )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                for k, v in geom.items():
                    geo_sums[k] += float(v)
                geo_count += 1

    task_metrics = compute_metrics_by_dataset(
        args.dataset,
        labels,
        preds,
        metric_scores,
        threshold=float(getattr(args, "mmimdb_threshold", 0.5)),
    )
    result = dict(task_metrics)
    if geo_count > 0:
        for k, v in geo_sums.items():
            result[k] = v / geo_count
    return result, preds_out


def checkpoint_mode_allows(mode: str, checkpoint_type: str) -> bool:
    mode = str(mode).lower()
    checkpoint_type = str(checkpoint_type).lower()
    if mode == "both":
        return True
    if mode == "none":
        return False
    return mode == checkpoint_type


def build_checkpoint_metadata(step: int, epoch: int, best_metric: float, args) -> Dict[str, Any]:
    return {
        "global_step": step,
        "epoch": epoch,
        "best_metric": best_metric,
        "method": args.method,
        "dataset": args.dataset,
        "model_name": getattr(args, "model_name", None),
        "model_family": getattr(args, "model_family", None),
        "trainable_scope": getattr(args, "trainable_scope", None),
    }


def save_checkpoint(wrapper, ckpt_dir: str, step: int, epoch: int, best_metric: float, args):
    wrapper.save_trainable_checkpoint(ckpt_dir, build_checkpoint_metadata(step, epoch, best_metric, args))


def log_trainable_report(wrapper, logger, args) -> Dict[str, Any]:
    if hasattr(wrapper, "describe_trainable_state"):
        report = wrapper.describe_trainable_state(preview_limit=args.preview_limit)
    else:
        report = {}
    if report:
        logger.info(f"Trainable report: {json.dumps(report, ensure_ascii=False)}")
    return report


def run_dry_run(wrapper, method, train_loader, args, dirs, logger) -> Dict[str, Any]:
    iterator = iter(train_loader)
    try:
        raw_batch = next(iterator)
    except StopIteration as exc:
        raise RuntimeError("Dry-run failed because the training dataloader is empty.") from exc

    batch = wrapper.move_batch_to_device(raw_batch)
    dry_run_report: Dict[str, Any] = {
        "model_name": getattr(args, "model_name", None),
        "model_family": getattr(args, "model_family", None),
        "trainable_scope": getattr(args, "trainable_scope", None),
        "runtime": wrapper.describe_runtime(),
        "trainable_report": log_trainable_report(wrapper, logger, args),
    }

    with torch.no_grad():
        prompt_scores = wrapper.compute_prompt_scores(batch)
        dry_run_report["prompt_score_shapes"] = {
            key: list(value.shape) if torch.is_tensor(value) else str(type(value))
            for key, value in prompt_scores.items()
        }

    base_outputs, _ = wrapper.compute_base_loss(batch, capture_layer0=False)
    if base_outputs.loss is None:
        raise RuntimeError("Dry-run forward did not produce a loss tensor.")
    dry_run_report["base_loss"] = float(base_outputs.loss.detach().cpu().item())

    with torch.enable_grad():
        geometry_probe = wrapper.compute_geometry_probe(batch)
        geometry_metrics = wrapper.compute_geometry_metrics_from_probe(batch, geometry_probe)
    dry_run_report["geometry_metrics"] = {
        key: float(value.detach().cpu().item()) if torch.is_tensor(value) and value.dim() == 0 else str(type(value))
        for key, value in geometry_metrics.items()
    }

    wrapper.zero_grad(set_to_none=True)
    step_output = method.training_step(wrapper, batch)
    step_output.loss.backward()
    grad_names = []
    for name, param in wrapper.get_fusion_trainable_named_parameters():
        if param.grad is None:
            continue
        if float(param.grad.detach().float().abs().sum().cpu().item()) > 0.0:
            grad_names.append(name)
    post_backward_logs = method.after_backward(wrapper)
    wrapper.zero_grad(set_to_none=True)

    dry_run_report["training_step_loss"] = float(step_output.loss.detach().cpu().item())
    dry_run_report["training_step_logs"] = dict(step_output.logs)
    dry_run_report["post_backward_logs"] = dict(post_backward_logs)
    dry_run_report["nonzero_grad_parameter_preview"] = grad_names[: args.preview_limit]
    dry_run_report["num_nonzero_grad_parameters"] = len(grad_names)

    save_json(dry_run_report, os.path.join(dirs["exp_dir"], "dry_run_report.json"))
    logger.info(f"Dry-run summary: {json.dumps(dry_run_report, ensure_ascii=False)}")
    return dry_run_report


def summary_text(
    best_val: Dict[str, float],
    test_metrics: Dict[str, float],
    args,
    runtime: Dict[str, Any],
    elapsed: Dict[str, Any],
) -> str:
    lines = []
    lines.append("Experiment summary")
    lines.append(f"method: {args.method}")
    lines.append(f"dataset: {args.dataset}")
    lines.append(f"exp_name: {args.exp_name}")
    lines.append(f"checkpoint_mode: {args.checkpoint_mode}")
    lines.append("")
    lines.append("Best validation metrics")
    for k, v in sorted(best_val.items()):
        lines.append(f"- {k}: {v:.6f}")
    lines.append("")
    lines.append("Final test metrics")
    for k, v in sorted(test_metrics.items()):
        lines.append(f"- {k}: {v:.6f}")
    lines.append("")
    lines.append("Elapsed time")
    lines.append(f"- elapsed_seconds: {elapsed['elapsed_seconds']}")
    lines.append(f"- elapsed_hms: {elapsed['elapsed_hms']}")
    lines.append("")
    lines.append("Runtime label words")
    li = runtime["label_info"]
    if li.get("mode") == "binary":
        lines.append(f"- positive_text: {li['positive_text']} | token_id: {li['positive_token_id']}")
        lines.append(f"- negative_text: {li['negative_text']} | token_id: {li['negative_token_id']}")
    else:
        lines.append(f"- texts: {li['texts']}")
        lines.append(f"- token_ids: {li['token_ids']}")
    lines.append(f"- used_fallback: {li['used_fallback']}")
    lines.append("")
    lines.append("Metric logging protocol")
    lines.append(
        "The evaluation protocol uses two groups of metrics. Task metrics are "
        "Accuracy for NLVR2 and ScienceQA, AUROC and Accuracy for Hateful Memes, "
        "and Macro-F1 and Micro-F1 for MM-IMDb. Geometry and mechanism metrics "
        "include Fusion Consistency Flatness (FCF), the fusion-side sharpness proxy "
        "(S_f^b), branch-side sharpness proxies (S_v^b), (S_t^b), and "
        "(S_branch^b), RFCF, Rlin_FCF, and their associated statistics. During "
        "training, the logs track loss, learning rate, gradient norm, (S_f^b), "
        "branch proxies, and FCF for diagnostics. During validation, task metrics "
        "and averaged geometry metrics are used for model selection. During final "
        "testing, the best checkpoint is evaluated on task metrics, with optional "
        "geometry metrics for analysis; test metrics are not used for tuning."
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    experiment_start_perf = time.perf_counter()
    set_seed(args.seed, deterministic=args.deterministic)
    resolve_mmimdb_runtime_args(args)
    dirs = experiment_dirs(args.result_root, args.exp_name)
    logger = setup_logger(os.path.join(dirs["logs"], "train.log"))

    wrapper = build_model_wrapper(args)
    wrapper.apply_method_setup(args.method)
    runtime = wrapper.describe_runtime()
    logger.info(json.dumps(runtime["label_info"], ensure_ascii=False))
    logger.info(f"Discovered fusion modules: {len(runtime['fusion_linear_module_names'])}")
    logger.info(f"Parameter counts: {runtime['parameter_counts']}")
    logger.info(f"Checkpoint mode: {args.checkpoint_mode}")
    trainable_report = log_trainable_report(wrapper, logger, args)

    _, _, _, train_loader, val_loader, test_loader = build_datasets(args, wrapper, logger)
    maybe_run_rho_f_calibration(args, wrapper, train_loader, dirs, logger)
    method = METHOD_MAP[args.method](args)

    config_blob = vars(args).copy()
    config_blob["runtime"] = runtime
    config_blob["trainable_report"] = trainable_report
    config_blob["adaptive_calibration"] = rho_f_calibration_payload(args)
    save_json(config_blob, os.path.join(dirs["exp_dir"], "config.json"))

    if args.dry_run:
        run_dry_run(wrapper, method, train_loader, args, dirs, logger)
        elapsed = build_elapsed_record(time.perf_counter() - experiment_start_perf)
        logger.info(f"Dry-run elapsed time: {elapsed['elapsed_seconds']}s ({elapsed['elapsed_hms']})")
        return

    updates_per_epoch = math.ceil(len(train_loader) / max(1, args.gradient_accumulation_steps))
    total_steps = max(1, updates_per_epoch * args.num_train_epochs)
    optimizer = build_optimizer(args, wrapper)
    scheduler = build_scheduler(args, optimizer, total_steps)

    global_step = 0
    best_val_metric = -1e18
    best_val_metrics = {}
    best_state = None
    last_eval_metrics = {}
    latest_ckpt = os.path.join(dirs["checkpoints"], "latest")
    best_ckpt = os.path.join(dirs["checkpoints"], "best")

    wrapper.train()
    optimizer.zero_grad(set_to_none=True)

    def run_optimizer_update(logs: Dict[str, Any], epoch_idx: int, accum_steps_in_update: int) -> None:
        nonlocal global_step, best_val_metric, best_val_metrics, best_state, last_eval_metrics

        trainable_params = wrapper.get_trainable_parameters()
        if accum_steps_in_update <= 0:
            return
        if accum_steps_in_update < args.gradient_accumulation_steps:
            scale = args.gradient_accumulation_steps / float(accum_steps_in_update)
            for param in trainable_params:
                if param.grad is not None:
                    param.grad.mul_(scale)

        grad_norm = compute_grad_norm(trainable_params)
        if args.max_grad_norm and args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        logs["lr"] = float(optimizer.param_groups[0]["lr"])
        logs["grad_norm"] = float(grad_norm)
        logs["global_step"] = int(global_step)
        logs["epoch"] = int(epoch_idx)

        if global_step % args.log_every_n_steps == 0:
            logger.info(" | ".join([f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in logs.items()]))
            append_jsonl({"type": "train", **logs}, os.path.join(dirs["metrics"], "metrics.jsonl"))

        if global_step % args.eval_every_n_steps == 0:
            val_metrics, val_preds = evaluate(wrapper, method, val_loader, args, split="val", logger=logger)
            last_eval_metrics = val_metrics
            logger.info(f"Validation @ step {global_step}: {val_metrics}")
            append_jsonl({"type": "eval", "split": "val", "global_step": global_step, **val_metrics}, os.path.join(dirs["metrics"], "metrics.jsonl"))
            save_json({"global_step": global_step, "metrics": val_metrics, "predictions": val_preds}, os.path.join(dirs["predictions"], "val_predictions.json"))
            current_metric = select_primary_metric(args.dataset, val_metrics)
            if current_metric > best_val_metric:
                best_val_metric = current_metric
                best_val_metrics = deepcopy(val_metrics)
                best_state = wrapper.export_trainable_state(
                    build_checkpoint_metadata(global_step, epoch_idx, best_val_metric, args)
                )
                if checkpoint_mode_allows(args.checkpoint_mode, "best"):
                    save_checkpoint(wrapper, best_ckpt, global_step, epoch_idx, best_val_metric, args)
                    logger.info(f"Updated best checkpoint at step {global_step} with score {best_val_metric:.6f}")
                else:
                    logger.info(f"Updated in-memory best state at step {global_step} with score {best_val_metric:.6f}")
            wrapper.train()

        if global_step % args.save_every_n_steps == 0 and checkpoint_mode_allows(args.checkpoint_mode, "latest"):
            save_checkpoint(wrapper, latest_ckpt, global_step, epoch_idx, best_val_metric, args)
            logger.info(f"Saved latest checkpoint at step {global_step}")

    for epoch in range(args.num_train_epochs):
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}")
        pending_micro_steps = 0
        pending_logs: Dict[str, Any] | None = None
        for step_in_epoch, raw_batch in enumerate(pbar):
            batch = wrapper.move_batch_to_device(raw_batch)
            step_output = method.training_step(wrapper, batch)
            loss = step_output.loss / args.gradient_accumulation_steps
            loss.backward()
            logs = dict(step_output.logs)
            post_logs = method.after_backward(wrapper)
            logs.update(post_logs)
            pending_micro_steps += 1
            pending_logs = logs

            if (step_in_epoch + 1) % args.gradient_accumulation_steps == 0:
                run_optimizer_update(logs, epoch, pending_micro_steps)
                pending_micro_steps = 0
                pending_logs = None

        if pending_micro_steps > 0 and pending_logs is not None:
            run_optimizer_update(pending_logs, epoch, pending_micro_steps)

    if checkpoint_mode_allows(args.checkpoint_mode, "latest"):
        save_checkpoint(wrapper, latest_ckpt, global_step, args.num_train_epochs - 1, best_val_metric, args)

    if not best_val_metrics:
        best_val_metrics, val_preds = evaluate(wrapper, method, val_loader, args, split="val", logger=logger)
        save_json({"global_step": global_step, "metrics": best_val_metrics, "predictions": val_preds}, os.path.join(dirs["predictions"], "val_predictions.json"))
        best_val_metric = select_primary_metric(args.dataset, best_val_metrics)
        best_state = wrapper.export_trainable_state(
            build_checkpoint_metadata(global_step, args.num_train_epochs - 1, best_val_metric, args)
        )
        if checkpoint_mode_allows(args.checkpoint_mode, "best"):
            save_checkpoint(wrapper, best_ckpt, global_step, args.num_train_epochs - 1, best_val_metric, args)

    load_info = wrapper.load_trainable_state(best_state, strict=False)
    logger.info(f"Loaded best state for test: {load_info}")
    test_metrics, test_preds = evaluate(wrapper, method, test_loader, args, split="test", logger=logger)
    logger.info(f"Test metrics: {test_metrics}")
    append_jsonl({"type": "eval", "split": "test", "global_step": global_step, **test_metrics}, os.path.join(dirs["metrics"], "metrics.jsonl"))
    save_json({"global_step": global_step, "metrics": test_metrics, "predictions": test_preds}, os.path.join(dirs["predictions"], "test_predictions.json"))
    elapsed = build_elapsed_record(time.perf_counter() - experiment_start_perf)
    logger.info(f"Elapsed time: {elapsed['elapsed_seconds']}s ({elapsed['elapsed_hms']})")
    final_metrics = {
        "dataset": args.dataset,
        "method": args.method,
        "seed": args.seed,
        "best_val_metric": best_val_metric,
        "best_val_metrics": best_val_metrics,
        "last_eval_metrics": last_eval_metrics,
        "test_metrics": test_metrics,
        "adaptive_calibration": rho_f_calibration_payload(args),
        "trainable_report": trainable_report,
        "elapsed": elapsed,
    }
    save_json(final_metrics, os.path.join(dirs["metrics"], "final_metrics.json"))
    save_text(summary_text(best_val_metrics, test_metrics, args, runtime, elapsed), os.path.join(dirs["exp_dir"], "summary.txt"))


if __name__ == "__main__":
    main()
