from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, CLIPImageProcessor

import train_controlled_latefusion as base
# COFLA-F implementation is inherited from train_controlled_latefusion.py.
from datasets.base import coerce_image_to_pil
from datasets.nlvr2 import NLVR2Dataset
from datasets.scienceqa import ScienceQADataset
from models.controlled_late_fusion import ControlledLateFusionModel
from utils.controlled_metrics import GEOMETRY_KEYS, mean_dict
from utils.io_utils import str2bool
from utils.seed import build_dataloader_generator, seed_worker

_ORIG_BUILD_ARG_PARSER = base.build_arg_parser
_ORIG_SAVE_PREDICTIONS_CSV = base.save_predictions_csv


DEFAULT_NLVR2_NUM_LABELS = 1
DEFAULT_SCIENCEQA_MAX_CHOICES = 5
BLANK_IMAGE_SIZE = (224, 224)


def _blank_image(size: Tuple[int, int] = BLANK_IMAGE_SIZE, color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    return Image.new("RGB", size, color)


def _concat_pair_horizontally(left: Image.Image, right: Image.Image, pad_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    left = left.convert("RGB")
    right = right.convert("RGB")
    width = left.width + right.width
    height = max(left.height, right.height)
    canvas = Image.new("RGB", (width, height), pad_color)
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


def _scienceqa_prompt(question: str, choices: Sequence[str], hint: str = "", include_hint: bool = True) -> str:
    option_labels = [chr(ord("A") + i) for i in range(len(choices))]
    parts: List[str] = []
    if include_hint and str(hint or "").strip():
        parts.append(f"Hint: {str(hint).strip()}")
    parts.append(f"Question: {str(question).strip()}")
    parts.append("Options:")
    for label, choice in zip(option_labels, choices):
        parts.append(f"{label}. {str(choice).strip()}")
    return "\n".join(parts)


class ControlledNLVR2Collator:
    def __init__(self, *, image_processor, tokenizer, max_text_length: int, data_root: Optional[str] = None) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_text_length = int(max_text_length)
        self.data_root = data_root

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        sample_ids: List[str] = []
        statements: List[str] = []
        images: List[Image.Image] = []
        labels: List[float] = []

        for sample in batch:
            left_image = coerce_image_to_pil(sample["left_image"], root=self.data_root)
            right_image = coerce_image_to_pil(sample["right_image"], root=self.data_root)
            merged = _concat_pair_horizontally(left_image, right_image)
            sample_ids.append(str(sample["id"]))
            statements.append(str(sample["statement"]))
            images.append(merged)
            labels.append(float(sample["label"]))

        vision_inputs = self.image_processor(images=images, return_tensors="pt")
        model_max_length = int(getattr(self.tokenizer, "model_max_length", self.max_text_length) or self.max_text_length)
        if model_max_length <= 0 or model_max_length > 100000:
            model_max_length = self.max_text_length
        text_inputs = self.tokenizer(
            statements,
            padding=True,
            truncation=True,
            max_length=min(self.max_text_length, model_max_length),
            return_tensors="pt",
        )
        return {
            "dataset_name": "nlvr2",
            "sample_ids": sample_ids,
            "texts": statements,
            "pixel_values": vision_inputs["pixel_values"],
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs.get("attention_mask"),
            "labels": torch.tensor(labels, dtype=torch.float32),
        }


class ControlledScienceQACollator:
    def __init__(
        self,
        *,
        image_processor,
        tokenizer,
        max_text_length: int,
        data_root: Optional[str] = None,
        include_hint: bool = True,
        max_choices: int = DEFAULT_SCIENCEQA_MAX_CHOICES,
    ) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_text_length = int(max_text_length)
        self.data_root = data_root
        self.include_hint = bool(include_hint)
        self.max_choices = int(max_choices)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        sample_ids: List[str] = []
        texts: List[str] = []
        images: List[Image.Image] = []
        labels: List[int] = []
        target_indices: List[int] = []
        choice_masks: List[List[float]] = []
        choice_counts: List[int] = []

        for sample in batch:
            choices = list(sample["choices"])
            num_choices = len(choices)
            if num_choices > self.max_choices:
                raise ValueError(
                    f"ScienceQA sample {sample['id']} has {num_choices} options, which exceeds max_choices={self.max_choices}."
                )
            prompt = _scienceqa_prompt(
                question=str(sample["question"]),
                choices=choices,
                hint=str(sample.get("hint", "")),
                include_hint=self.include_hint,
            )
            image_value = sample.get("image")
            image = coerce_image_to_pil(image_value, root=self.data_root) if image_value is not None else _blank_image()
            label_index = int(sample["label"])
            if not 0 <= label_index < num_choices:
                raise ValueError(
                    f"ScienceQA sample {sample['id']} has invalid label={label_index} "
                    f"for num_choices={num_choices}."
                )
            mask = [1.0] * num_choices + [0.0] * (self.max_choices - num_choices)

            sample_ids.append(str(sample["id"]))
            texts.append(prompt)
            images.append(image)
            labels.append(label_index)
            target_indices.append(label_index)
            choice_masks.append(mask)
            choice_counts.append(num_choices)

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
        return {
            "dataset_name": "scienceqa",
            "sample_ids": sample_ids,
            "texts": texts,
            "pixel_values": vision_inputs["pixel_values"],
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs.get("attention_mask"),
            # Dedicated ScienceQA training target: integer answer index for
            # masked single-choice cross-entropy. target_indices is retained for
            # prediction export and backward-compatible evaluation code.
            "labels": torch.tensor(labels, dtype=torch.long),
            "target_indices": torch.tensor(target_indices, dtype=torch.long),
            "choice_mask": torch.tensor(choice_masks, dtype=torch.float32),
            "choice_count": torch.tensor(choice_counts, dtype=torch.long),
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = _ORIG_BUILD_ARG_PARSER()
    for action in parser._actions:
        if action.dest == "dataset":
            action.choices = ["nlvr2", "scienceqa"]
            action.default = "nlvr2"

    parser.add_argument("--nlvr2_root", type=str, default="./data/nlvr2")
    parser.add_argument("--scienceqa_root", type=str, default="./data/scienceqa")
    parser.add_argument("--text_encoder_type", type=str, default="roberta", choices=["roberta", "clip_text"])

    parser.add_argument("--nlvr2_variant", type=str, default="balanced", choices=["balanced", "unbalanced"])
    parser.add_argument("--nlvr2_train_split", type=str, default="train")
    parser.add_argument("--nlvr2_val_split", type=str, default="dev")
    parser.add_argument("--nlvr2_test_split", type=str, default="test_public")

    parser.add_argument("--scienceqa_train_split", type=str, default="train")
    parser.add_argument("--scienceqa_val_split", type=str, default="validation")
    parser.add_argument("--scienceqa_test_split", type=str, default="test")
    parser.add_argument("--scienceqa_include_hint", type=str2bool, default=True)
    parser.add_argument("--scienceqa_num_labels", type=int, default=0, help="0 means infer from the loaded ScienceQA splits.")
    parser.add_argument(
        "--nlvr2_scienceqa_geometry_fp32",
        type=str2bool,
        default=True,
        help=(
            "Run COFLA perturbation geometry/calibration in fp32 for NLVR2 and "
            "ScienceQA only. This flag is not used by the original Hateful "
            "Memes or MM-IMDb entrypoint."
        ),
    )
    return parser


def encoder_setup_for_dataset(dataset: str) -> Tuple[str, str]:
    text = "roberta"
    return "clip", text


def text_encoder_type_for_dataset(dataset: str) -> str:
    return "roberta"


def build_experiment_name(args) -> str:
    scope = str(getattr(args, "branch_tuning_mode", "frozen")).strip().lower()
    scope_tag = "branch_lora_fullfusion" if scope == "branch_lora_full_fusion" else "frozen_branch"
    return f"{args.dataset}_clip_roberta_{args.fusion_type}_{args.method}_{scope_tag}_seed{args.seed}"


def _infer_scienceqa_num_labels(*datasets: ScienceQADataset) -> int:
    max_choices = 0
    for ds in datasets:
        for item in getattr(ds, "records", []):
            max_choices = max(max_choices, len(item.get("choices", [])))
    return max_choices or DEFAULT_SCIENCEQA_MAX_CHOICES


def build_dataloaders(args, image_processor, tokenizer, logger):
    if args.dataset == "nlvr2":
        train_dataset = NLVR2Dataset(
            args.nlvr2_root,
            args.nlvr2_train_split,
            max_samples=args.max_train_samples,
            seed=args.seed,
            variant=args.nlvr2_variant,
        )
        val_dataset = NLVR2Dataset(
            args.nlvr2_root,
            args.nlvr2_val_split,
            max_samples=args.max_val_samples,
            seed=args.seed,
            variant=args.nlvr2_variant,
        )
        test_dataset = NLVR2Dataset(
            args.nlvr2_root,
            args.nlvr2_test_split,
            max_samples=args.max_test_samples,
            seed=args.seed,
            variant=args.nlvr2_variant,
        )
        label_names = ["true"]
        collator = ControlledNLVR2Collator(
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_text_length=args.max_text_length,
            data_root=args.nlvr2_root,
        )
    else:
        train_dataset = ScienceQADataset(
            args.scienceqa_root,
            args.scienceqa_train_split,
            max_samples=args.max_train_samples,
            seed=args.seed,
        )
        val_dataset = ScienceQADataset(
            args.scienceqa_root,
            args.scienceqa_val_split,
            max_samples=args.max_val_samples,
            seed=args.seed,
        )
        test_dataset = ScienceQADataset(
            args.scienceqa_root,
            args.scienceqa_test_split,
            max_samples=args.max_test_samples,
            seed=args.seed,
        )
        if int(getattr(args, "scienceqa_num_labels", 0) or 0) <= 0:
            args.scienceqa_num_labels = _infer_scienceqa_num_labels(train_dataset, val_dataset, test_dataset)
        label_names = [chr(ord("A") + i) for i in range(int(args.scienceqa_num_labels))]
        collator = ControlledScienceQACollator(
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_text_length=args.max_text_length,
            data_root=args.scienceqa_root,
            include_hint=bool(args.scienceqa_include_hint),
            max_choices=int(args.scienceqa_num_labels),
        )

    logger.info("Train dataset summary: %s", json.dumps(train_dataset.summary(), ensure_ascii=False))
    logger.info("Val dataset summary: %s", json.dumps(val_dataset.summary(), ensure_ascii=False))
    logger.info("Test dataset summary: %s", json.dumps(test_dataset.summary(), ensure_ascii=False))

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


def build_model_and_tokenizers(args, device: torch.device):
    if not os.path.exists(args.clip_path):
        raise FileNotFoundError(f"CLIP path does not exist: {args.clip_path}")
    if str(args.text_encoder_type).strip().lower() == "roberta" and not os.path.exists(args.roberta_path):
        raise FileNotFoundError(f"RoBERTa path does not exist: {args.roberta_path}")

    image_processor = CLIPImageProcessor.from_pretrained(
        args.clip_path,
        local_files_only=args.local_files_only,
    )
    tokenizer_path = args.roberta_path if str(args.text_encoder_type).strip().lower() == "roberta" else args.clip_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=args.local_files_only,
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    if args.dataset == "nlvr2":
        # Explicit dedicated branch; do not route through Hateful Memes.
        model_dataset_type = "nlvr2"
        num_labels = DEFAULT_NLVR2_NUM_LABELS
    else:
        # Explicit dedicated branch; do not route through MM-IMDb.
        model_dataset_type = "scienceqa"
        num_labels = int(args.scienceqa_num_labels or DEFAULT_SCIENCEQA_MAX_CHOICES)

    model = ControlledLateFusionModel(
        dataset_type=model_dataset_type,
        text_encoder_type=str(args.text_encoder_type).strip().lower(),
        fusion_type=args.fusion_type,
        clip_path=args.clip_path,
        roberta_path=args.roberta_path if str(args.text_encoder_type).strip().lower() == "roberta" else None,
        fusion_dim=args.fusion_dim,
        dropout=args.dropout,
        num_labels=num_labels,
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


def compute_controlled_metrics(dataset: str, labels, probs, threshold: float = 0.5) -> Dict[str, float]:
    dataset_key = str(dataset).strip().lower()
    if dataset_key == "nlvr2":
        labels_t = torch.as_tensor(labels, dtype=torch.long)
        preds_t = torch.as_tensor(probs, dtype=torch.float32) >= float(threshold)
        accuracy = float((preds_t.to(torch.long) == labels_t).float().mean().item()) if labels_t.numel() else float("nan")
        return {"accuracy": accuracy}
    if dataset_key == "scienceqa":
        labels_t = torch.as_tensor(labels, dtype=torch.long)
        preds_t = torch.as_tensor(probs, dtype=torch.long)
        accuracy = float((preds_t == labels_t).float().mean().item()) if labels_t.numel() else float("nan")
        return {"accuracy": accuracy}
    raise ValueError(f"Unsupported controlled dataset: {dataset}")


def select_primary_metric(dataset: str, metrics: Dict[str, float]) -> float:
    return float(metrics.get("accuracy", float("-inf")))


def primary_metric_name(dataset: str) -> str:
    return "Accuracy"


def save_predictions_csv(path: str, prediction_rows: List[Dict[str, Any]], dataset: str, label_names: Sequence[str]) -> None:
    dataset_key = str(dataset).strip().lower()
    if dataset_key == "nlvr2":
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

    if dataset_key == "scienceqa":
        fieldnames = ["id", "label_index", "pred_index", "score", "num_choices"]
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in prediction_rows:
                writer.writerow(
                    {
                        "id": row["id"],
                        "label_index": int(row["label_index"]),
                        "pred_index": int(row["pred_index"]),
                        "score": float(row["score"]),
                        "num_choices": int(row["num_choices"]),
                    }
                )
        return

    return _ORIG_SAVE_PREDICTIONS_CSV(path, prediction_rows, dataset, label_names)


def evaluate(
    model: ControlledLateFusionModel,
    dataloader: DataLoader,
    args,
    *,
    device: torch.device,
    split: str,
    compute_geometry: bool,
):
    model.eval()
    labels: List[Any] = []
    probs_or_preds: List[Any] = []
    prediction_rows: List[Dict[str, Any]] = []
    geometry_records: List[Dict[str, float]] = []

    for raw_batch in base.tqdm(dataloader, desc=f"eval:{split}"):
        batch = base.move_batch_to_device(raw_batch, device)
        with torch.no_grad():
            with base.maybe_autocast(device, args.fp16):
                outputs = model(batch)

        if args.dataset == "nlvr2":
            batch_probs = torch.sigmoid(outputs.logits).squeeze(-1)
            batch_preds = (batch_probs >= 0.5).long()
            batch_labels = batch["labels"].detach().cpu().to(torch.int64).tolist()
            probs_list = batch_probs.detach().cpu().tolist()
            preds_list = batch_preds.detach().cpu().tolist()
            labels.extend(batch_labels)
            probs_or_preds.extend(probs_list)
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
            logits = outputs.logits
            choice_mask = batch["choice_mask"].to(device=logits.device, dtype=logits.dtype)
            masked_logits = logits.masked_fill(choice_mask <= 0, torch.finfo(logits.dtype).min)
            pred_indices = masked_logits.argmax(dim=-1)
            probs = torch.softmax(masked_logits, dim=-1)
            pred_scores = probs.gather(dim=-1, index=pred_indices.unsqueeze(-1)).squeeze(-1)

            gold_indices = batch["target_indices"].detach().cpu().tolist()
            pred_indices_list = pred_indices.detach().cpu().tolist()
            pred_scores_list = pred_scores.detach().cpu().tolist()
            choice_count_list = batch["choice_count"].detach().cpu().tolist()
            labels.extend(gold_indices)
            probs_or_preds.extend(pred_indices_list)
            for sample_id, label_value, pred_value, score_value, num_choices in zip(
                batch["sample_ids"], gold_indices, pred_indices_list, pred_scores_list, choice_count_list
            ):
                prediction_rows.append(
                    {
                        "id": sample_id,
                        "label_index": int(label_value),
                        "pred_index": int(pred_value),
                        "score": float(score_value),
                        "num_choices": int(num_choices),
                    }
                )

        if compute_geometry:
            with torch.enable_grad():
                # Perturbation differences can be below fp16 resolution.  This
                # dedicated entrypoint evaluates NLVR2/ScienceQA geometry in fp32
                # without changing the original two datasets' evaluation path.
                geometry_amp = args.fp16 and not bool(args.nlvr2_scienceqa_geometry_fp32)
                with base.maybe_autocast(device, geometry_amp):
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
            geometry_records.append({key: base.tensor_item(geom[key]) for key in GEOMETRY_KEYS if key in geom})

    metrics = compute_controlled_metrics(args.dataset, labels, probs_or_preds, threshold=0.5)
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
    lines = [
        f"dataset: {args.dataset}",
        f"visual branch: CLIP-ViT",
        f"text branch: {'RoBERTa-base' if str(args.text_encoder_type).lower() == 'roberta' else 'CLIP-text'}",
        f"fusion type: {args.fusion_type}",
        f"method: {args.method}",
        f"branch tuning mode: {getattr(args, 'branch_tuning_mode', 'frozen')}",
        f"seed: {args.seed}",
        f"best epoch: {best_epoch}",
        f"best validation metric (Accuracy): {select_primary_metric(args.dataset, best_val_metrics):.6f}",
        f"test metric (Accuracy): {select_primary_metric(args.dataset, test_metrics):.6f}",
        f"Accuracy: {test_metrics.get('accuracy', float('nan')):.6f}",
    ]
    for key in ["fast_s_v", "fast_s_t", "fast_s_branch", "fast_s_f", "fast_fcf", "fast_rfcf"]:
        value = best_val_metrics.get(key, float("nan"))
        lines.append(f"mean {key}: {'nan' if (isinstance(value, float) and math.isnan(value)) else f'{float(value):.6f}'}")
    if str(getattr(args, "geometry_metric", "fast_fcf")) in {"exact_fcf", "both"}:
        for key in ["exact_s_v", "exact_s_t", "exact_s_branch", "exact_s_f", "exact_fcf", "exact_rfcf"]:
            value = best_val_metrics.get(key, float("nan"))
            lines.append(f"mean {key}: {'nan' if (isinstance(value, float) and math.isnan(value)) else f'{float(value):.6f}'}")
    lines.extend(
        [
            f"trainable parameter count: {int(trainable_report['trainable_parameters'])}",
            f"total wall-clock time: {elapsed['elapsed_hms']} ({elapsed['elapsed_seconds']}s)",
            f"peak gpu memory mb: {peak_gpu_memory_mb:.2f}" if not math.isnan(peak_gpu_memory_mb) else "peak gpu memory mb: nan",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    base.build_arg_parser = build_arg_parser
    base.encoder_setup_for_dataset = encoder_setup_for_dataset
    base.text_encoder_type_for_dataset = text_encoder_type_for_dataset
    base.build_experiment_name = build_experiment_name
    base.build_dataloaders = build_dataloaders
    base.build_model_and_tokenizers = build_model_and_tokenizers
    base.compute_controlled_metrics = compute_controlled_metrics
    base.select_primary_metric = select_primary_metric
    base.primary_metric_name = primary_metric_name
    base.save_predictions_csv = save_predictions_csv
    base.evaluate = evaluate
    base.build_summary_text = build_summary_text
    base.main()


if __name__ == "__main__":
    main()
