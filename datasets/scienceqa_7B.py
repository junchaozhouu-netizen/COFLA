from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional

from .base_7B import (
    BaseMultimodalDataset,
    first_non_empty,
    is_missing,
    normalize_choice_list,
    qwen2vl_blank_image,
    read_parquet_records,
    safe_coerce_image_to_pil,
    sample_records,
    validate_image_value,
)


class ScienceQADataset(BaseMultimodalDataset):
    """7B-only ScienceQA dataset with robust image validation.

    It mirrors datasets/scienceqa.py, but handles samples whose image exists as a
    field but is missing/unreadable/zero-size.  When image_only=True, invalid-image
    samples are skipped; otherwise they are kept as text-only samples.
    """

    task_name = "scienceqa"

    def __init__(
        self,
        root: str,
        split: str,
        max_samples: Optional[int] = None,
        seed: int = 42,
        image_only: bool = False,
    ) -> None:
        self.root = root
        self.split = split
        self.path = self._resolve_parquet_path(root, split)
        self.image_only = bool(image_only)

        rows = read_parquet_records(self.path)
        self.num_raw_rows = len(rows)
        self.records: List[Dict[str, Any]] = []
        self.missing_records: List[Dict[str, Any]] = []
        self.invalid_image_records: List[Dict[str, Any]] = []
        self.num_without_image = 0
        self.num_invalid_image = 0
        self.num_image_only_skipped = 0
        self.num_invalid_image_as_text_only = 0

        for row in rows:
            question = str(first_non_empty(row, ["question", "query", "problem"], default="")).strip()
            choices = normalize_choice_list(first_non_empty(row, ["choices", "options"], default=[]))
            label = self._normalize_answer(first_non_empty(row, ["answer", "label", "target"], default=None), choices)
            image = first_non_empty(row, ["image", "image_path", "image_file", "picture"], default=None)
            hint = str(first_non_empty(row, ["hint", "context"], default="")).strip()
            sample_id = first_non_empty(row, ["id", "problem_id", "pid", "sample_id"], default=len(self.records))

            if not question or not choices or label is None:
                self.missing_records.append({"id": sample_id})
                continue

            if is_missing(image):
                image = None
                self.num_without_image += 1
                if self.image_only:
                    self.num_image_only_skipped += 1
                    continue
            else:
                ok, reason = validate_image_value(image, root=root)
                if not ok:
                    self.num_invalid_image += 1
                    self.invalid_image_records.append({"id": sample_id, "reason": reason})
                    if self.image_only:
                        self.num_image_only_skipped += 1
                        continue
                    image = None
                    self.num_invalid_image_as_text_only += 1

            self.records.append(
                {
                    "id": sample_id,
                    "question": question,
                    "choices": choices,
                    "label": label,
                    "hint": hint,
                    "image": image,
                }
            )

        self.num_after_filter_before_sampling = len(self.records)
        self.records = sample_records(self.records, max_samples=max_samples, seed=seed)

    @staticmethod
    def _normalize_answer(value: Any, choices: List[str]) -> Optional[int]:
        if value is None:
            return None

        if isinstance(value, bool):
            value = int(value)

        if isinstance(value, int):
            return int(value) if 0 <= int(value) < len(choices) else None

        text = str(value).strip()
        if not text:
            return None

        if text.isdigit():
            idx = int(text)
            return idx if 0 <= idx < len(choices) else None

        if len(text) == 1 and text.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            idx = ord(text.upper()) - ord("A")
            return idx if 0 <= idx < len(choices) else None

        lowered_choices = [choice.strip().lower() for choice in choices]
        if text.lower() in lowered_choices:
            return lowered_choices.index(text.lower())
        return None

    @classmethod
    def _resolve_parquet_path(cls, root: str, split: str) -> str:
        split_norm = str(split).strip().lower().replace("-", "_")
        pattern_map = {
            "train": ["train*.parquet"],
            "val": ["validation*.parquet"],
            "validation": ["validation*.parquet"],
            "dev": ["validation*.parquet"],
            "test": ["test*.parquet"],
        }
        patterns = pattern_map.get(split_norm, [f"{split_norm}*.parquet"])

        candidates: List[str] = []
        for pattern in patterns:
            candidates.extend(sorted(glob.glob(os.path.join(root, "data", pattern))))
            candidates.extend(sorted(glob.glob(os.path.join(root, pattern))))

        if not candidates:
            raise FileNotFoundError(f"Could not locate ScienceQA parquet for split={split} under {root}")
        return candidates[0]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.records[index]
        return {
            "id": item["id"],
            "question": item["question"],
            "choices": list(item["choices"]),
            "label": int(item["label"]),
            "hint": item["hint"],
            "image": item["image"],
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "path": self.path,
            "image_only": self.image_only,
            "num_raw_rows": self.num_raw_rows,
            "num_samples": len(self.records),
            "num_after_filter_before_sampling": self.num_after_filter_before_sampling,
            "num_missing_skipped": len(self.missing_records),
            "num_without_image": self.num_without_image,
            "num_invalid_image": self.num_invalid_image,
            "num_image_only_skipped": self.num_image_only_skipped,
            "num_invalid_image_as_text_only": self.num_invalid_image_as_text_only,
        }


class ScienceQACollator:
    def __init__(self, wrapper, split: str, data_root: Optional[str] = None, include_hint: bool = True) -> None:
        self.wrapper = wrapper
        self.split = split
        self.data_root = data_root
        self.include_hint = include_hint

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        sample_ids = []
        questions = []
        choices = []
        labels = []
        hints = []
        images = []

        for sample in batch:
            sample_ids.append(sample["id"])
            questions.append(sample["question"])
            choices.append(list(sample["choices"]))
            labels.append(int(sample["label"]))
            hints.append(sample.get("hint", "") if self.include_hint else "")
            image = sample.get("image")
            if image is not None:
                pil_image = safe_coerce_image_to_pil(image, root=self.data_root)
                if pil_image is None:
                    # Last-resort fallback: keep the batch valid for Qwen2-VL-7B.
                    pil_image = qwen2vl_blank_image()
                images.append(pil_image)
            else:
                images.append(None)

        return self.wrapper.prepare_batch_by_dataset(
            dataset="scienceqa",
            sample_ids=sample_ids,
            questions=questions,
            choices=choices,
            labels=labels,
            hints=hints,
            images=images,
            split=self.split,
        )
