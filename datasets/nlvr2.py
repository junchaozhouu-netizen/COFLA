from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional

from .base import (
    BaseMultimodalDataset,
    coerce_image_to_pil,
    first_non_empty,
    is_missing,
    read_parquet_records,
    sample_records,
)


class NLVR2Dataset(BaseMultimodalDataset):
    task_name = "nlvr2"

    def __init__(
        self,
        root: str,
        split: str,
        max_samples: Optional[int] = None,
        seed: int = 42,
        variant: str = "balanced",
    ) -> None:
        self.root = root
        self.split = split
        self.variant = variant
        self.paths = self._resolve_parquet_paths(root, split, variant)
        self.path = self.paths[0]

        rows: List[Dict[str, Any]] = []
        for path in self.paths:
            rows.extend(read_parquet_records(path))
        self.records: List[Dict[str, Any]] = []
        self.missing_records: List[Dict[str, Any]] = []

        for row in rows:
            statement = str(first_non_empty(row, ["sentence", "statement", "question"], default="")).strip()
            label = self._normalize_label(first_non_empty(row, ["label", "answer", "gold_label"], default=None))
            left_image = first_non_empty(row, ["left_image", "image_left", "left", "image1", "left_img"], default=None)
            right_image = first_non_empty(row, ["right_image", "image_right", "right", "image2", "right_img"], default=None)
            sample_id = first_non_empty(row, ["identifier", "id", "uid", "sample_id"], default=len(self.records))

            if not statement or label is None or is_missing(left_image) or is_missing(right_image):
                self.missing_records.append({"id": sample_id})
                continue

            self.records.append(
                {
                    "id": sample_id,
                    "statement": statement,
                    "label": label,
                    "left_image": left_image,
                    "right_image": right_image,
                }
            )

        self.records = sample_records(self.records, max_samples=max_samples, seed=seed)

    @staticmethod
    def _normalize_label(value: Any) -> Optional[int]:
        if value is None:
            return None
        text = str(value).strip().lower()
        if text in {"true", "entailment", "yes", "1"}:
            return 1
        if text in {"false", "contradiction", "no", "0"}:
            return 0
        return None

    @classmethod
    def _resolve_parquet_paths(cls, root: str, split: str, variant: str) -> List[str]:
        split_norm = str(split).strip().lower().replace("-", "_")
        variant_norm = str(variant).strip().lower()
        if variant_norm not in {"balanced", "unbalanced"}:
            raise ValueError(f"Unsupported NLVR2 variant: {variant}")

        pattern_map = {
            "train": [f"{variant_norm}_train*.parquet"],
            "dev": [f"{variant_norm}_dev*.parquet"],
            "val": [f"{variant_norm}_dev*.parquet"],
            "validation": [f"{variant_norm}_dev*.parquet"],
            "test": [f"{variant_norm}_test_public*.parquet", f"{variant_norm}_test1*.parquet"],
            "test_public": [f"{variant_norm}_test_public*.parquet", f"{variant_norm}_test1*.parquet"],
            "test1": [f"{variant_norm}_test_public*.parquet", f"{variant_norm}_test1*.parquet"],
            "test_unseen": [f"{variant_norm}_test_unseen*.parquet", f"{variant_norm}_test2*.parquet"],
            "test2": [f"{variant_norm}_test_unseen*.parquet", f"{variant_norm}_test2*.parquet"],
        }
        patterns = pattern_map.get(split_norm, [f"*{split_norm}*.parquet"])

        candidates: List[str] = []
        search_roots = [os.path.join(root, "data"), root]
        for pattern in patterns:
            for search_root in search_roots:
                candidates.extend(sorted(glob.glob(os.path.join(search_root, pattern))))
                candidates.extend(sorted(glob.glob(os.path.join(search_root, "**", pattern), recursive=True)))
        candidates = list(dict.fromkeys(candidates))

        if not candidates:
            available: List[str] = []
            for search_root in search_roots:
                available.extend(glob.glob(os.path.join(search_root, "**", "*.parquet"), recursive=True))
            available = sorted(set(os.path.relpath(path, root) for path in available))
            detail = ""
            if available:
                preview = ", ".join(available[:8])
                more = "" if len(available) <= 8 else f", ... (+{len(available) - 8} more)"
                detail = f" Available parquet files: {preview}{more}"
            raise FileNotFoundError(
                f"Could not locate NLVR2 parquet for split={split} variant={variant} under {root}.{detail}"
            )
        return candidates

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.records[index]
        return {
            "id": item["id"],
            "statement": item["statement"],
            "label": int(item["label"]),
            "left_image": item["left_image"],
            "right_image": item["right_image"],
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "variant": self.variant,
            "path": self.path,
            "num_shards": len(self.paths),
            "paths": self.paths,
            "num_samples": len(self.records),
            "num_missing_skipped": len(self.missing_records),
        }


class NLVR2Collator:
    def __init__(self, wrapper, split: str, data_root: Optional[str] = None) -> None:
        self.wrapper = wrapper
        self.split = split
        self.data_root = data_root

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        sample_ids = []
        statements = []
        labels = []
        image_pairs = []

        for sample in batch:
            left_image = coerce_image_to_pil(sample["left_image"], root=self.data_root)
            right_image = coerce_image_to_pil(sample["right_image"], root=self.data_root)
            sample_ids.append(sample["id"])
            statements.append(sample["statement"])
            labels.append(int(sample["label"]))
            image_pairs.append((left_image, right_image))

        return self.wrapper.prepare_batch_by_dataset(
            dataset="nlvr2",
            sample_ids=sample_ids,
            statements=statements,
            labels=labels,
            image_pairs=image_pairs,
            split=self.split,
        )
