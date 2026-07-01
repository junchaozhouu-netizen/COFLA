from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset

from .base import BaseMultimodalDataset


SPLIT_FILENAME_MAP = {
    "train": "train.jsonl",
    "dev_seen": "dev_seen.jsonl",
    "dev_unseen": "dev_unseen.jsonl",
    "test_seen": "test_seen.jsonl",
    "test_unseen": "test_unseen.jsonl",
}


class HatefulMemesDataset(BaseMultimodalDataset):
    task_name = "hateful_memes"

    def __init__(
        self,
        root: str,
        split: str,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.root = root
        self.split = split
        if split not in SPLIT_FILENAME_MAP:
            raise ValueError(f"Unsupported Hateful Memes split: {split}")
        self.path = os.path.join(root, SPLIT_FILENAME_MAP[split])
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Split file not found: {self.path}")

        self.records: List[Dict[str, Any]] = []
        self.missing_records: List[Dict[str, Any]] = []

        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                img_rel = item.get("img", "")
                img_path = os.path.join(root, img_rel)
                item["image_path"] = img_path
                if not os.path.exists(img_path):
                    self.missing_records.append({
                        "id": item.get("id"),
                        "image_path": img_path,
                    })
                    continue
                self.records.append(item)

        if max_samples is not None and max_samples > 0 and len(self.records) > max_samples:
            rng = random.Random(seed)
            indices = list(range(len(self.records)))
            rng.shuffle(indices)
            indices = indices[:max_samples]
            self.records = [self.records[i] for i in indices]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.records[index]
        return {
            "id": item.get("id", index),
            "text": item.get("text", ""),
            "label": int(item.get("label", -1)) if item.get("label") is not None else -1,
            "image_path": item["image_path"],
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "num_samples": len(self.records),
            "num_missing_skipped": len(self.missing_records),
        }


class HatefulMemesCollator:
    def __init__(self, wrapper, split: str) -> None:
        self.wrapper = wrapper
        self.split = split

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = []
        ids = []
        texts = []
        labels = []
        for sample in batch:
            image = Image.open(sample["image_path"]).convert("RGB")
            images.append(image)
            ids.append(sample["id"])
            texts.append(sample["text"])
            labels.append(int(sample["label"]))
        return self.wrapper.prepare_batch_by_dataset(
            dataset="hateful_memes",
            sample_ids=ids,
            texts=texts,
            labels=labels,
            images=images,
            split=self.split,
        )
