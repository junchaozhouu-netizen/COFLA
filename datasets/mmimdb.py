from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, UnidentifiedImageError

from .base import BaseMultimodalDataset, coerce_image_to_pil, first_non_empty, maybe_parse_literal, sample_records


class MMIMDbDataset(BaseMultimodalDataset):
    task_name = "mmimdb"

    # The Qwen2-VL image processor requires image height and width to be larger than the patch factor.
    # A threshold of 32 is used to filter abnormally small images such as 5x7 inputs.
    min_valid_image_size = 32

    def __init__(
        self,
        root: str,
        split: str,
        max_samples: Optional[int] = None,
        seed: int = 42,
        label_names: Optional[Sequence[str]] = None,
        split_file: str = "split.json",
    ) -> None:
        self.root = root
        self.split = split
        self.split_path = self._resolve_split_path(root, split_file)
        self.dataset_dir = self._resolve_dataset_dir(root)
        self.label_names = list(label_names) if label_names else self.infer_label_names(root, split_file=split_file)
        if not self.label_names:
            raise ValueError(f"Could not infer any MMIMDb label names under {root}")
        self.label_to_index = {name.lower(): idx for idx, name in enumerate(self.label_names)}

        split_ids = self._load_split_ids(self.split_path, split)
        self.records: List[Dict[str, Any]] = []
        self.missing_records: List[Dict[str, Any]] = []
        self.skipped_log_path: Optional[str] = None

        for sample_id in split_ids:
            sample_key = self._normalize_sample_id(sample_id)
            meta_path = os.path.join(self.dataset_dir, f"{sample_key}.json")
            image_path = self._resolve_image_path(self.dataset_dir, sample_key)

            if not os.path.exists(meta_path) or image_path is None:
                self.missing_records.append(
                    {
                        "id": sample_key,
                        "reason": "missing_json_or_image",
                        "meta_path": meta_path,
                        "image_path": image_path,
                    }
                )
                continue

            is_valid_image, image_reason, image_width, image_height = self._check_valid_image_file(
                image_path,
                min_size=self.min_valid_image_size,
            )

            if not is_valid_image:
                self.missing_records.append(
                    {
                        "id": sample_key,
                        "reason": image_reason,
                        "image_path": image_path,
                        "width": image_width,
                        "height": image_height,
                    }
                )
                continue

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            genres = self._normalize_genres(first_non_empty(meta, ["genres", "genre", "labels", "target"], default=[]))
            title = str(first_non_empty(meta, ["title", "movie_title", "name"], default="")).strip()
            plot = self._normalize_plot(first_non_empty(meta, ["plot", "plots", "synopsis", "storyline", "description"], default=""))

            multi_hot = [0] * len(self.label_names)
            active = 0
            for genre in genres:
                idx = self.label_to_index.get(genre.lower())
                if idx is None:
                    continue
                if multi_hot[idx] == 0:
                    multi_hot[idx] = 1
                    active += 1

            if active == 0:
                self.missing_records.append(
                    {
                        "id": sample_key,
                        "reason": "no_known_labels",
                        "meta_path": meta_path,
                        "image_path": image_path,
                    }
                )
                continue

            self.records.append(
                {
                    "id": sample_key,
                    "title": title,
                    "plot": plot,
                    "labels": multi_hot,
                    "image_path": image_path,
                }
            )

        self.records = sample_records(self.records, max_samples=max_samples, seed=seed)
        self._write_skipped_records_log()

    @staticmethod
    def parse_label_names(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    @classmethod
    def infer_label_names(cls, root: str, split_file: str = "split.json") -> List[str]:
        dataset_dir = cls._resolve_dataset_dir(root)
        label_map: Dict[str, str] = {}

        for entry in sorted(os.listdir(dataset_dir)):
            if not entry.lower().endswith(".json"):
                continue
            path = os.path.join(dataset_dir, entry)
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            genres = cls._normalize_genres(first_non_empty(meta, ["genres", "genre", "labels", "target"], default=[]))
            for genre in genres:
                key = genre.lower()
                if key not in label_map:
                    label_map[key] = genre

        if not label_map and os.path.exists(cls._resolve_split_path(root, split_file)):
            return []
        return [label_map[key] for key in sorted(label_map.keys())]

    @staticmethod
    def _resolve_dataset_dir(root: str) -> str:
        candidates = [
            os.path.join(root, "dataset"),
            os.path.join(root, "data", "dataset"),
            root,
        ]
        for path in candidates:
            if os.path.isdir(path):
                return path
        raise FileNotFoundError(f"Could not locate MMIMDb dataset directory under {root}")

    @staticmethod
    def _resolve_split_path(root: str, split_file: str) -> str:
        candidates = [
            os.path.join(root, split_file),
            os.path.join(root, "data", split_file),
            os.path.join(os.path.dirname(root), split_file),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]

    @staticmethod
    def _resolve_image_path(dataset_dir: str, sample_key: str) -> Optional[str]:
        for ext in [".jpeg", ".jpg", ".png", ".webp"]:
            path = os.path.join(dataset_dir, f"{sample_key}{ext}")
            if os.path.exists(path):
                return path
        return None

    @classmethod
    def _check_valid_image_file(
        cls,
        image_path: str,
        min_size: Optional[int] = None,
    ) -> Tuple[bool, str, Optional[int], Optional[int]]:
        """
        Check whether an MM-IMDb image is valid for vision-language processors.

        Some processors require both height and width to exceed their patch factor.
        MM-IMDb may contain abnormally small images, which can fail during data loading.
        This routine filters missing images, unreadable images, corrupted images, and
        images whose width or height is smaller than min_size.
        """
        threshold = int(min_size or cls.min_valid_image_size)

        if not image_path or not os.path.exists(image_path):
            return False, "missing_image", None, None

        try:
            # verify() checks file integrity; reopen the image afterward to read its size.
            with Image.open(image_path) as img:
                img.verify()

            with Image.open(image_path) as img:
                width, height = img.size

            if width < threshold or height < threshold:
                return False, "invalid_small_image", width, height

            return True, "ok", width, height

        except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as exc:
            return False, f"invalid_or_corrupt_image:{type(exc).__name__}", None, None

    @classmethod
    def _normalize_split_name(cls, split: str) -> str:
        split_norm = str(split).strip().lower().replace("-", "_")
        alias_map = {
            "val": "dev",
            "validation": "dev",
            "dev_seen": "dev",
            "valid": "dev",
        }
        return alias_map.get(split_norm, split_norm)

    @classmethod
    def _normalize_sample_id(cls, value: Any) -> str:
        text = str(value).strip()
        if not text:
            return text
        return os.path.splitext(os.path.basename(text))[0]

    @classmethod
    def _extract_entry_id(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, (str, int)):
            text = cls._normalize_sample_id(value)
            return text or None
        if isinstance(value, dict):
            sample_id = first_non_empty(
                value,
                ["id", "imdb_id", "movie_id", "sample_id", "key", "image_id", "filename", "file_name"],
                default=None,
            )
            if sample_id is not None:
                text = cls._normalize_sample_id(sample_id)
                return text or None
        return None

    @classmethod
    def _extract_ids_from_split_value(cls, value: Any) -> List[str]:
        value = maybe_parse_literal(value)
        if value is None:
            return []
        if isinstance(value, dict):
            nested = first_non_empty(value, ["ids", "samples", "movies", "items", "data"], default=None)
            if nested is not None:
                return cls._extract_ids_from_split_value(nested)

            ids: List[str] = []
            for key, item in value.items():
                if isinstance(item, bool):
                    if item:
                        ids.append(cls._normalize_sample_id(key))
                    continue
                if isinstance(item, (int, float)) and int(item) == 1:
                    ids.append(cls._normalize_sample_id(key))
                    continue
                sample_id = cls._extract_entry_id(item)
                if sample_id is not None:
                    ids.append(sample_id)
            return ids

        if isinstance(value, (list, tuple)):
            ids = []
            for item in value:
                sample_id = cls._extract_entry_id(item)
                if sample_id is not None:
                    ids.append(sample_id)
            return ids

        sample_id = cls._extract_entry_id(value)
        return [sample_id] if sample_id is not None else []

    @classmethod
    def _load_split_ids(cls, split_path: str, split: str) -> List[str]:
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"MMIMDb split file not found: {split_path}")

        with open(split_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        split_norm = cls._normalize_split_name(split)
        target_keys = {split_norm}
        if split_norm == "dev":
            target_keys.update({"val", "validation"})

        if isinstance(data, dict):
            for key, value in data.items():
                if cls._normalize_split_name(key) in target_keys:
                    ids = cls._extract_ids_from_split_value(value)
                    if ids:
                        return ids

            ids = []
            for key, value in data.items():
                value_split = cls._normalize_split_name(str(value))
                if value_split in target_keys:
                    ids.append(cls._normalize_sample_id(key))
                    continue
                if isinstance(value, dict):
                    item_split = cls._normalize_split_name(str(first_non_empty(value, ["split", "subset", "set"], default="")))
                    if item_split in target_keys:
                        sample_id = cls._extract_entry_id(value) or cls._normalize_sample_id(key)
                        ids.append(sample_id)
            if ids:
                return ids

        if isinstance(data, list):
            ids = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                item_split = cls._normalize_split_name(str(first_non_empty(item, ["split", "subset", "set"], default="")))
                if item_split not in target_keys:
                    continue
                sample_id = cls._extract_entry_id(item)
                if sample_id is not None:
                    ids.append(sample_id)
            if ids:
                return ids

        raise ValueError(f"Could not extract MMIMDb ids for split={split} from {split_path}")

    @classmethod
    def _normalize_genres(cls, value: Any) -> List[str]:
        value = maybe_parse_literal(value)
        if value is None:
            return []

        if isinstance(value, dict):
            items: Iterable[Any] = [key for key, flag in value.items() if bool(flag)]
        elif isinstance(value, (list, tuple, set)):
            items = value
        else:
            text = str(value).strip()
            if not text:
                return []
            if "," in text:
                items = [part.strip() for part in text.split(",")]
            elif "|" in text:
                items = [part.strip() for part in text.split("|")]
            elif ";" in text:
                items = [part.strip() for part in text.split(";")]
            else:
                items = [text]

        genres: List[str] = []
        seen = set()
        for item in items:
            text = str(item).strip()
            key = text.lower()
            if text and key not in seen:
                genres.append(text)
                seen.add(key)
        return genres

    @classmethod
    def _normalize_plot(cls, value: Any) -> str:
        value = maybe_parse_literal(value)
        if value is None:
            return ""
        if isinstance(value, dict):
            nested = first_non_empty(value, ["plot", "text", "summary", "description"], default="")
            return cls._normalize_plot(nested)
        if isinstance(value, (list, tuple)):
            texts = [str(item).strip() for item in value if str(item).strip()]
            if not texts:
                return ""
            return max(texts, key=len)
        return str(value).strip()

    def _write_skipped_records_log(self) -> None:
        """
        Write skipped samples to a log file for later data-quality inspection.
        """
        if not self.missing_records:
            return

        safe_split = str(self.split).strip().replace("/", "_").replace("\\", "_")
        log_path = os.path.join(
            self.root,
            f"mmimdb_{safe_split}_skipped_records.txt",
        )

        try:
            with open(log_path, "w", encoding="utf-8") as f:
                for item in self.missing_records:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            self.skipped_log_path = log_path
        except OSError:
            self.skipped_log_path = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.records[index]
        return {
            "id": item["id"],
            "title": item["title"],
            "plot": item["plot"],
            "labels": list(item["labels"]),
            "image_path": item["image_path"],
        }

    def summary(self) -> Dict[str, Any]:
        small_images = [
            item for item in self.missing_records
            if str(item.get("reason", "")).startswith("invalid_small_image")
        ]
        corrupt_images = [
            item for item in self.missing_records
            if str(item.get("reason", "")).startswith("invalid_or_corrupt_image")
        ]
        missing_json_or_image = [
            item for item in self.missing_records
            if str(item.get("reason", "")) == "missing_json_or_image"
        ]
        no_known_labels = [
            item for item in self.missing_records
            if str(item.get("reason", "")) == "no_known_labels"
        ]

        return {
            "split": self.split,
            "split_path": self.split_path,
            "dataset_dir": self.dataset_dir,
            "num_samples": len(self.records),
            "num_labels": len(self.label_names),
            "num_missing_skipped": len(self.missing_records),
            "num_missing_json_or_image_skipped": len(missing_json_or_image),
            "num_small_image_skipped": len(small_images),
            "num_corrupt_image_skipped": len(corrupt_images),
            "num_no_known_labels_skipped": len(no_known_labels),
            "min_valid_image_size": self.min_valid_image_size,
            "skipped_log_path": self.skipped_log_path,
        }


class MMIMDbCollator:
    def __init__(self, wrapper, split: str, data_root: Optional[str] = None) -> None:
        self.wrapper = wrapper
        self.split = split
        self.data_root = data_root

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        sample_ids = []
        titles = []
        plots = []
        labels = []
        images = []

        for sample in batch:
            sample_ids.append(sample["id"])
            titles.append(sample.get("title", ""))
            plots.append(sample.get("plot", ""))
            labels.append(list(sample["labels"]))
            images.append(coerce_image_to_pil(sample["image_path"], root=self.data_root))

        return self.wrapper.prepare_batch_by_dataset(
            dataset="mmimdb",
            sample_ids=sample_ids,
            titles=titles,
            plots=plots,
            labels=labels,
            images=images,
            split=self.split,
        )