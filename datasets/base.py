from __future__ import annotations

import ast
import io
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class BaseMultimodalDataset(Dataset):
    task_name: str = "base"

    def summary(self) -> Dict[str, Any]:
        return {}


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def first_non_empty(mapping: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key not in mapping:
            continue
        value = mapping[key]
        if not is_missing(value):
            return value
    return default


def sample_records(records: List[Dict[str, Any]], max_samples: Optional[int], seed: int) -> List[Dict[str, Any]]:
    if max_samples is None or max_samples <= 0 or len(records) <= max_samples:
        return records
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    indices = indices[:max_samples]
    return [records[i] for i in indices]


def read_parquet_records(path: str) -> List[Dict[str, Any]]:
    import pandas as pd

    df = pd.read_parquet(path)
    return df.to_dict(orient="records")


def maybe_parse_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return ast.literal_eval(text)
    except Exception:
        return value


def normalize_choice_list(value: Any) -> List[str]:
    value = maybe_parse_literal(value)
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, np.ndarray):
        return [str(item).strip() for item in value.tolist() if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    parts = [part.strip() for part in text.split("|||")]
    parts = [part for part in parts if part]
    if parts:
        return parts
    return [text]


def coerce_image_to_pil(value: Any, root: Optional[str] = None) -> Optional[Image.Image]:
    value = maybe_parse_literal(value)

    if is_missing(value):
        return None

    if isinstance(value, Image.Image):
        return value.convert("RGB")

    if isinstance(value, np.ndarray):
        array = value
        if array.dtype != np.uint8:
            array = array.astype(np.uint8)
        return Image.fromarray(array).convert("RGB")

    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value)).convert("RGB")

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return coerce_image_to_pil(value.get("bytes"), root=root)
        if value.get("path"):
            return coerce_image_to_pil(value.get("path"), root=root)
        if value.get("array") is not None:
            return coerce_image_to_pil(value.get("array"), root=root)
        if value.get("image") is not None:
            return coerce_image_to_pil(value.get("image"), root=root)

    if hasattr(value, "as_py") and callable(value.as_py):
        return coerce_image_to_pil(value.as_py(), root=root)

    if isinstance(value, str):
        path = value
        if root is not None and not os.path.isabs(path):
            path = os.path.join(root, path)
        if os.path.exists(path):
            return Image.open(path).convert("RGB")

    raise ValueError(f"Unsupported image value type: {type(value)!r}")
