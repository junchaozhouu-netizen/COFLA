from __future__ import annotations

import ast
import io
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError
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


def _resolve_image_path(path: str, root: Optional[str] = None) -> str:
    if root is not None and not os.path.isabs(path):
        return os.path.join(root, path)
    return path


def _validate_pil_image(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: width={width}, height={height}")
    return image.convert("RGB")


def qwen2vl_blank_image(size: int = 56) -> Image.Image:
    """Return a tiny valid RGB image as a last-resort 7B fallback."""
    size = max(int(size), 28)
    return Image.new("RGB", (size, size), color=(255, 255, 255))


def qwen2vl_sanitize_pil_image(
    image: Image.Image,
    *,
    min_side: int = 28,
    max_aspect_ratio: float = 8.0,
) -> Optional[Image.Image]:
    """Make a PIL image safe for Qwen2-VL's smart_resize.

    Qwen2-VL rounds resized image dimensions to multiples of 28. With very
    small max_pixels settings such as 6272, extremely wide/tall images can make
    the shorter resized side round down to 0 inside the Transformers image
    processor, causing: ValueError: height and width must be > 0.

    We do not distort the image.  We only paste it onto a white square canvas
    when the original image is too thin or has an extreme aspect ratio.  This is
    used only by 7B-specific datasets/collators.
    """
    try:
        image = image.convert("RGB")
        width, height = image.size
    except Exception:
        return None

    if width <= 0 or height <= 0:
        return None

    min_side = max(int(min_side), 1)
    short = min(width, height)
    long = max(width, height)
    aspect = float(long) / float(short) if short > 0 else float("inf")

    if short < min_side or aspect > float(max_aspect_ratio):
        canvas_side = max(width, height, min_side)
        canvas = Image.new("RGB", (canvas_side, canvas_side), color=(255, 255, 255))
        x = (canvas_side - width) // 2
        y = (canvas_side - height) // 2
        canvas.paste(image, (x, y))
        return canvas

    return image


def coerce_image_to_pil(value: Any, root: Optional[str] = None) -> Optional[Image.Image]:
    """Original-style image conversion, but with explicit nonzero-size validation.

    This keeps the same public function name as datasets/base.py while making the
    7B-only dataset path safer before Qwen2-VL's image processor receives inputs.
    """
    value = maybe_parse_literal(value)

    if is_missing(value):
        return None

    if isinstance(value, Image.Image):
        return _validate_pil_image(value)

    if isinstance(value, np.ndarray):
        array = value
        if array.size == 0:
            raise ValueError("Empty ndarray image")
        if array.dtype != np.uint8:
            array = array.astype(np.uint8)
        return _validate_pil_image(Image.fromarray(array))

    if isinstance(value, bytes):
        with Image.open(io.BytesIO(value)) as img:
            img.load()
            return _validate_pil_image(img)

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
        path = _resolve_image_path(value, root=root)
        if os.path.exists(path):
            with Image.open(path) as img:
                img.load()
                return _validate_pil_image(img)
        raise FileNotFoundError(f"Image path does not exist: {path}")

    raise ValueError(f"Unsupported image value type: {type(value)!r}")


def safe_coerce_image_to_pil(value: Any, root: Optional[str] = None) -> Optional[Image.Image]:
    """Return RGB PIL image if valid; otherwise return None.

    Use only in 7B-specific datasets/collators, so legacy 0.5B/2B/control-late-fusion
    data behavior remains untouched.
    """
    try:
        image = coerce_image_to_pil(value, root=root)
    except (FileNotFoundError, OSError, ValueError, UnidentifiedImageError, TypeError):
        return None
    if image is None:
        return None
    return qwen2vl_sanitize_pil_image(image)


def is_valid_image_value(value: Any, root: Optional[str] = None) -> bool:
    image = safe_coerce_image_to_pil(value, root=root)
    if image is None:
        return False
    try:
        image.close()
    except Exception:
        pass
    return True


def validate_image_value(value: Any, root: Optional[str] = None) -> Tuple[bool, str]:
    try:
        image = coerce_image_to_pil(value, root=root)
        if image is None:
            return False, "missing"
        width, height = image.size
        if width <= 0 or height <= 0:
            try:
                image.close()
            except Exception:
                pass
            return False, f"nonpositive_size:{width}x{height}"
        safe_image = qwen2vl_sanitize_pil_image(image)
        try:
            image.close()
        except Exception:
            pass
        if safe_image is None:
            return False, "qwen2vl_sanitize_failed"
        try:
            safe_image.close()
        except Exception:
            pass
        return True, "ok"
    except FileNotFoundError:
        return False, "file_not_found"
    except UnidentifiedImageError:
        return False, "unidentified_image"
    except OSError:
        return False, "os_error"
    except ValueError as exc:
        return False, f"value_error:{str(exc)[:80]}"
    except TypeError as exc:
        return False, f"type_error:{str(exc)[:80]}"
    except Exception as exc:
        return False, f"unexpected:{type(exc).__name__}"
