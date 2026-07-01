from __future__ import annotations

from typing import Optional

from transformers import AutoConfig

MODEL_ALIASES = {
    "qwen2vl2b": "qwen2vl2b",
    "qwen2_vl_2b": "qwen2vl2b",
    "qwen2-vl-2b": "qwen2vl2b",
    "qwen2-vl-2b-instruct": "qwen2vl2b",
    "qwen25vl3b": "qwen25vl3b",
    "qwen2.5-vl-3b": "qwen25vl3b",
    "qwen2.5-vl-3b-instruct": "qwen25vl3b",
    "qwen2_5_vl_3b": "qwen25vl3b",
    "internvl3": "internvl3_1b",
    "internvl3_1b": "internvl3_1b",
    "internvl3-1b": "internvl3_1b",
    "internvl3-1b-instruct": "internvl3_1b",
    "llava_onevision_qwen2_0_5b": "llava_onevision_qwen2_0_5b",
    "llava-onevision-qwen2-0.5b": "llava_onevision_qwen2_0_5b",
    "llava_onevision_qwen2_0.5b": "llava_onevision_qwen2_0_5b",
    "llava-onevision-qwen2-0.5b-ov-hf": "llava_onevision_qwen2_0_5b",
    "llava_onevision": "llava_onevision_qwen2_0_5b",
}

MODEL_FAMILY = {
    "qwen2vl2b": "qwen",
    "qwen25vl3b": "qwen",
    "internvl3_1b": "internvl",
    "llava_onevision_qwen2_0_5b": "llava_onevision",
}

def _normalize_model_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    key = str(value).strip().lower()
    if not key:
        return None
    return MODEL_ALIASES.get(key, key)


def _infer_model_name_from_path(model_path: str) -> Optional[str]:
    lowered = str(model_path or "").strip().lower()
    if "llava-onevision" in lowered or "llava_onevision" in lowered:
        return "llava_onevision_qwen2_0_5b"
    if "internvl3" in lowered:
        return "internvl3_1b"
    if "qwen2.5-vl" in lowered or "qwen25vl" in lowered:
        return "qwen25vl3b"
    if "qwen2-vl" in lowered:
        return "qwen2vl2b"
    return None


def _infer_model_name_from_config(model_path: str, trust_remote_code: bool, local_files_only: bool) -> Optional[str]:
    try:
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    except Exception:
        return None

    model_type = str(getattr(config, "model_type", "")).lower().strip()
    if model_type == "llava_onevision":
        return "llava_onevision_qwen2_0_5b"
    if model_type == "internvl_chat":
        return "internvl3_1b"
    if model_type == "qwen2_vl":
        return "qwen2vl2b"
    if model_type == "qwen2_5_vl":
        return "qwen25vl3b"

    architectures = " ".join(str(item).lower() for item in getattr(config, "architectures", []) or [])
    if "llavaonevision" in architectures:
        return "llava_onevision_qwen2_0_5b"
    if "internvlchatmodel" in architectures:
        return "internvl3_1b"
    if "qwen2_5_vl" in architectures:
        return "qwen25vl3b"
    if "qwen2vl" in architectures or "qwen2_vl" in architectures:
        return "qwen2vl2b"
    return None


def resolve_model_name(args) -> str:
    requested_name = _normalize_model_name(getattr(args, "model_name", None))
    if requested_name in MODEL_FAMILY:
        return requested_name

    requested_family = str(getattr(args, "model_family", "") or "").strip().lower()
    if requested_family == "llava_onevision":
        return "llava_onevision_qwen2_0_5b"
    if requested_family == "internvl":
        return "internvl3_1b"
    if requested_family == "qwen":
        inferred_from_path = _infer_model_name_from_path(getattr(args, "model_path", ""))
        return inferred_from_path or "qwen25vl3b"

    inferred = _infer_model_name_from_path(getattr(args, "model_path", ""))
    if inferred is not None:
        return inferred

    inferred = _infer_model_name_from_config(
        getattr(args, "model_path", ""),
        trust_remote_code=bool(getattr(args, "trust_remote_code", True)),
        local_files_only=bool(getattr(args, "local_files_only", True)),
    )
    if inferred is not None:
        return inferred

    raise ValueError(
        "Could not infer model_name from --model_name / --model_family / --model_path. "
        "Please pass --model_name {qwen2vl2b,qwen25vl3b,internvl3_1b,llava_onevision_qwen2_0_5b} explicitly."
    )


def build_model_wrapper(args):
    model_name = resolve_model_name(args)
    setattr(args, "model_name", model_name)
    setattr(args, "model_family", MODEL_FAMILY[model_name])
    if model_name == "llava_onevision_qwen2_0_5b":
        from .llava_onevision_wrapper import LlavaOnevisionWrapper

        return LlavaOnevisionWrapper(args)
    if model_name == "internvl3_1b":
        from .internvl3_wrapper import InternVL3Wrapper

        return InternVL3Wrapper(args)

    from .qwen25vl_wrapper import Qwen25VLWrapper

    return Qwen25VLWrapper(args)
