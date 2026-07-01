from __future__ import annotations

import contextlib
import json
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoProcessor, BitsAndBytesConfig

try:
    import bitsandbytes as bnb
except Exception:
    bnb = None

try:
    from transformers import Qwen2VLForConditionalGeneration as _Qwen2VLModelClass
except Exception:
    _Qwen2VLModelClass = None

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as _Qwen25VLModelClass
except Exception:
    _Qwen25VLModelClass = None

try:
    from transformers import AutoModelForVision2Seq as _AutoVision2Seq
except Exception:
    _AutoVision2Seq = None

try:
    from transformers import AutoModelForImageTextToText as _AutoImageTextToText
except Exception:
    _AutoImageTextToText = None

from .prompting import hateful_memes_yes_no_prompt, mmimdb_multilabel_prompt, nlvr2_yes_no_prompt, scienceqa_multichoice_prompt
from .trainable_scope import (
    collect_trainable_parameter_report,
    count_parameters,
    freeze_all_parameters,
    normalize_scope_name,
    resolve_trainable_scope,
    unfreeze_modules_by_prefix,
)


@dataclass
class LabelWordInfo:
    texts: Tuple[str, ...]
    token_ids: Tuple[int, ...]
    used_fallback: bool
    positive_text: str = ""
    negative_text: str = ""
    positive_token_id: int = -1
    negative_token_id: int = -1


@dataclass
class TaskHeadForwardOutput:
    loss: Optional[torch.Tensor]
    logits: torch.Tensor


class Qwen25VLWrapper(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._input_grad_hook_handle = None
        self.method_name = str(args.method).lower()
        self.dataset_name = str(args.dataset).lower()
        self.model_name = str(getattr(args, "model_name", "") or "").lower()
        self.model_family = str(getattr(args, "model_family", "qwen") or "qwen").lower()
        self.quantization_mode = str(getattr(args, "quantization", "none")).lower()
        self.local_files_only = bool(getattr(args, "local_files_only", True))
        self.trust_remote_code = bool(getattr(args, "trust_remote_code", True))
        self.preview_limit = int(getattr(args, "preview_limit", 50))
        self.trainable_scope_name = str(getattr(args, "trainable_scope", "auto") or "auto")
        self.num_lm_fusion_layers = int(getattr(args, "num_lm_fusion_layers", getattr(args, "num_fusion_layers", 4)) or 4)
        self.adapter_methods = {
            "vanilla_lora",
            "sam_lora",
            "esam_lora",
            "msam_lora",
            "masam_lora",
            "dgl_lora",
            "cofla",
            "fast_cofla",
        }
        self.is_adapter_method = self.method_name in self.adapter_methods
        self.is_quantized_model = False

        if self.dataset_name == "mmimdb" and str(args.task_format).lower() != "task_head_cls":
            raise ValueError("MMIMDb is a multi-label task and must use --task_format task_head_cls.")

        self.device_obj = torch.device(
            f"cuda:{args.device}" if torch.cuda.is_available() and str(args.device) != "cpu" else "cpu"
        )
        self._validate_quantization_configuration()
        self.compute_dtype = self._resolve_dtype(args.precision)

        # First read the config to determine whether the checkpoint is qwen2_vl or qwen2_5_vl.
        self.hf_config = AutoConfig.from_pretrained(
            args.model_path,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
        )
        self.model_type = self._infer_model_type(self.hf_config)

        # The processor is still loaded through AutoProcessor.
        processor_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
        }
        # Avoid silent differences caused by default slow/fast processor switching.
        try:
            self.processor = AutoProcessor.from_pretrained(
                args.model_path,
                use_fast=False,
                **processor_kwargs,
            )
        except TypeError:
            self.processor = AutoProcessor.from_pretrained(
                args.model_path,
                **processor_kwargs,
            )

        if getattr(self.processor, "tokenizer", None) is not None:
            self.processor.tokenizer.padding_side = "right"
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        tokenizer = getattr(self.processor, "tokenizer", None)
        self.special_token_ids = tuple(
            sorted({int(token_id) for token_id in getattr(tokenizer, "all_special_ids", []) if token_id is not None})
        )
        self._geometry_warning_keys = set()

        # Select the correct model class according to model_type.
        model_cls, need_trust_remote_code = self._resolve_model_class()
        model_load_kwargs = {
            "torch_dtype": self.compute_dtype if self.device_obj.type == "cuda" else torch.float32,
            "local_files_only": self.local_files_only,
        }
        if need_trust_remote_code:
            model_load_kwargs["trust_remote_code"] = self.trust_remote_code
        if getattr(args, "deterministic", True):
            model_load_kwargs["attn_implementation"] = "eager"

        quantization_config = self._build_quantization_config()
        if quantization_config is not None:
            model_load_kwargs["quantization_config"] = quantization_config
            model_load_kwargs["device_map"] = {"": self._get_quantized_device_map_target()}
            self.is_quantized_model = True

        try:
            self.model = model_cls.from_pretrained(args.model_path, **model_load_kwargs)
        except TypeError:
            model_load_kwargs.pop("attn_implementation", None)
            self.model = model_cls.from_pretrained(args.model_path, **model_load_kwargs)

        self._enable_model_gradient_checkpointing()

        self.config = self.model.config
        self.image_token_id = int(getattr(self.config, "image_token_id", 151655))
        self.text_proxy_excluded_token_ids = tuple(token_id for token_id in self.special_token_ids if token_id != self.image_token_id)
        self.label_info = self._resolve_label_words()
        self.scope_config = resolve_trainable_scope(
            self.model,
            self.model_family,
            self.trainable_scope_name,
            num_fusion_layers=int(getattr(args, "num_fusion_layers", 4)),
            num_lm_fusion_layers=self.num_lm_fusion_layers,
            num_branch_vision_layers=int(getattr(args, "num_branch_vision_layers", 2)),
            num_branch_lm_layers=int(getattr(args, "num_branch_lm_layers", 2)),
        )
        self.fusion_linear_module_names = list(self.scope_config.fusion_module_names)
        self.branch_visual_module_names = list(self.scope_config.branch_visual_module_names)
        self.branch_text_module_names = list(self.scope_config.branch_text_module_names)
        self.branch_linear_module_names = list(self.scope_config.branch_module_names)
        self.lora_target_module_names = list(self.scope_config.lora_target_modules)
        if not self.lora_target_module_names:
            raise RuntimeError("Could not discover any LoRA target modules. Check the model structure and trainable scope.")

        self._task_head = None
        if args.task_format == "task_head_cls":
            hidden = self._resolve_hidden_size()
            self._task_head = nn.Linear(hidden, args.num_labels)
            self.add_module("task_head", self._task_head)

    @staticmethod
    def _resolve_dtype(precision: str):
        precision = str(precision).lower()
        if precision == "bf16":
            return torch.bfloat16
        if precision == "fp16":
            return torch.float16
        return torch.float32

    def _warn_geometry_once(self, key: str, message: str) -> None:
        if key in self._geometry_warning_keys:
            return
        warnings.warn(message, stacklevel=2)
        self._geometry_warning_keys.add(key)

    def _geometry_eps(self) -> float:
        return float(max(getattr(self.args, "fcf_eps", 1e-8), 1e-12))

    def _branch_aggregation_mode(self) -> str:
        mode = str(getattr(self.args, "branch_sharpness_aggregation", "symmetric")).lower().strip()
        if mode in {"symmetric", "sym", "paper"}:
            return "symmetric"
        if mode in {"weighted", "legacy", "legacy_weighted"}:
            return "weighted"
        raise ValueError(f"Unsupported branch sharpness aggregation mode: {mode}")

    def _aggregate_branch_sharpness(self, s_v: torch.Tensor, s_t: torch.Tensor) -> torch.Tensor:
        # The paper default is a symmetric average over the two branch-side proxies.
        if self._branch_aggregation_mode() == "weighted":
            return self.args.alpha_v * s_v + self.args.alpha_t * s_t
        return 0.5 * (s_v + s_t)

    def _aggregate_branch_linear_proxy(self, gv_norm: torch.Tensor, gt_norm: torch.Tensor) -> torch.Tensor:
        # Keep the FAST-COFLA linearized proxy aligned with the same branch aggregation default.
        if self._branch_aggregation_mode() == "weighted":
            return self.args.alpha_v * self.args.rho_v * gv_norm + self.args.alpha_t * self.args.rho_t * gt_norm
        return 0.5 * (self.args.rho_v * gv_norm + self.args.rho_t * gt_norm)

    def _stabilize_non_negative_sharpness(self, value: torch.Tensor, name: str) -> torch.Tensor:
        if not bool(torch.isfinite(value).all()):
            self._warn_geometry_once(
                f"nonfinite_{name}",
                f"{name} produced a non-finite value before FCF/RFCF computation; replacing it with 0 for numerical stability.",
            )
        value = torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0)
        scalar = float(value.detach().cpu().item())
        if scalar < 0.0:
            self._warn_geometry_once(
                f"negative_{name}",
                f"{name} became negative before FCF/RFCF computation; clamping it to preserve the paper definition.",
            )
        return value.clamp_min(0.0)

    @staticmethod
    def _infer_model_type(config) -> str:
        model_type = str(getattr(config, "model_type", "")).lower().strip()
        if model_type:
            return model_type

        architectures = getattr(config, "architectures", None)
        if architectures:
            joined = " ".join(str(x).lower() for x in architectures)
            if "qwen2_5_vl" in joined:
                return "qwen2_5_vl"
            if "qwen2vl" in joined or "qwen2_vl" in joined:
                return "qwen2_vl"

        return ""

    def _enable_model_gradient_checkpointing(self) -> None:
        if not getattr(self.args, "gradient_checkpointing", False):
            return
        enable_fn = getattr(self.model, "gradient_checkpointing_enable", None)
        if not callable(enable_fn):
            return
        gc_kwargs = self._build_gradient_checkpointing_kwargs()
        try:
            if gc_kwargs is not None:
                enable_fn(gradient_checkpointing_kwargs=gc_kwargs)
            else:
                enable_fn()
            return
        except TypeError:
            pass
        try:
            if gc_kwargs is not None and "use_reentrant" in gc_kwargs:
                enable_fn(use_reentrant=gc_kwargs["use_reentrant"])
            else:
                enable_fn()
            return
        except TypeError:
            pass
        enable_fn()

    def _resolve_gradient_checkpointing_use_reentrant(self) -> Optional[bool]:
        raw_value = str(getattr(self.args, "gradient_checkpointing_use_reentrant", "auto")).lower().strip()
        if raw_value == "auto":
            # Quantized mixed-precision Qwen runs can hit checkpoint dtype-mismatch errors
            # under the non-reentrant engine; prefer the reentrant path in that regime.
            if self.quantization_mode != "none" and self.compute_dtype in {torch.float16, torch.bfloat16}:
                return True
            return False
        if raw_value == "true":
            return True
        if raw_value == "false":
            return False
        raise ValueError(f"Unsupported gradient_checkpointing_use_reentrant value: {raw_value}")

    def _build_gradient_checkpointing_kwargs(self) -> Optional[Dict[str, Any]]:
        if not getattr(self.args, "gradient_checkpointing", False):
            return None
        return {"use_reentrant": self._resolve_gradient_checkpointing_use_reentrant()}

    def _validate_quantization_configuration(self) -> None:
        if self.quantization_mode not in {"none", "8bit", "4bit"}:
            raise ValueError(f"Unsupported quantization mode: {self.quantization_mode}")
        if self.method_name == "vanilla_ft" and self.quantization_mode != "none":
            raise ValueError(
                "vanilla_ft trains original fusion-module parameters in the base model. "
                "4bit/8bit weight quantization is only supported for adapter/LoRA methods. "
                "Please rerun with --quantization none."
            )
        if self.quantization_mode != "none" and not self.is_adapter_method:
            raise ValueError(
                f"Method '{self.method_name}' does not support weight quantization. "
                "Only adapter/LoRA methods support --quantization 4bit/8bit in this repo."
            )
        if self.quantization_mode != "none" and self.device_obj.type != "cuda":
            raise ValueError("4bit/8bit weight quantization requires a CUDA device. Please use --device <gpu_id>.")

    def _get_quantized_device_map_target(self):
        if self.device_obj.type != "cuda":
            return str(self.device_obj)
        return f"cuda:{self.device_obj.index}"

    def _require_bitsandbytes(self) -> None:
        if bnb is None:
            raise ImportError(
                f"Quantization mode '{self.quantization_mode}' requires bitsandbytes to be installed. "
                "Install bitsandbytes and retry, or switch to --quantization none."
            )

    def _build_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        if self.quantization_mode == "none":
            return None
        self._require_bitsandbytes()
        if self.quantization_mode == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=float(getattr(self.args, "llm_int8_threshold", 6.0)),
            )
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(getattr(self.args, "bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(getattr(self.args, "bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=self.compute_dtype,
        )

    def _resolve_model_class(self):
        model_type = str(getattr(self, "model_type", "")).lower()

        if model_type == "qwen2_vl":
            if _Qwen2VLModelClass is not None:
                return _Qwen2VLModelClass, False
            if _AutoImageTextToText is not None:
                return _AutoImageTextToText, True
            if _AutoVision2Seq is not None:
                return _AutoVision2Seq, True
            raise ImportError("No compatible Transformers model loader found for Qwen2-VL.")

        if model_type == "qwen2_5_vl":
            if _Qwen25VLModelClass is not None:
                return _Qwen25VLModelClass, False
            if _AutoImageTextToText is not None:
                return _AutoImageTextToText, True
            if _AutoVision2Seq is not None:
                return _AutoVision2Seq, True
            raise ImportError("No compatible Transformers model loader found for Qwen2.5-VL.")

        # Fallback: use Auto classes for unknown checkpoint types.
        if _AutoImageTextToText is not None:
            return _AutoImageTextToText, True
        if _AutoVision2Seq is not None:
            return _AutoVision2Seq, True
        raise ImportError(f"No compatible Transformers model loader found for model_type={model_type!r}.")

    def _resolve_hidden_size(self) -> int:
        candidates = [
            getattr(self.config, "hidden_size", None),
            getattr(getattr(self.config, "text_config", None), "hidden_size", None),
            getattr(getattr(self.config, "llm_config", None), "hidden_size", None),
        ]
        for value in candidates:
            if value is not None:
                return int(value)

        to_dict = getattr(self.config, "to_dict", None)
        if callable(to_dict):
            config_dict = to_dict()
            for key in ("hidden_size",):
                value = config_dict.get(key)
                if value is not None:
                    return int(value)
            for nested_key in ("text_config", "llm_config"):
                nested = config_dict.get(nested_key)
                if isinstance(nested, dict) and nested.get("hidden_size") is not None:
                    return int(nested["hidden_size"])

        return 2048

    def _enable_gradient_checkpointing_inputs(self) -> None:
        if not getattr(self.args, "gradient_checkpointing", False):
            return

        enable_fn = getattr(self.model, "enable_input_require_grads", None)
        if callable(enable_fn):
            enable_fn()
            return

        if self._input_grad_hook_handle is not None:
            return

        get_embeddings = getattr(self.model, "get_input_embeddings", None)
        if not callable(get_embeddings):
            return

        try:
            embeddings = get_embeddings()
        except Exception:
            embeddings = None
        if embeddings is None:
            return

        def _make_outputs_require_grad(module, inputs, output):
            del module, inputs
            if torch.is_tensor(output):
                output.requires_grad_(True)
                return
            if isinstance(output, (list, tuple)):
                for item in output:
                    if torch.is_tensor(item):
                        item.requires_grad_(True)

        self._input_grad_hook_handle = embeddings.register_forward_hook(_make_outputs_require_grad)

    def _candidate_single_token(self, word: str) -> Optional[Tuple[str, int]]:
        tok = self.processor.tokenizer
        candidates = [word, f" {word}", word.lower(), f" {word.lower()}"]
        for cand in candidates:
            ids = tok.encode(cand, add_special_tokens=False)
            if len(ids) == 1:
                return cand, ids[0]
        return None

    def _default_label_words(self) -> List[str]:
        if self.dataset_name == "mmimdb":
            return self._parse_mmimdb_label_names()
        if self.dataset_name == "scienceqa":
            return ["A", "B", "C", "D", "E"]
        return ["Yes", "No"]

    def _parse_mmimdb_label_names(self) -> List[str]:
        value = getattr(self.args, "mmimdb_label_names", "")
        return [item.strip() for item in str(value).split(",") if item.strip()]

    def _parse_label_words(self) -> List[str]:
        words = [word.strip() for word in str(self.args.label_words).split(",") if word.strip()]
        if self.dataset_name == "scienceqa" and words == ["Yes", "No"]:
            return self._default_label_words()
        return words or self._default_label_words()

    def _resolve_binary_label_words(self) -> LabelWordInfo:
        pos = (self.args.positive_label_word or "").strip() or self._default_label_words()[0]
        neg = (self.args.negative_label_word or "").strip() or self._default_label_words()[1]
        pos_found = self._candidate_single_token(pos)
        neg_found = self._candidate_single_token(neg)
        if pos_found is not None and neg_found is not None:
            return LabelWordInfo(
                texts=(neg_found[0], pos_found[0]),
                token_ids=(neg_found[1], pos_found[1]),
                used_fallback=False,
                positive_text=pos_found[0],
                negative_text=neg_found[0],
                positive_token_id=pos_found[1],
                negative_token_id=neg_found[1],
            )

        pos_fb = self._candidate_single_token("1")
        neg_fb = self._candidate_single_token("0")
        if pos_fb is None or neg_fb is None:
            raise RuntimeError("Could not find single-token label words for Yes/No or 1/0 fallback.")
        return LabelWordInfo(
            texts=(neg_fb[0], pos_fb[0]),
            token_ids=(neg_fb[1], pos_fb[1]),
            used_fallback=True,
            positive_text=pos_fb[0],
            negative_text=neg_fb[0],
            positive_token_id=pos_fb[1],
            negative_token_id=neg_fb[1],
        )

    def _resolve_multiclass_label_words(self, words: List[str]) -> LabelWordInfo:
        texts: List[str] = []
        token_ids: List[int] = []
        used_fallback = False

        for word in words:
            found = self._candidate_single_token(word)
            if found is None:
                used_fallback = True
                break
            texts.append(found[0])
            token_ids.append(found[1])

        if used_fallback or not texts:
            texts = []
            token_ids = []
            for word in self._default_label_words():
                found = self._candidate_single_token(word)
                if found is None:
                    raise RuntimeError(f"Could not find a single-token verbalizer for label word: {word}")
                texts.append(found[0])
                token_ids.append(found[1])
            used_fallback = True

        return LabelWordInfo(texts=tuple(texts), token_ids=tuple(token_ids), used_fallback=used_fallback)

    def _resolve_label_words(self) -> LabelWordInfo:
        if self.dataset_name in {"hateful_memes", "nlvr2"}:
            return self._resolve_binary_label_words()
        if self.dataset_name == "scienceqa":
            return self._resolve_multiclass_label_words(self._parse_label_words())
        if self.dataset_name == "mmimdb":
            label_names = self._parse_mmimdb_label_names()
            if not label_names:
                raise ValueError("MMIMDb requires non-empty label names to build the task head.")
            return LabelWordInfo(texts=tuple(label_names), token_ids=tuple(), used_fallback=False)
        return self._resolve_binary_label_words()

    def describe_runtime(self) -> Dict[str, Any]:
        if self.dataset_name == "mmimdb":
            label_info = {
                "mode": "multilabel_task_head",
                "texts": list(self.label_info.texts),
                "token_ids": [],
                "used_fallback": self.label_info.used_fallback,
            }
        else:
            label_info = {
                "mode": "binary" if len(self.label_info.token_ids) == 2 else "multiclass",
                "texts": list(self.label_info.texts),
                "token_ids": list(self.label_info.token_ids),
                "used_fallback": self.label_info.used_fallback,
            }
        if self.dataset_name != "mmimdb" and len(self.label_info.token_ids) == 2:
            label_info.update(
                {
                    "positive_text": self.label_info.positive_text,
                    "negative_text": self.label_info.negative_text,
                    "positive_token_id": self.label_info.positive_token_id,
                    "negative_token_id": self.label_info.negative_token_id,
                }
            )

        return {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "model_type": self.model_type,
            "image_token_id": self.image_token_id,
            "label_info": label_info,
            "quantization": self.quantization_mode,
            "is_quantized_model": self.is_quantized_model,
            "fusion_linear_module_names": self.fusion_linear_module_names,
            "parameter_counts": count_parameters(self),
            "trainable_report": self.describe_trainable_state(preview_limit=self.preview_limit),
        }

    def describe_trainable_state(self, preview_limit: Optional[int] = None) -> Dict[str, Any]:
        return collect_trainable_parameter_report(
            self,
            expected_modules=self.lora_target_module_names if self.is_adapter_method else self.fusion_linear_module_names,
            preview_limit=int(preview_limit or self.preview_limit),
            lm_layer_boundary=None if ("matched_lora" in str(getattr(self.scope_config, "scope_name", "")) or "branch_lora" in str(getattr(self.scope_config, "scope_name", ""))) else int(getattr(self.args, "num_fusion_layers", 4)),
        )

    def apply_method_setup(self, method_name: str) -> None:
        method_name = method_name.lower()
        if method_name in self.adapter_methods:
            self._setup_lora()
        elif method_name == "vanilla_ft":
            self._setup_vanilla_ft()
        else:
            raise ValueError(f"Unsupported method: {method_name}")
        self._enable_gradient_checkpointing_inputs()
        if self.is_quantized_model:
            if self._task_head is not None:
                self._task_head.to(self.device_obj)
        else:
            self.to(self.device_obj)

    def _setup_vanilla_ft(self) -> None:
        self._validate_quantization_configuration()
        freeze_all_parameters(self.model)
        unfreeze_modules_by_prefix(self.model, self.scope_config.unfreeze_prefixes)
        if self._task_head is not None:
            for p in self._task_head.parameters():
                p.requires_grad = True

    def _patch_bnb_8bit_state_for_peft(self) -> None:
        if self.quantization_mode != "8bit" or bnb is None:
            return
        linear8bit_cls = getattr(getattr(bnb, "nn", None), "Linear8bitLt", None)
        if linear8bit_cls is None:
            return

        patched_states = 0
        patched_indices = 0
        for module in self.model.modules():
            if not isinstance(module, linear8bit_cls):
                continue
            state = getattr(module, "state", None)
            if state is not None and not hasattr(state, "memory_efficient_backward"):
                setattr(state, "memory_efficient_backward", False)
                patched_states += 1
            if not hasattr(module, "index"):
                setattr(module, "index", None)
                patched_indices += 1

        if patched_states or patched_indices:
            warnings.warn(
                "Applied an 8bit bitsandbytes/PEFT compatibility patch before LoRA injection. "
                "This runtime is missing attributes expected by PEFT on Linear8bitLt modules. "
                "Consider using a tested package combination to avoid relying on this shim.",
                RuntimeWarning,
            )

    def _setup_lora(self) -> None:
        if self.is_quantized_model:
            use_gc = bool(getattr(self.args, "gradient_checkpointing", False))
            gc_kwargs = self._build_gradient_checkpointing_kwargs() if use_gc else None
            try:
                prepare_kwargs = {"use_gradient_checkpointing": use_gc}
                if gc_kwargs is not None:
                    prepare_kwargs["gradient_checkpointing_kwargs"] = gc_kwargs
                self.model = prepare_model_for_kbit_training(self.model, **prepare_kwargs)
            except TypeError:
                try:
                    self.model = prepare_model_for_kbit_training(
                        self.model,
                        use_gradient_checkpointing=use_gc,
                    )
                except TypeError:
                    self.model = prepare_model_for_kbit_training(self.model)
            self._enable_model_gradient_checkpointing()
            self._patch_bnb_8bit_state_for_peft()
        freeze_all_parameters(self.model)
        lora_cfg = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            bias=self.args.bias,
            target_modules=list(self.lora_target_module_names),
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_cfg)
        if self._task_head is not None:
            for p in self._task_head.parameters():
                p.requires_grad = True

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def get_trainable_named_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        return [(n, p) for n, p in self.named_parameters() if p.requires_grad]

    @staticmethod
    def _parameter_hits_module_scope(normalized_name: str, module_names: List[str]) -> bool:
        for prefix in module_names:
            if normalized_name == prefix or normalized_name.startswith(prefix + "."):
                return True
            if normalized_name.startswith(prefix + ".lora_"):
                return True
        return False

    def _scope_trainable_named_parameters(
        self,
        module_names: List[str],
        *,
        include_task_head: bool = False,
    ) -> List[Tuple[str, nn.Parameter]]:
        named: List[Tuple[str, nn.Parameter]] = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            normalized = normalize_scope_name(name)
            if include_task_head and normalized.startswith("task_head"):
                named.append((name, param))
                continue
            if self._parameter_hits_module_scope(normalized, module_names):
                named.append((name, param))
        return named

    def get_branch_lora_named_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        return self._scope_trainable_named_parameters(self.branch_linear_module_names, include_task_head=False)

    def get_branch_lora_grouped_named_parameters(
        self,
    ) -> Tuple[List[Tuple[str, nn.Parameter]], List[Tuple[str, nn.Parameter]]]:
        visual = self._scope_trainable_named_parameters(self.branch_visual_module_names, include_task_head=False)
        text = self._scope_trainable_named_parameters(self.branch_text_module_names, include_task_head=False)
        return visual, text

    def get_fusion_trainable_named_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        # FCF perturbs the fusion path theta_f only.  The task head theta_h remains
        # trainable through the normal task loss, but is excluded from fusion-side
        # FCF/FAST-FCF perturbation and projection so the VLM PEFT implementation
        # matches the controlled late-fusion definition.
        return self._scope_trainable_named_parameters(self.fusion_linear_module_names, include_task_head=False)

    def get_autocast_context(self):
        if self.device_obj.type != "cuda":
            return contextlib.nullcontext()
        if self.compute_dtype == torch.float32:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.compute_dtype)

    def _render_prompt(self, messages: List[Dict[str, Any]], add_vision_id: bool = False) -> str:
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_vision_id=add_vision_id,
        )

    def _encode_single_prompt(
        self,
        prompt_text: str,
        images: Optional[List[Any]] = None,
        truncation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        image_count = len(images) if images is not None else 0
        effective_truncation = bool(truncation) and image_count <= 1
        kwargs = {
            "text": [prompt_text],
            "padding": False,
            "truncation": effective_truncation,
            "return_tensors": "pt",
        }
        if effective_truncation:
            kwargs["max_length"] = max(8, self.args.max_seq_length - 1)
        if images:
            kwargs["images"] = images
        return self.processor(**kwargs)

    def _build_single_token_full_inputs(self, prompt_inputs: Dict[str, torch.Tensor], label_token_id: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        label_tensor = torch.tensor([[label_token_id]], dtype=prompt_inputs["input_ids"].dtype)
        full_inputs = {}
        for key, value in prompt_inputs.items():
            if key in {"input_ids", "attention_mask", "token_type_ids"}:
                continue
            full_inputs[key] = value

        full_inputs["input_ids"] = torch.cat([prompt_inputs["input_ids"], label_tensor], dim=1)
        full_inputs["attention_mask"] = torch.cat(
            [prompt_inputs["attention_mask"], torch.ones((1, 1), dtype=prompt_inputs["attention_mask"].dtype)],
            dim=1,
        )
        if "token_type_ids" in prompt_inputs:
            full_inputs["token_type_ids"] = torch.cat(
                [prompt_inputs["token_type_ids"], torch.zeros((1, 1), dtype=prompt_inputs["token_type_ids"].dtype)],
                dim=1,
            )

        lm_labels = torch.full_like(full_inputs["input_ids"], fill_value=-100)
        lm_labels[:, -1] = label_token_id
        return full_inputs, lm_labels

    def _merge_feature_list(self, feature_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        merged: Dict[str, torch.Tensor] = {}
        keys = set()
        for features in feature_list:
            keys.update(features.keys())

        for key in keys:
            values = [features[key] for features in feature_list if key in features]
            if not values:
                continue
            if key == "input_ids":
                merged[key] = pad_sequence(
                    [value.squeeze(0) for value in values],
                    batch_first=True,
                    padding_value=self.processor.tokenizer.pad_token_id,
                )
                continue
            if key in {"attention_mask", "token_type_ids"}:
                merged[key] = pad_sequence(
                    [value.squeeze(0) for value in values],
                    batch_first=True,
                    padding_value=0,
                )
                continue
            merged[key] = torch.cat(values, dim=0)
        return merged

    def _prepare_single_token_batch(
        self,
        sample_specs: List[Dict[str, Any]],
        split: str,
        dataset_name: str,
        extra_tensor_fields: Optional[Dict[str, List[int]]] = None,
    ) -> Dict[str, Any]:
        prompt_inputs_list: List[Dict[str, torch.Tensor]] = []
        full_inputs_list: List[Dict[str, torch.Tensor]] = []
        lm_label_list: List[torch.Tensor] = []
        sample_ids: List[Any] = []
        labels: List[int] = []

        for spec in sample_specs:
            prompt_inputs = self._encode_single_prompt(spec["prompt_text"], images=spec.get("images"))
            full_inputs, lm_labels = self._build_single_token_full_inputs(prompt_inputs, spec["label_token_id"])
            prompt_inputs_list.append(prompt_inputs)
            full_inputs_list.append(full_inputs)
            lm_label_list.append(lm_labels.squeeze(0))
            sample_ids.append(spec["id"])
            labels.append(int(spec["label"]))

        batch = {
            "sample_ids": sample_ids,
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_inputs": self._merge_feature_list(prompt_inputs_list),
            "full_inputs": self._merge_feature_list(full_inputs_list),
            "lm_labels": pad_sequence(lm_label_list, batch_first=True, padding_value=-100),
            "split": split,
            "dataset_name": dataset_name,
        }

        if extra_tensor_fields:
            for key, values in extra_tensor_fields.items():
                batch[key] = torch.tensor(values, dtype=torch.long)
        return batch

    def _prepare_task_head_batch(
        self,
        sample_specs: List[Dict[str, Any]],
        split: str,
        dataset_name: str,
    ) -> Dict[str, Any]:
        prompt_inputs_list: List[Dict[str, torch.Tensor]] = []
        sample_ids: List[Any] = []
        labels: List[List[float]] = []

        for spec in sample_specs:
            prompt_inputs = self._encode_single_prompt(spec["prompt_text"], images=spec.get("images"))
            prompt_inputs_list.append(prompt_inputs)
            sample_ids.append(spec["id"])
            labels.append([float(x) for x in spec["labels"]])

        prompt_inputs = self._merge_feature_list(prompt_inputs_list)
        return {
            "sample_ids": sample_ids,
            "labels": torch.tensor(labels, dtype=torch.float32),
            "prompt_inputs": prompt_inputs,
            "full_inputs": dict(prompt_inputs),
            "lm_labels": None,
            "split": split,
            "dataset_name": dataset_name,
        }

    def _binary_label_token_id(self, label: int) -> int:
        return self.label_info.positive_token_id if int(label) == 1 else self.label_info.negative_token_id

    def _scienceqa_option_labels(self, num_choices: int) -> List[str]:
        if num_choices > len(self.label_info.texts):
            raise ValueError(
                f"ScienceQA sample requires {num_choices} options, but only {len(self.label_info.texts)} label words are configured."
            )
        return list(self.label_info.texts[:num_choices])

    def prepare_batch_by_dataset(self, dataset: str, **kwargs) -> Dict[str, Any]:
        dataset = str(dataset).lower()
        if dataset == "hateful_memes":
            return self.prepare_hateful_memes_batch(**kwargs)
        if dataset == "nlvr2":
            return self.prepare_nlvr2_batch(**kwargs)
        if dataset == "scienceqa":
            return self.prepare_scienceqa_batch(**kwargs)
        if dataset == "mmimdb":
            return self.prepare_mmimdb_batch(**kwargs)
        raise ValueError(f"Unsupported dataset: {dataset}")

    def prepare_hateful_memes_batch(
        self,
        sample_ids: List[Any],
        texts: List[str],
        labels: List[int],
        images: List[Any],
        split: str,
    ) -> Dict[str, Any]:
        sample_specs = []
        for sample_id, text, label, image in zip(sample_ids, texts, labels, images):
            prompt_text = hateful_memes_yes_no_prompt(text)
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
            sample_specs.append(
                {
                    "id": sample_id,
                    "label": int(label),
                    "label_token_id": self._binary_label_token_id(int(label)),
                    "prompt_text": self._render_prompt(messages),
                    "images": [image],
                }
            )
        return self._prepare_single_token_batch(sample_specs, split=split, dataset_name="hateful_memes")

    def prepare_nlvr2_batch(
        self,
        sample_ids: List[Any],
        statements: List[str],
        labels: List[int],
        image_pairs: List[Tuple[Any, Any]],
        split: str,
    ) -> Dict[str, Any]:
        sample_specs = []
        for sample_id, statement, label, image_pair in zip(sample_ids, statements, labels, image_pairs):
            left_image, right_image = image_pair
            prompt_text = nlvr2_yes_no_prompt(statement)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            sample_specs.append(
                {
                    "id": sample_id,
                    "label": int(label),
                    "label_token_id": self._binary_label_token_id(int(label)),
                    "prompt_text": self._render_prompt(messages, add_vision_id=True),
                    "images": [left_image, right_image],
                }
            )
        return self._prepare_single_token_batch(sample_specs, split=split, dataset_name="nlvr2")

    def prepare_scienceqa_batch(
        self,
        sample_ids: List[Any],
        questions: List[str],
        choices: List[List[str]],
        labels: List[int],
        hints: List[str],
        images: List[Any],
        split: str,
    ) -> Dict[str, Any]:
        sample_specs = []
        num_choices: List[int] = []

        for sample_id, question, option_list, label, hint, image in zip(sample_ids, questions, choices, labels, hints, images):
            option_labels = self._scienceqa_option_labels(len(option_list))
            prompt_text = scienceqa_multichoice_prompt(question=question, choices=option_list, option_labels=option_labels, hint=hint)
            content: List[Dict[str, Any]] = []
            image_inputs: Optional[List[Any]] = None
            if image is not None:
                content.append({"type": "image"})
                image_inputs = [image]
            content.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content}]
            sample_specs.append(
                {
                    "id": sample_id,
                    "label": int(label),
                    "label_token_id": int(self.label_info.token_ids[int(label)]),
                    "prompt_text": self._render_prompt(messages),
                    "images": image_inputs,
                }
            )
            num_choices.append(len(option_list))

        return self._prepare_single_token_batch(
            sample_specs,
            split=split,
            dataset_name="scienceqa",
            extra_tensor_fields={"num_choices": num_choices},
        )

    def prepare_mmimdb_batch(
        self,
        sample_ids: List[Any],
        titles: List[str],
        plots: List[str],
        labels: List[List[int]],
        images: List[Any],
        split: str,
    ) -> Dict[str, Any]:
        sample_specs = []
        label_names = list(self.label_info.texts)

        for sample_id, title, plot, sample_labels, image in zip(sample_ids, titles, plots, labels, images):
            prompt_text = mmimdb_multilabel_prompt(title=title, plot=plot, label_names=label_names)
            content: List[Dict[str, Any]] = []
            image_inputs: Optional[List[Any]] = None
            if image is not None:
                content.append({"type": "image"})
                image_inputs = [image]
            content.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content}]
            sample_specs.append(
                {
                    "id": sample_id,
                    "labels": list(sample_labels),
                    "prompt_text": self._render_prompt(messages),
                    "images": image_inputs,
                }
            )

        return self._prepare_task_head_batch(sample_specs, split=split, dataset_name="mmimdb")

    def _get_hook_target(self):
        candidates = [self.model]
        seen = set()
        while candidates:
            obj = candidates.pop(0)
            if obj is None:
                continue
            oid = id(obj)
            if oid in seen:
                continue
            seen.add(oid)
            inner = getattr(obj, "model", None)
            if inner is not None and hasattr(inner, "layers"):
                return inner
            candidates.append(getattr(obj, "model", None))
            candidates.append(getattr(obj, "base_model", None))
        raise RuntimeError("Could not locate the inner decoder model for registering hooks.")

    def _register_layer0_hook(self, capture: bool = False, perturb: Optional[torch.Tensor] = None):
        cache: Dict[str, Any] = {}

        def pre_hook(module, inputs):
            del module
            hidden = inputs[0]
            if capture:
                if not hidden.requires_grad:
                    hidden = hidden.detach().requires_grad_(True)
                else:
                    hidden.requires_grad_(True)
                hidden.retain_grad()
                cache["hidden"] = hidden
            if perturb is not None:
                hidden = hidden + perturb
                return (hidden, *inputs[1:])
            if capture:
                return (hidden, *inputs[1:])
            return inputs

        hook_target = self._get_hook_target()
        handle = hook_target.layers[0].register_forward_pre_hook(pre_hook)
        return handle, cache

    def _apply_param_perturb(self, perturb_dict: Dict[str, torch.Tensor], sign: int) -> None:
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in perturb_dict:
                    param.add_(perturb_dict[name] * sign)

    def _move_any_to_device(self, value, device):
        if torch.is_tensor(value):
            return value.to(device, non_blocking=True)

        if hasattr(value, "to") and callable(value.to):
            try:
                return value.to(device, non_blocking=True)
            except TypeError:
                try:
                    return value.to(device)
                except Exception:
                    pass
            except Exception:
                pass

        if isinstance(value, Mapping):
            return {k: self._move_any_to_device(v, device) for k, v in value.items()}
        if isinstance(value, list):
            return [self._move_any_to_device(v, device) for v in value]
        if isinstance(value, tuple):
            return tuple(self._move_any_to_device(v, device) for v in value)
        return value

    def move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {k: self._move_any_to_device(v, self.device_obj) for k, v in batch.items()}

    def _get_model_input_device(self):
        try:
            return self.model.get_input_embeddings().weight.device
        except Exception:
            return next(self.model.parameters()).device

    def _call_model(self, inputs: Dict[str, torch.Tensor], labels: Optional[torch.Tensor] = None, **extra_kwargs):
        model_device = self._get_model_input_device()
        kwargs = self._move_any_to_device(dict(inputs), model_device)
        if labels is not None:
            kwargs["labels"] = labels.to(model_device, non_blocking=True)
        kwargs.update(extra_kwargs)
        return self.model(**kwargs, return_dict=True)

    def forward_lm(
        self,
        batch: Dict[str, Any],
        use_full_inputs: bool = True,
        labels: Optional[torch.Tensor] = None,
        capture_layer0: bool = False,
        hidden_perturb: Optional[torch.Tensor] = None,
        param_perturb: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        inputs = batch["full_inputs"] if use_full_inputs else batch["prompt_inputs"]
        hook, cache = self._register_layer0_hook(capture=capture_layer0, perturb=hidden_perturb)
        if param_perturb:
            self._apply_param_perturb(param_perturb, sign=1)
        try:
            with self.get_autocast_context():
                outputs = self._call_model(inputs, labels=labels, use_cache=False)
        finally:
            hook.remove()
            if param_perturb:
                self._apply_param_perturb(param_perturb, sign=-1)
        return outputs, cache

    @staticmethod
    def _gather_last_token(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_idx = attention_mask.sum(dim=1) - 1
        return hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), last_idx]

    def forward_task_head(
        self,
        batch: Dict[str, Any],
        use_full_inputs: bool = True,
        compute_loss: bool = True,
        capture_layer0: bool = False,
        hidden_perturb: Optional[torch.Tensor] = None,
        param_perturb: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[TaskHeadForwardOutput, Dict[str, Any]]:
        if self._task_head is None:
            raise RuntimeError("Task-head forward requested, but task_head_cls is not initialized.")

        inputs = batch["full_inputs"] if use_full_inputs else batch["prompt_inputs"]
        labels = batch["labels"] if compute_loss else None
        hook, cache = self._register_layer0_hook(capture=capture_layer0, perturb=hidden_perturb)
        if param_perturb:
            self._apply_param_perturb(param_perturb, sign=1)
        try:
            with self.get_autocast_context():
                outputs = self._call_model(inputs, labels=None, output_hidden_states=True, use_cache=False)
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states is None:
                    raise RuntimeError("The base model did not return hidden states for task-head classification.")
                attention_mask = inputs["attention_mask"].to(hidden_states[-1].device, non_blocking=True)
                pooled = self._gather_last_token(hidden_states[-1], attention_mask)
                logits = self._task_head(pooled)
                loss = None
                if labels is not None:
                    loss = F.binary_cross_entropy_with_logits(logits.float(), labels.to(logits.device, dtype=torch.float32))
        finally:
            hook.remove()
            if param_perturb:
                self._apply_param_perturb(param_perturb, sign=-1)
        return TaskHeadForwardOutput(loss=loss, logits=logits), cache

    def compute_base_loss(self, batch: Dict[str, Any], capture_layer0: bool = False):
        if batch["dataset_name"] == "mmimdb":
            return self.forward_task_head(batch, use_full_inputs=True, compute_loss=True, capture_layer0=capture_layer0)
        return self.forward_lm(batch, use_full_inputs=True, labels=batch["lm_labels"], capture_layer0=capture_layer0)

    def compute_prompt_scores(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if batch["dataset_name"] == "mmimdb":
            outputs, _ = self.forward_task_head(batch, use_full_inputs=False, compute_loss=False, capture_layer0=False)
            probs = torch.sigmoid(outputs.logits.float())
            preds = (probs >= float(getattr(self.args, "mmimdb_threshold", 0.5))).long()
            return {
                "probs": probs,
                "scores": probs,
                "preds": preds,
            }

        outputs, _ = self.forward_lm(batch, use_full_inputs=False, labels=None, capture_layer0=False)
        logits = outputs.logits
        attention_mask = batch["prompt_inputs"]["attention_mask"].to(logits.device, non_blocking=True)
        gathered = self._gather_last_token(logits, attention_mask)

        if batch["dataset_name"] in {"hateful_memes", "nlvr2"}:
            candidate_ids = [self.label_info.negative_token_id, self.label_info.positive_token_id]
            pair_logits = gathered[:, candidate_ids]
            pair_probs = torch.softmax(pair_logits.float(), dim=-1)
            probs = pair_probs[:, 1]
            preds = pair_probs.argmax(dim=-1)
            return {
                "probs": probs,
                "scores": pair_probs,
                "preds": preds,
            }

        if batch["dataset_name"] == "scienceqa":
            class_logits = gathered[:, list(self.label_info.token_ids)]
            num_choices = batch["num_choices"].to(class_logits.device, non_blocking=True)
            choice_mask = torch.arange(class_logits.size(1), device=class_logits.device).unsqueeze(0) < num_choices.unsqueeze(1)
            masked_logits = class_logits.masked_fill(~choice_mask, float("-inf"))
            probs = torch.softmax(masked_logits.float(), dim=-1)
            preds = probs.argmax(dim=-1)
            return {
                "probs": probs,
                "scores": probs,
                "preds": preds,
            }

        raise NotImplementedError(f"Prompt scoring is not implemented for dataset: {batch['dataset_name']}")

    def _build_special_token_mask(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if not self.text_proxy_excluded_token_ids:
            return torch.zeros_like(attention_mask, dtype=torch.bool)

        special_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        for token_id in self.text_proxy_excluded_token_ids:
            special_mask |= input_ids == token_id
        return special_mask & attention_mask

    @staticmethod
    def _expand_token_mask(token_mask: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return token_mask.unsqueeze(-1).expand_as(hidden)

    def _build_branch_masks(self, batch: Dict[str, Any], hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = hidden.device
        input_ids = batch["full_inputs"]["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["full_inputs"]["attention_mask"].to(device, non_blocking=True).bool()
        image_token_mask = (input_ids == self.image_token_id) & attention_mask
        special_token_mask = self._build_special_token_mask(input_ids, attention_mask)

        supervision_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        if batch.get("lm_labels") is not None:
            supervision_mask = batch["lm_labels"].to(device, non_blocking=True).ne(-100) & attention_mask

        # Layer-0 decoder states already contain mixed-modal information, so these masks are
        # proxy regions for the visual/text branches rather than the idealized z_v / z_t from the paper.
        text_token_mask = attention_mask & (~image_token_mask) & (~special_token_mask) & (~supervision_mask)
        if not bool(text_token_mask.any()):
            self._warn_geometry_once(
                "empty_text_proxy",
                "Text branch proxy mask is empty after excluding image/special/supervision tokens; text-side geometry will default to zero for this batch.",
            )

        return self._expand_token_mask(image_token_mask, hidden), self._expand_token_mask(text_token_mask, hidden)

    @staticmethod
    def masked_frobenius_norm(
        tensor: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        tensor_fp32 = tensor.float()
        if mask is None:
            return tensor_fp32.pow(2).sum().sqrt()

        mask_bool = mask.bool()
        token_mask = mask_bool.any(dim=-1) if mask_bool.dim() == tensor_fp32.dim() else mask_bool
        token_sq = tensor_fp32.pow(2).sum(dim=-1)
        token_mask_fp32 = token_mask.to(dtype=tensor_fp32.dtype)
        denom = token_mask_fp32.sum()
        if float(denom.detach().cpu().item()) <= 0.0:
            return tensor_fp32.new_tensor(0.0)
        mean_sq = (token_sq * token_mask_fp32).sum() / denom
        return torch.sqrt(mean_sq.clamp_min(0.0) + eps)

    def build_hidden_perturb(self, hidden_grad: torch.Tensor, mask: torch.Tensor, radius: float, eps: float = 1e-12) -> torch.Tensor:
        if radius <= 0.0:
            return torch.zeros_like(hidden_grad)

        masked_grad = hidden_grad * mask
        norm = self.masked_frobenius_norm(hidden_grad, mask=mask, eps=eps)
        if float(norm.detach().cpu().item()) <= eps:
            return torch.zeros_like(hidden_grad)
        scale = radius / (norm + eps)
        return (masked_grad.float() * scale).to(dtype=hidden_grad.dtype)

    def build_param_perturb(self, named_params: List[Tuple[str, nn.Parameter]], radius: float, eps: float = 1e-12) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        sq = 0.0
        grads: Dict[str, torch.Tensor] = {}
        for name, p in named_params:
            if p.grad is None:
                continue
            g = p.grad.detach().float()
            grads[name] = g
            sq += g.pow(2).sum().item()
        norm = math.sqrt(sq)
        if norm == 0.0:
            return {}, torch.tensor(0.0, device=self.device_obj)
        scale = radius / (norm + eps)
        perturb = {
            name: grad.to(dtype=p.dtype, device=p.device) * scale
            for (name, p), grad in ((item, grads[item[0]]) for item in named_params if item[0] in grads)
        }
        return perturb, torch.tensor(norm, device=self.device_obj)

    def compute_geometry_probe(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs, cache = self.compute_base_loss(batch, capture_layer0=True)
        probe_loss = outputs.loss
        hidden = cache.get("hidden")
        if hidden is None:
            raise RuntimeError("Failed to capture layer-0 hidden states for geometry computation.")
        eps = self._geometry_eps()

        fusion_named = self.get_fusion_trainable_named_parameters()
        grad_inputs = [hidden] + [p for _, p in fusion_named]
        grads = torch.autograd.grad(
            outputs=probe_loss,
            inputs=grad_inputs,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        hidden_grad = grads[0]
        if hidden_grad is None:
            raise RuntimeError("Failed to compute gradients w.r.t. captured layer-0 hidden states.")

        image_mask, text_mask = self._build_branch_masks(batch, hidden)
        hidden_grad = hidden_grad.detach()
        gv_norm = self.masked_frobenius_norm(hidden_grad, mask=image_mask, eps=eps)
        gt_norm = self.masked_frobenius_norm(hidden_grad, mask=text_mask, eps=eps)

        delta_v = self.build_hidden_perturb(hidden_grad, image_mask, self.args.rho_v, eps=eps)
        delta_t = self.build_hidden_perturb(hidden_grad, text_mask, self.args.rho_t, eps=eps)

        sq = 0.0
        param_perturb = {}
        grad_map = {}
        for (name, param), grad in zip(fusion_named, grads[1:]):
            if grad is None:
                continue
            g = grad.detach().float()
            grad_map[name] = g
            sq += g.pow(2).sum().item()
        gf_norm_value = math.sqrt(sq)
        if gf_norm_value > 0.0:
            scale = self.args.rho_f / (gf_norm_value + eps)
            for name, param in fusion_named:
                if name in grad_map:
                    param_perturb[name] = grad_map[name].to(dtype=param.dtype, device=param.device) * scale
        gf_norm = torch.tensor(gf_norm_value, device=probe_loss.device, dtype=torch.float32)

        return {
            "probe_loss": probe_loss.detach(),
            "delta_v": delta_v.detach(),
            "delta_t": delta_t.detach(),
            "param_perturb": {k: v.detach() for k, v in param_perturb.items()},
            "gv_norm": gv_norm.detach(),
            "gt_norm": gt_norm.detach(),
            "gf_norm": gf_norm.detach(),
        }

    def compute_linear_geometry_stats(self, batch: Dict[str, torch.Tensor], retain_graph: bool = False) -> Dict[str, torch.Tensor]:
        _ = retain_graph
        probe = self.compute_geometry_probe(batch)
        branch_linear_proxy = self._aggregate_branch_linear_proxy(
            probe["gv_norm"].float(),
            probe["gt_norm"].float(),
        )
        return {
            "loss": probe["probe_loss"],
            "gv_norm": probe["gv_norm"],
            "gt_norm": probe["gt_norm"],
            "gf_norm": probe["gf_norm"],
            "branch_linear_proxy": branch_linear_proxy,
        }


    @staticmethod
    def _differentiable_grad_norm(
        grads: List[Optional[torch.Tensor]],
        *,
        device: torch.device,
        dtype: torch.dtype,
        eps: float,
    ) -> torch.Tensor:
        """Detached L2 norm used by legacy FAST-FCF diagnostics."""
        sq = None
        for grad in grads:
            if grad is None:
                continue
            term = grad.float().pow(2).sum()
            sq = term if sq is None else sq + term.to(device=sq.device)
        if sq is None:
            return torch.zeros((), device=device, dtype=dtype)
        return torch.sqrt(sq.to(device=device) + float(eps)).to(device=device, dtype=dtype)

    def compute_fast_train_geometry(self, batch: Dict[str, torch.Tensor], retain_graph: bool = True) -> Dict[str, torch.Tensor]:
        """Detached FAST-FCF diagnostics for COFLA-F.

        FAST-FCF is a first-order diagnostic/gate.  This legacy helper keeps
        gradient-norm quantities detached and never builds second-order graphs.
        The current FastCOFLAMethod performs the one-probe finite-step fusion correction itself.
        """
        outputs, cache = self.compute_base_loss(batch, capture_layer0=True)
        loss0 = outputs.loss.float()
        hidden = cache.get("hidden")
        if hidden is None:
            raise RuntimeError("Failed to capture layer-0 hidden states for FAST-FCF training geometry.")

        eps = self._geometry_eps()
        device = loss0.device
        dtype = loss0.dtype
        fusion_named = self.get_fusion_trainable_named_parameters()
        grad_inputs = [hidden] + [param for _, param in fusion_named]
        grads = torch.autograd.grad(
            outputs=loss0,
            inputs=grad_inputs,
            retain_graph=retain_graph,
            create_graph=False,
            allow_unused=True,
        )
        hidden_grad = grads[0]
        if hidden_grad is None:
            hidden_grad = torch.zeros_like(hidden)

        image_mask, text_mask = self._build_branch_masks(batch, hidden)
        gv_norm = self.masked_frobenius_norm(hidden_grad, mask=image_mask, eps=eps)
        gt_norm = self.masked_frobenius_norm(hidden_grad, mask=text_mask, eps=eps)
        gf_norm = self._differentiable_grad_norm(
            list(grads[1:]),
            device=device,
            dtype=dtype,
            eps=eps,
        )

        fast_s_v = (float(self.args.rho_v) * gv_norm).detach()
        fast_s_t = (float(self.args.rho_t) * gt_norm).detach()
        fast_s_branch = self._aggregate_branch_sharpness(fast_s_v, fast_s_t).detach()
        fast_s_f = (float(self.args.rho_f) * gf_norm).detach()
        eps_tensor = torch.tensor(float(eps), device=device, dtype=dtype)
        fast_fcf = ((fast_s_f + eps_tensor) / (fast_s_branch + eps_tensor)).detach()
        fast_rfcf = torch.log(fast_fcf.clamp_min(eps)).detach()

        return {
            "loss": loss0,
            "loss_v_plus": loss0 + fast_s_v,
            "loss_t_plus": loss0 + fast_s_t,
            "loss_f_plus": loss0 + fast_s_f,
            "fast_s_v": fast_s_v,
            "fast_s_t": fast_s_t,
            "fast_s_branch": fast_s_branch,
            "fast_s_f": fast_s_f,
            "fast_fcf": fast_fcf,
            "fast_rfcf": fast_rfcf,
            # Aliases keep the COFLA objective unchanged while swapping exact
            # FCF/RFCF for their FAST-FCF/FAST-RFCF surrogates.
            "s_v": fast_s_v,
            "s_t": fast_s_t,
            "s_branch": fast_s_branch,
            "s_f": fast_s_f,
            "s_v_b": fast_s_v,
            "s_t_b": fast_s_t,
            "s_branch_b": fast_s_branch,
            "s_f_b": fast_s_f,
            "fcf": fast_fcf,
            "rfcf": fast_rfcf,
            "rlin_fcf": fast_rfcf,
            "gv_norm": gv_norm,
            "gt_norm": gt_norm,
            "gf_norm": gf_norm,
            "branch_linear_proxy": fast_s_branch,
        }

    def compute_geometry_metrics_from_probe(self, batch: Dict[str, torch.Tensor], probe: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if batch["dataset_name"] == "mmimdb":
            out0, _ = self.forward_task_head(batch, use_full_inputs=True, compute_loss=True, capture_layer0=False)
            l0 = out0.loss
            outv, _ = self.forward_task_head(
                batch,
                use_full_inputs=True,
                compute_loss=True,
                capture_layer0=False,
                hidden_perturb=probe["delta_v"],
            )
            outt, _ = self.forward_task_head(
                batch,
                use_full_inputs=True,
                compute_loss=True,
                capture_layer0=False,
                hidden_perturb=probe["delta_t"],
            )
            outf, _ = self.forward_task_head(
                batch,
                use_full_inputs=True,
                compute_loss=True,
                capture_layer0=False,
                param_perturb=probe["param_perturb"],
            )
        else:
            out0, _ = self.forward_lm(batch, use_full_inputs=True, labels=batch["lm_labels"], capture_layer0=False)
            l0 = out0.loss
            outv, _ = self.forward_lm(
                batch,
                use_full_inputs=True,
                labels=batch["lm_labels"],
                capture_layer0=False,
                hidden_perturb=probe["delta_v"],
            )
            outt, _ = self.forward_lm(
                batch,
                use_full_inputs=True,
                labels=batch["lm_labels"],
                capture_layer0=False,
                hidden_perturb=probe["delta_t"],
            )
            outf, _ = self.forward_lm(
                batch,
                use_full_inputs=True,
                labels=batch["lm_labels"],
                capture_layer0=False,
                param_perturb=probe["param_perturb"],
            )

        eps = self._geometry_eps()
        l0 = l0.float()
        s_v = self._stabilize_non_negative_sharpness(outv.loss.float() - l0, "s_v_b")
        s_t = self._stabilize_non_negative_sharpness(outt.loss.float() - l0, "s_t_b")
        s_f = self._stabilize_non_negative_sharpness(outf.loss.float() - l0, "s_f_b")
        s_branch = self._aggregate_branch_sharpness(s_v, s_t)
        fcf = (s_f + eps) / (s_branch + eps)
        # RFCF is defined as log(FCF); using abs() would hide invalid negative ratios instead of fixing them.
        rfcf = torch.log(fcf.clamp_min(eps))
        branch_linear_proxy = self._aggregate_branch_linear_proxy(probe["gv_norm"].float(), probe["gt_norm"].float())
        rlin = torch.log(
            (self.args.rho_f * probe["gf_norm"].float() + eps)
            / (branch_linear_proxy + eps)
        )
        return {
            "loss": l0,
            "loss_v_plus": l0 + s_v,
            "loss_t_plus": l0 + s_t,
            "loss_f_plus": l0 + s_f,
            "s_v_b": s_v,
            "s_t_b": s_t,
            "s_f_b": s_f,
            "s_branch_b": s_branch,
            # Aliases used by the COFLA training code, matching the controlled
            # late-fusion implementation.
            "s_v": s_v,
            "s_t": s_t,
            "s_f": s_f,
            "s_branch": s_branch,
            "fcf": fcf,
            "rfcf": rfcf,
            "rlin_fcf": rlin,
            "gv_norm": probe["gv_norm"],
            "gt_norm": probe["gt_norm"],
            "gf_norm": probe["gf_norm"],
        }

    def export_trainable_state(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        trainable_keys = {name for name, param in self.named_parameters() if param.requires_grad}
        state_dict = self.state_dict()
        filtered = {k: v.detach().cpu() for k, v in state_dict.items() if k in trainable_keys or k.startswith("task_head")}
        return {
            "model_state_dict": filtered,
            "metadata": dict(metadata or {}),
        }

    def load_trainable_state(self, state: Dict[str, Any], strict: bool = False) -> Dict[str, Any]:
        load_result = self.load_state_dict(state["model_state_dict"], strict=strict)
        if isinstance(load_result, tuple):
            missing, unexpected = load_result
        else:
            missing = getattr(load_result, "missing_keys", [])
            unexpected = getattr(load_result, "unexpected_keys", [])
        return {
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "metadata": state.get("metadata", {}),
        }

    def save_trainable_checkpoint(self, save_dir: str, metadata: Dict[str, Any]) -> None:
        import os

        os.makedirs(save_dir, exist_ok=True)
        state = self.export_trainable_state(metadata)
        torch.save(state, os.path.join(save_dir, "trainable_state.pt"))
        with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(state["metadata"], f, ensure_ascii=False, indent=2)

    def load_trainable_checkpoint(self, ckpt_dir: str, strict: bool = False) -> Dict[str, Any]:
        import os

        path = os.path.join(ckpt_dir, "trainable_state.pt")
        state = torch.load(path, map_location="cpu")
        return self.load_trainable_state(state, strict=strict)
