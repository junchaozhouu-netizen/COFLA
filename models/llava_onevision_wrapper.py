from __future__ import annotations

import inspect
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

try:
    from transformers import LlavaOnevisionForConditionalGeneration as _LlavaOnevisionModelClass
except Exception:
    _LlavaOnevisionModelClass = None

try:
    from transformers import AutoModelForImageTextToText as _AutoImageTextToText
except Exception:
    _AutoImageTextToText = None

try:
    from transformers import AutoModelForVision2Seq as _AutoVision2Seq
except Exception:
    _AutoVision2Seq = None

from .qwen25vl_wrapper import Qwen25VLWrapper
from .prompting import hateful_memes_yes_no_prompt, mmimdb_multilabel_prompt, nlvr2_yes_no_prompt
from .trainable_scope import (
    collect_trainable_parameter_report,
    count_parameters,
    enforce_llava_onevision_scope_safety,
    freeze_all_parameters,
    normalize_scope_name,
    resolve_trainable_scope,
    unfreeze_modules_by_prefix,
)


class LlavaOnevisionWrapper(Qwen25VLWrapper):
    def __init__(self, args):
        nn.Module.__init__(self)
        self.args = args
        self._input_grad_hook_handle = None
        self.method_name = str(args.method).lower()
        self.dataset_name = str(args.dataset).lower()
        self.model_name = str(getattr(args, "model_name", "llava_onevision_qwen2_0_5b")).lower()
        self.model_family = str(getattr(args, "model_family", "llava_onevision")).lower()
        self.quantization_mode = str(getattr(args, "quantization", "none")).lower()
        self.local_files_only = bool(getattr(args, "local_files_only", True))
        self.trust_remote_code = bool(getattr(args, "trust_remote_code", True))
        self.preview_limit = int(getattr(args, "preview_limit", 50))
        self.min_pixels = int(getattr(args, "min_pixels", 0) or 0)
        self.max_pixels = int(getattr(args, "max_pixels", 0) or 0)
        self.trainable_scope_name = str(getattr(args, "trainable_scope", "auto") or "auto")
        self.num_lm_fusion_layers = int(getattr(args, "num_lm_fusion_layers", 4) or 4)
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

        self.hf_config = AutoConfig.from_pretrained(
            args.model_path,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
        )
        self.model_type = str(getattr(self.hf_config, "model_type", "")).lower().strip()
        if self.model_type != "llava_onevision":
            raise ValueError(
                f"LlavaOnevisionWrapper expected model_type='llava_onevision', but got {self.model_type!r} "
                f"from {args.model_path}"
            )

        # Important for LLaVA-OneVision memory control:
        # the command-line min_pixels/max_pixels must affect this wrapper.  Some
        # processor versions accept these kwargs directly, while others ignore or
        # reject them.  We therefore (1) try to pass them into AutoProcessor, and
        # (2) still enforce max_pixels manually before calling the processor in
        # _encode_single_prompt().
        processor_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        if self.min_pixels > 0:
            processor_kwargs["min_pixels"] = self.min_pixels
        if self.max_pixels > 0:
            processor_kwargs["max_pixels"] = self.max_pixels

        try:
            self.processor = AutoProcessor.from_pretrained(
                args.model_path,
                use_fast=False,
                **processor_kwargs,
            )
        except TypeError:
            # Some LLaVA-OneVision processors do not expose min_pixels/max_pixels
            # in from_pretrained().  Fall back to the standard loader but keep the
            # manual pixel-budget resize path enabled.
            processor_kwargs_fallback = dict(processor_kwargs)
            processor_kwargs_fallback.pop("min_pixels", None)
            processor_kwargs_fallback.pop("max_pixels", None)
            try:
                self.processor = AutoProcessor.from_pretrained(
                    args.model_path,
                    use_fast=False,
                    **processor_kwargs_fallback,
                )
            except TypeError:
                self.processor = AutoProcessor.from_pretrained(args.model_path, **processor_kwargs_fallback)

        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None:
            # Harmless if the processor does not use these attributes; useful for
            # processors that read them after construction.
            if self.min_pixels > 0:
                setattr(image_processor, "min_pixels", self.min_pixels)
            if self.max_pixels > 0:
                setattr(image_processor, "max_pixels", self.max_pixels)

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    args.model_path,
                    use_fast=False,
                    trust_remote_code=self.trust_remote_code,
                    local_files_only=self.local_files_only,
                )
            except TypeError:
                tokenizer = AutoTokenizer.from_pretrained(
                    args.model_path,
                    trust_remote_code=self.trust_remote_code,
                    local_files_only=self.local_files_only,
                )
            setattr(self.processor, "tokenizer", tokenizer)
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        if self.tokenizer.pad_token is None:
            raise RuntimeError(
                f"Tokenizer for {args.model_path} does not expose pad/eos/unk tokens required for batching."
            )

        self.special_token_ids = tuple(
            sorted(
                {
                    int(token_id)
                    for token_id in getattr(self.tokenizer, "all_special_ids", [])
                    if token_id is not None
                }
            )
        )
        self.image_token_id = int(
            getattr(
                self.hf_config,
                "image_token_index",
                getattr(self.hf_config, "image_token_id", -1),
            )
        )
        self.video_token_id = int(getattr(self.hf_config, "video_token_index", -1) or -1)
        excluded_text_tokens = set(self.special_token_ids)
        excluded_text_tokens.discard(self.image_token_id)
        if self.video_token_id >= 0:
            excluded_text_tokens.add(self.video_token_id)
        self.text_proxy_excluded_token_ids = tuple(sorted(excluded_text_tokens))
        self._geometry_warning_keys = set()

        model_cls, need_trust_remote_code = self._resolve_model_class()
        model_load_kwargs = {
            "torch_dtype": self.compute_dtype if self.device_obj.type == "cuda" else torch.float32,
            "local_files_only": self.local_files_only,
        }
        if need_trust_remote_code:
            model_load_kwargs["trust_remote_code"] = self.trust_remote_code
        # Do not force eager attention for LLaVA-OneVision.  Eager attention
        # materializes full [B, H, L, L] attention weights inside Qwen2 and can
        # easily OOM for MSAM/MASAM-style multi-forward training.  SDPA keeps the
        # algorithm unchanged while using a more memory-efficient attention backend.
        model_load_kwargs["attn_implementation"] = "sdpa"

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
        self.image_token_id = int(
            getattr(
                self.config,
                "image_token_index",
                getattr(self.config, "image_token_id", self.image_token_id),
            )
        )
        self.video_token_id = int(getattr(self.config, "video_token_index", self.video_token_id) or self.video_token_id)
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

        self._task_head = None
        if str(args.task_format).lower() == "task_head_cls":
            hidden = self._resolve_hidden_size()
            self._task_head = nn.Linear(hidden, args.num_labels)
            self.add_module("task_head", self._task_head)

    def _resolve_model_class(self):
        if _LlavaOnevisionModelClass is not None:
            return _LlavaOnevisionModelClass, False
        if _AutoImageTextToText is not None:
            return _AutoImageTextToText, True
        if _AutoVision2Seq is not None:
            return _AutoVision2Seq, True
        raise ImportError("No compatible Transformers model loader found for LLaVA-OneVision.")

    def _encode_single_prompt(
        self,
        prompt_text: str,
        images: Optional[List[Any]] = None,
        truncation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        actual_images = [image for image in (images or []) if image is not None]
        # Enforce the command-line max_pixels budget before LLaVA-OneVision expands
        # images into visual tokens.  This is necessary because LLaVA-OneVision
        # disables text truncation for image prompts to preserve image-token/feature
        # alignment, so max_seq_length alone cannot bound the final multimodal length.
        actual_images = [self._resize_image_max_pixels(image) for image in actual_images]
        kwargs: Dict[str, Any] = {
            "text": prompt_text,
            "padding": False,
            "return_tensors": "pt",
        }

        if actual_images:
            # LLaVA-OneVision expands <image> placeholders into a large number of image tokens
            # based on the processed vision features. Truncating the tokenized sequence when
            # images are present can break the one-to-one token/feature alignment and trigger:
            # "Image features and image tokens do not match".
            kwargs["truncation"] = False
            if len(actual_images) == 1:
                kwargs["images"] = actual_images[0]
            else:
                # For multi-image prompts, pass a nested list so the processor treats this as
                # one sample containing multiple images instead of a flat image batch.
                kwargs["images"] = [actual_images]
        else:
            kwargs["truncation"] = bool(truncation)
            if kwargs["truncation"]:
                kwargs["max_length"] = max(8, self.args.max_seq_length - 1)

        return self.processor(**kwargs)

    def _render_prompt(self, messages: List[Dict[str, Any]], add_vision_id: bool = False) -> str:
        apply_chat_template = getattr(self.processor, "apply_chat_template", None)
        if not callable(apply_chat_template):
            apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if not callable(apply_chat_template):
            raise RuntimeError(
                "LLaVA-OneVision processor/tokenizer does not expose apply_chat_template(). "
                "Check the local processor files in the checkpoint."
            )

        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if add_vision_id:
            kwargs["add_vision_id"] = True
        try:
            return apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("add_vision_id", None)
            return apply_chat_template(messages, **kwargs)

    def _resolve_hateful_memes_prompt(self, meme_text: str) -> str:
        template_name = str(getattr(self.args, "prompt_template", "hateful_memes_yes_no") or "").lower().strip()
        if template_name in {"", "auto", "default", "hateful_memes_yes_no"}:
            return hateful_memes_yes_no_prompt(meme_text)
        raise ValueError(
            f"Unsupported LLaVA-OneVision prompt_template={template_name!r}. "
            "Currently only hateful_memes_yes_no is implemented for Hateful Memes."
        )

    @staticmethod
    def _truncate_prompt_text(text: str, max_chars: int) -> str:
        normalized = " ".join(str(text or "").split())
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max(0, max_chars - 3)].rstrip() + "..."

    def _resize_image_max_pixels(self, image: Any):
        if image is None or not isinstance(image, Image.Image):
            return image
        max_pixels = int(getattr(self, "max_pixels", 0) or 0)
        if max_pixels <= 0:
            return image
        width, height = image.size
        area = int(width) * int(height)
        if area <= max_pixels:
            return image
        scale = math.sqrt(float(max_pixels) / float(max(1, area)))
        new_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        resampling_ns = getattr(Image, "Resampling", Image)
        return image.resize(new_size, resample=resampling_ns.BICUBIC)

    @staticmethod
    def _resize_image_max_edge(image: Any, max_edge: int = 384):
        if image is None or not isinstance(image, Image.Image):
            return image
        width, height = image.size
        longest = max(width, height)
        if longest <= max_edge:
            return image
        scale = float(max_edge) / float(longest)
        new_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        resampling_ns = getattr(Image, "Resampling", Image)
        return image.resize(new_size, resample=resampling_ns.BICUBIC)

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
            prompt_text = self._resolve_hateful_memes_prompt(text)
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
            # Prefer native multi-image processor input for NLVR2.
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
            # LLaVA-OneVision image inputs are not processor-truncated, so MM-IMDb needs a
            # bounded text prompt budget and smaller posters to stay within memory.
            compact_title = self._truncate_prompt_text(title, max_chars=96)
            compact_plot = self._truncate_prompt_text(plot, max_chars=640)
            prompt_text = mmimdb_multilabel_prompt(
                title=compact_title,
                plot=compact_plot,
                label_names=label_names,
            )
            content: List[Dict[str, Any]] = []
            image_inputs: Optional[List[Any]] = None
            if image is not None:
                content.append({"type": "image"})
                image_inputs = [self._resize_image_max_edge(image, max_edge=384)]
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

    def describe_trainable_state(self, preview_limit: Optional[int] = None) -> Dict[str, Any]:
        report = collect_trainable_parameter_report(
            self,
            expected_modules=self.lora_target_module_names if self.is_adapter_method else self.scope_config.unfreeze_prefixes,
            preview_limit=int(preview_limit or self.preview_limit),
            lm_layer_boundary=None if ("matched_lora" in str(getattr(self.scope_config, "scope_name", "")) or "branch_lora" in str(getattr(self.scope_config, "scope_name", ""))) else int(self.scope_config.num_lm_layers),
        )
        report["scope_name"] = self.scope_config.scope_name
        report["scope_notes"] = self.scope_config.notes
        return report

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
            "trainable_scope": self.scope_config.scope_name,
            "scope_notes": self.scope_config.notes,
            "image_token_index": self.image_token_id,
            "video_token_index": self.video_token_id,
            "label_info": label_info,
            "quantization": self.quantization_mode,
            "is_quantized_model": self.is_quantized_model,
            "image_pixel_budget": {
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
                "manual_resize_enabled": self.max_pixels > 0,
                "attention_backend": "sdpa",
            },
            "fusion_linear_module_names": list(self.fusion_linear_module_names),
            "parameter_counts": count_parameters(self),
            "trainable_report": self.describe_trainable_state(preview_limit=self.preview_limit),
            "proxy_definition": {
                "visual_proxy": "Layer-0 mixed hidden states restricted to image placeholder token positions after multi_modal_projector injection.",
                "text_proxy": "Layer-0 mixed hidden states restricted to non-image text token positions after excluding special and supervision tokens.",
                "fusion_path": "multi_modal_projector plus the configured early language_model.model.layers modules.",
            },
        }

    def _verify_trainable_scope(self) -> None:
        report = self.describe_trainable_state(preview_limit=self.preview_limit)
        enforce_llava_onevision_scope_safety(
            report,
            scope_name=self.scope_config.scope_name,
            lm_layer_boundary=int(self.scope_config.num_lm_layers),
            adapter_expected=self.is_adapter_method,
            allow_lm_head=False,
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
        self._verify_trainable_scope()

    def _setup_vanilla_ft(self) -> None:
        self._validate_quantization_configuration()
        freeze_all_parameters(self.model)
        unfreeze_modules_by_prefix(self.model, self.scope_config.unfreeze_prefixes)
        if self._task_head is not None:
            for param in self._task_head.parameters():
                param.requires_grad = True

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
            target_modules=list(self.scope_config.lora_target_modules),
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_cfg)
        if self._task_head is not None:
            for param in self._task_head.parameters():
                param.requires_grad = True

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
            language_model = getattr(obj, "language_model", None)
            decoder = getattr(language_model, "model", None)
            if decoder is not None and hasattr(decoder, "layers"):
                return decoder
            inner = getattr(obj, "model", None)
            if inner is not None and hasattr(inner, "layers"):
                return inner
            candidates.append(getattr(obj, "model", None))
            candidates.append(getattr(obj, "base_model", None))
            candidates.append(language_model)
        raise RuntimeError("Could not locate the LLaVA-OneVision decoder model for registering layer-0 hooks.")

    def _get_forward_target(self):
        model = self.model
        if not hasattr(model, "peft_config"):
            return model

        getter = getattr(model, "get_base_model", None)
        if callable(getter):
            try:
                base_model = getter()
                if base_model is not None and callable(getattr(base_model, "forward", None)):
                    return base_model
            except Exception:
                pass

        base_model = getattr(model, "base_model", None)
        for candidate in (getattr(base_model, "model", None), base_model):
            if candidate is not None and callable(getattr(candidate, "forward", None)):
                return candidate
        return model

    def _filter_forward_kwargs(self, target, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        forward = getattr(target, "forward", None)
        if forward is None:
            return kwargs
        try:
            signature = inspect.signature(forward)
        except (TypeError, ValueError):
            return kwargs

        accepts_var_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )
        if accepts_var_kwargs:
            return kwargs

        allowed = set(signature.parameters.keys())
        filtered = {key: value for key, value in kwargs.items() if key in allowed}
        removed = sorted(set(kwargs.keys()) - allowed)
        if removed:
            self._warn_geometry_once(
                f"filtered_forward_kwargs::{','.join(removed)}",
                "LLaVA-OneVision forward does not accept some generic wrapper kwargs; "
                f"dropping {removed} before calling the underlying model.",
            )
        return filtered

    def _call_model(self, inputs: Dict[str, torch.Tensor], labels: Optional[torch.Tensor] = None, **extra_kwargs):
        model_device = self._get_model_input_device()
        kwargs = self._move_any_to_device(dict(inputs), model_device)
        if labels is not None:
            kwargs["labels"] = labels.to(model_device, non_blocking=True)
        kwargs.update(extra_kwargs)
        forward_target = self._get_forward_target()
        kwargs["return_dict"] = True
        kwargs = self._filter_forward_kwargs(forward_target, kwargs)
        return forward_target(**kwargs)

    def _build_branch_masks(self, batch: Dict[str, Any], hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = hidden.device
        input_ids = batch["full_inputs"]["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["full_inputs"]["attention_mask"].to(device, non_blocking=True).bool()
        if self.image_token_id < 0:
            raise RuntimeError(
                "LLaVA-OneVision config does not expose image_token_index/image_token_id, so geometry token masks "
                "cannot be built reliably for this checkpoint."
            )

        image_token_mask = (input_ids == self.image_token_id) & attention_mask
        special_token_mask = self._build_special_token_mask(input_ids, attention_mask)

        supervision_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        if batch.get("lm_labels") is not None:
            supervision_mask = batch["lm_labels"].to(device, non_blocking=True).ne(-100) & attention_mask

        # These decoder layer-0 states are post-projector mixed-token activations.
        # They are practical branch proxies for COFLA/FAST-COFLA, not the idealized paper z_v / z_t variables.
        text_token_mask = attention_mask & (~image_token_mask) & (~special_token_mask) & (~supervision_mask)
        if not bool(text_token_mask.any()):
            self._warn_geometry_once(
                "empty_text_proxy",
                "LLaVA-OneVision text proxy mask is empty after excluding image/special/supervision tokens; "
                "text-side geometry will default to zero for this batch.",
            )
        return self._expand_token_mask(image_token_mask, hidden), self._expand_token_mask(text_token_mask, hidden)

    def compute_geometry_probe(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs, cache = self.compute_base_loss(batch, capture_layer0=True)
        probe_loss = outputs.loss
        hidden = cache.get("hidden")
        if hidden is None:
            raise RuntimeError("Failed to capture LLaVA-OneVision layer-0 hidden states for geometry computation.")

        eps = self._geometry_eps()
        fusion_named = self.get_fusion_trainable_named_parameters()
        grad_inputs = [hidden] + [param for _, param in fusion_named]
        grads = torch.autograd.grad(
            outputs=probe_loss,
            inputs=grad_inputs,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        hidden_grad = grads[0]
        if hidden_grad is None:
            raise RuntimeError("Failed to compute gradients w.r.t. LLaVA-OneVision captured layer-0 hidden states.")

        image_mask, text_mask = self._build_branch_masks(batch, hidden)
        if not bool(image_mask.any()):
            raise RuntimeError(
                "LLaVA-OneVision geometry probe could not find any image placeholder tokens in this batch. "
                "This usually means the processor/chat template formatting changed and the image/text proxy mask "
                "needs to be updated for this checkpoint before geometry metrics can be trusted."
            )

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
            grad_fp32 = grad.detach().float()
            grad_map[name] = grad_fp32
            sq += grad_fp32.pow(2).sum().item()
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
            "param_perturb": {key: value.detach() for key, value in param_perturb.items()},
            "gv_norm": gv_norm.detach(),
            "gt_norm": gt_norm.detach(),
            "gf_norm": gf_norm.detach(),
        }
