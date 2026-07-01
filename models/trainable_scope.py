from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


QWEN_FUSION_VISUAL_EXACT = {"visual.merger.mlp.0", "visual.merger.mlp.2"}
INTERNVL3_PROJECTOR_MODULES = ("mlp1.1", "mlp1.3")
INTERNVL3_LM_FUSION_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
LLAVA_ONEVISION_PROJECTOR_MODULES = (
    "multi_modal_projector.linear_1",
    "multi_modal_projector.linear_2",
)
LLAVA_ONEVISION_LM_FUSION_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

_KNOWN_WRAPPER_PREFIXES = (
    "model.base_model.model.",
    "base_model.model.",
    "module.model.",
    "module.",
    "model.",
)


@dataclass(frozen=True)
class TrainableScopeConfig:
    model_family: str
    scope_name: str
    fusion_module_names: Tuple[str, ...]
    lora_target_modules: Tuple[str, ...]
    unfreeze_prefixes: Tuple[str, ...]
    num_lm_layers: int = 0
    branch_visual_module_names: Tuple[str, ...] = ()
    branch_text_module_names: Tuple[str, ...] = ()
    notes: str = ""

    @property
    def branch_module_names(self) -> Tuple[str, ...]:
        return _deduplicate(self.branch_visual_module_names + self.branch_text_module_names)

    @property
    def all_peft_module_names(self) -> Tuple[str, ...]:
        return _deduplicate(self.branch_visual_module_names + self.branch_text_module_names + self.fusion_module_names)


def normalize_scope_name(name: str) -> str:
    normalized = str(name)
    for prefix in _KNOWN_WRAPPER_PREFIXES:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    return normalized


def _collect_linear_module_names(model: nn.Module) -> set[str]:
    return {name for name, module in model.named_modules() if isinstance(module, nn.Linear)}


def _deduplicate(items: Iterable[str]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(str(item) for item in items if str(item)))


def _collect_layer_indices(linear_names: set[str], pattern: str) -> List[int]:
    indices: List[int] = []
    regex = re.compile(pattern)
    for name in linear_names:
        match = regex.search(name)
        if match:
            indices.append(int(match.group(1)))
    return sorted(set(indices))


def _select_last_indices(indices: Sequence[int], count: int) -> List[int]:
    if count <= 0:
        return []
    ordered = sorted(set(int(i) for i in indices))
    return ordered[-int(count):]


def _linear_modules_for_prefixes(
    model: nn.Module,
    prefixes: Sequence[str],
    *,
    suffixes: Optional[Sequence[str]] = None,
) -> List[str]:
    linear_names = _collect_linear_module_names(model)
    out: List[str] = []
    suffix_tuple = tuple(str(s) for s in suffixes) if suffixes is not None else None
    for name in linear_names:
        if not any(name.startswith(prefix + ".") or name == prefix for prefix in prefixes):
            continue
        if suffix_tuple is not None and not any(name.endswith(suffix) for suffix in suffix_tuple):
            continue
        out.append(name)
    return sorted(set(out))


QWEN_VISION_SUFFIXES = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
QWEN_LM_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
INTERNVL3_VISION_SUFFIXES = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
LLAVA_ONEVISION_VISION_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.out_proj",
    "mlp.fc1",
    "mlp.fc2",
)


def _discover_qwen_vision_branch_modules(model: nn.Module, num_branch_vision_layers: int) -> List[str]:
    linear_names = _collect_linear_module_names(model)
    indices = _collect_layer_indices(linear_names, r"^visual\.blocks\.(\d+)\.")
    prefixes = [f"visual.blocks.{i}" for i in _select_last_indices(indices, num_branch_vision_layers)]
    return _linear_modules_for_prefixes(model, prefixes, suffixes=QWEN_VISION_SUFFIXES)


def _discover_qwen_text_branch_modules(model: nn.Module, num_branch_lm_layers: int) -> List[str]:
    linear_names = _collect_linear_module_names(model)
    indices = _collect_layer_indices(linear_names, r"^model\.layers\.(\d+)\.")
    prefixes = [f"model.layers.{i}" for i in _select_last_indices(indices, num_branch_lm_layers)]
    return _linear_modules_for_prefixes(model, prefixes, suffixes=QWEN_LM_SUFFIXES)


def _discover_qwen_lm_fusion_modules(model: nn.Module, num_fusion_layers: int) -> List[str]:
    if num_fusion_layers <= 0:
        return []
    prefixes = [f"model.layers.{i}" for i in range(int(num_fusion_layers))]
    return _linear_modules_for_prefixes(model, prefixes, suffixes=QWEN_LM_SUFFIXES)


def _discover_llava_onevision_vision_branch_modules(model: nn.Module, num_branch_vision_layers: int) -> List[str]:
    linear_names = _collect_linear_module_names(model)
    indices = _collect_layer_indices(linear_names, r"^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.")
    prefixes = [f"vision_tower.vision_model.encoder.layers.{i}" for i in _select_last_indices(indices, num_branch_vision_layers)]
    return _linear_modules_for_prefixes(model, prefixes, suffixes=LLAVA_ONEVISION_VISION_SUFFIXES)


def _discover_llava_onevision_text_branch_modules(model: nn.Module, num_branch_lm_layers: int) -> List[str]:
    linear_names = _collect_linear_module_names(model)
    indices = _collect_layer_indices(linear_names, r"^language_model\.model\.layers\.(\d+)\.")
    prefixes = [f"language_model.model.layers.{i}" for i in _select_last_indices(indices, num_branch_lm_layers)]
    return _linear_modules_for_prefixes(model, prefixes, suffixes=LLAVA_ONEVISION_LM_FUSION_SUFFIXES)


def _discover_internvl3_vision_branch_modules(model: nn.Module, num_branch_vision_layers: int) -> List[str]:
    linear_names = _collect_linear_module_names(model)
    indices = _collect_layer_indices(linear_names, r"^vision_model\.encoder\.layers\.(\d+)\.")
    prefixes = [f"vision_model.encoder.layers.{i}" for i in _select_last_indices(indices, num_branch_vision_layers)]
    return _linear_modules_for_prefixes(model, prefixes, suffixes=INTERNVL3_VISION_SUFFIXES)


def _discover_internvl3_text_branch_modules(model: nn.Module, num_branch_lm_layers: int) -> List[str]:
    linear_names = _collect_linear_module_names(model)
    indices = _collect_layer_indices(linear_names, r"^language_model\.model\.layers\.(\d+)\.")
    prefixes = [f"language_model.model.layers.{i}" for i in _select_last_indices(indices, num_branch_lm_layers)]
    return _linear_modules_for_prefixes(model, prefixes, suffixes=INTERNVL3_LM_FUSION_SUFFIXES)


def discover_fusion_linear_module_names(model: nn.Module, num_fusion_layers: int = 4) -> List[str]:
    names: List[str] = []
    layer_prefixes = [f"model.layers.{i}.self_attn" for i in range(num_fusion_layers)] + [
        f"model.layers.{i}.mlp" for i in range(num_fusion_layers)
    ]

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name in QWEN_FUSION_VISUAL_EXACT:
            names.append(name)
            continue
        if any(name.startswith(prefix) for prefix in layer_prefixes):
            names.append(name)
    return sorted(set(names))


def _discover_internvl3_projector_modules(model: nn.Module) -> List[str]:
    linear_names = _collect_linear_module_names(model)
    found = [name for name in INTERNVL3_PROJECTOR_MODULES if name in linear_names]
    if len(found) != len(INTERNVL3_PROJECTOR_MODULES):
        missing = sorted(set(INTERNVL3_PROJECTOR_MODULES) - set(found))
        raise RuntimeError(
            "InternVL3 projector scope discovery failed. Missing expected projector modules: "
            f"{missing}. Re-run the structure inspection script against the current checkpoint."
        )
    return found


def _discover_internvl3_lm_fusion_modules(model: nn.Module, num_lm_fusion_layers: int) -> List[str]:
    if num_lm_fusion_layers <= 0:
        return []

    linear_names = _collect_linear_module_names(model)
    found: List[str] = []
    missing: List[str] = []
    for layer_idx in range(num_lm_fusion_layers):
        prefix = f"language_model.model.layers.{layer_idx}"
        for suffix in INTERNVL3_LM_FUSION_SUFFIXES:
            name = f"{prefix}.{suffix}"
            if name in linear_names:
                found.append(name)
            else:
                missing.append(name)
    if missing:
        raise RuntimeError(
            "InternVL3 LM fusion scope discovery failed. Missing expected modules: "
            f"{missing[:16]}{' ...' if len(missing) > 16 else ''}. "
            "Re-run the structure inspection script against the current checkpoint."
        )
    return found


def _discover_llava_onevision_projector_modules(model: nn.Module) -> List[str]:
    linear_names = _collect_linear_module_names(model)
    found = [name for name in LLAVA_ONEVISION_PROJECTOR_MODULES if name in linear_names]
    if len(found) != len(LLAVA_ONEVISION_PROJECTOR_MODULES):
        missing = sorted(set(LLAVA_ONEVISION_PROJECTOR_MODULES) - set(found))
        raise RuntimeError(
            "LLaVA-OneVision projector scope discovery failed. Missing expected projector modules: "
            f"{missing}. Re-run the structure inspection script against the current checkpoint."
        )
    return found


def _discover_llava_onevision_lm_fusion_modules(model: nn.Module, num_lm_fusion_layers: int) -> List[str]:
    if num_lm_fusion_layers <= 0:
        return []

    linear_names = _collect_linear_module_names(model)
    found: List[str] = []
    missing: List[str] = []
    for layer_idx in range(num_lm_fusion_layers):
        prefix = f"language_model.model.layers.{layer_idx}"
        for suffix in LLAVA_ONEVISION_LM_FUSION_SUFFIXES:
            name = f"{prefix}.{suffix}"
            if name in linear_names:
                found.append(name)
            else:
                missing.append(name)
    if missing:
        raise RuntimeError(
            "LLaVA-OneVision LM fusion scope discovery failed. Missing expected modules: "
            f"{missing[:16]}{' ...' if len(missing) > 16 else ''}. "
            "Re-run the structure inspection script against the current checkpoint."
        )
    return found


def resolve_trainable_scope(
    model: nn.Module,
    model_family: str,
    trainable_scope: Optional[str] = None,
    *,
    num_fusion_layers: int = 4,
    num_lm_fusion_layers: int = 4,
    num_branch_vision_layers: int = 2,
    num_branch_lm_layers: int = 2,
) -> TrainableScopeConfig:
    """Resolve VLM PEFT/LoRA trainable scopes.

    For pretrained VLM experiments, the canonical main scope is matched LoRA:
    both unimodal branch modules and fusion-sensitive modules receive LoRA adapters.

    Supported canonical scopes:
      * ``*_branch_lora``:
        branch-side visual/text modules receive LoRA; fusion-sensitive modules are frozen.
      * ``*_fusion_lora``:
        fusion-sensitive modules receive LoRA; branch-side visual/text modules are frozen.
      * ``*_matched_lora``:
        branch-side visual/text modules and fusion-sensitive modules all receive LoRA.
        This is the main VLM setting corresponding to "branch LoRA + fusion LoRA".
      * ``*_branch_lora_fusion_lora``:
        explicit alias of ``*_matched_lora`` for readability.

    Important:
      ``*_branch_lora_fusion_full`` is intentionally not supported for pretrained VLM
      PEFT experiments. That design belongs to controlled late-fusion models where the
      fusion block is a small explicit MLP/Gated-MLP trained in full precision. In VLMs,
      fusion-sensitive modules are large pretrained layers and should be adapted with LoRA.
    """
    family = str(model_family).lower().strip()
    scope = str(trainable_scope or "auto").lower().strip()
    branch_v_layers = int(max(0, num_branch_vision_layers or 0))
    branch_l_layers = int(max(0, num_branch_lm_layers or 0))

    def _require_nonempty(modules: Sequence[str], message: str) -> None:
        if not tuple(modules):
            raise RuntimeError(message)

    def _reject_fusion_full_scope() -> None:
        raise ValueError(
            f"Unsupported VLM trainable_scope={trainable_scope!r}. "
            "For pretrained VLMs, use branch+fusion LoRA instead: "
            "internvl3_matched_lora / llava_onevision_matched_lora / qwen_matched_lora, "
            "or the explicit alias *_branch_lora_fusion_lora. "
            "The *_branch_lora_fusion_full design is only for controlled late-fusion models."
        )

    if family == "qwen":
        resolved_scope = scope if scope not in {"", "auto", "default"} else "qwen_matched_lora"
        branch_visual = tuple(_discover_qwen_vision_branch_modules(model, branch_v_layers))
        branch_text = tuple(_discover_qwen_text_branch_modules(model, branch_l_layers))
        branch_modules = _deduplicate(branch_visual + branch_text)

        linear_names = _collect_linear_module_names(model)
        visual_fusion = sorted(name for name in QWEN_FUSION_VISUAL_EXACT if name in linear_names)
        fusion_modules = tuple(
            _deduplicate(
                visual_fusion
                + _discover_qwen_lm_fusion_modules(model, num_fusion_layers=num_fusion_layers)
            )
        )

        if resolved_scope in {"qwen_branch_lora", "branch_lora"}:
            _require_nonempty(branch_modules, "Qwen branch_lora scope discovery returned no branch LoRA target modules.")
            return TrainableScopeConfig(
                model_family="qwen",
                scope_name="qwen_branch_lora",
                fusion_module_names=fusion_modules,
                branch_visual_module_names=branch_visual,
                branch_text_module_names=branch_text,
                lora_target_modules=branch_modules,
                unfreeze_prefixes=(),
                num_lm_layers=num_fusion_layers,
                notes=(
                    f"Qwen branch-only LoRA scope: visual last {branch_v_layers} layers and "
                    f"language last {branch_l_layers} layers receive LoRA; fusion-sensitive modules are frozen."
                ),
            )

        if resolved_scope in {"qwen_fusion_lora", "qwen_fusion_default", "qwen_fusion_only"}:
            _require_nonempty(fusion_modules, "Qwen fusion_lora scope discovery returned no fusion LoRA target modules.")
            return TrainableScopeConfig(
                model_family="qwen",
                scope_name="qwen_fusion_lora" if resolved_scope == "qwen_fusion_lora" else "qwen_fusion_default",
                fusion_module_names=fusion_modules,
                lora_target_modules=fusion_modules,
                unfreeze_prefixes=fusion_modules,
                num_lm_layers=num_fusion_layers,
                notes="Qwen fusion-only LoRA scope: merger/early fusion-sensitive language modules receive LoRA; branches are frozen.",
            )

        if resolved_scope in {"qwen_branch_lora_fusion_full", "branch_lora_fusion_full"}:
            _reject_fusion_full_scope()

        matched_aliases = {
            "qwen_matched_lora",
            "vlm_matched_lora",
            "matched_lora",
            "qwen_branch_lora_fusion_lora",
            "branch_lora_fusion_lora",
        }
        if resolved_scope not in matched_aliases:
            raise ValueError(
                f"Unsupported Qwen trainable_scope={trainable_scope!r}. Use qwen_branch_lora, "
                "qwen_fusion_lora, qwen_matched_lora, or qwen_branch_lora_fusion_lora."
            )

        targets = _deduplicate(branch_modules + fusion_modules)
        _require_nonempty(targets, "Qwen matched LoRA scope discovery returned no target modules.")
        return TrainableScopeConfig(
            model_family="qwen",
            scope_name="qwen_matched_lora",
            fusion_module_names=fusion_modules,
            branch_visual_module_names=branch_visual,
            branch_text_module_names=branch_text,
            lora_target_modules=targets,
            unfreeze_prefixes=targets,
            num_lm_layers=num_fusion_layers,
            notes=(
                f"Matched VLM PEFT scope for Qwen: visual last {branch_v_layers} layers, "
                f"language last {branch_l_layers} layers, merger, and early {num_fusion_layers} fusion-sensitive LM layers all receive LoRA."
            ),
        )

    if family == "llava_onevision":
        resolved_scope = scope if scope not in {"", "auto", "default"} else "llava_onevision_matched_lora"
        lm_layers = int(num_lm_fusion_layers or 4)
        branch_visual = tuple(_discover_llava_onevision_vision_branch_modules(model, branch_v_layers))
        branch_text = tuple(_discover_llava_onevision_text_branch_modules(model, branch_l_layers))
        branch_modules = _deduplicate(branch_visual + branch_text)
        projector_modules = tuple(_discover_llava_onevision_projector_modules(model))

        if resolved_scope == "llava_onevision_fusion_minimal":
            fusion_modules = projector_modules
            return TrainableScopeConfig(
                model_family="llava_onevision",
                scope_name=resolved_scope,
                fusion_module_names=fusion_modules,
                lora_target_modules=fusion_modules,
                unfreeze_prefixes=fusion_modules,
                num_lm_layers=0,
                notes="Legacy LLaVA-OneVision projector-only fusion LoRA scope for low-memory debugging.",
            )

        fusion_modules = tuple(
            list(projector_modules)
            + _discover_llava_onevision_lm_fusion_modules(model, num_lm_fusion_layers=lm_layers)
        )

        if resolved_scope in {"llava_onevision_branch_lora", "branch_lora"}:
            _require_nonempty(branch_modules, "LLaVA-OneVision branch_lora scope discovery returned no branch LoRA target modules.")
            return TrainableScopeConfig(
                model_family="llava_onevision",
                scope_name="llava_onevision_branch_lora",
                fusion_module_names=fusion_modules,
                branch_visual_module_names=branch_visual,
                branch_text_module_names=branch_text,
                lora_target_modules=branch_modules,
                unfreeze_prefixes=(),
                num_lm_layers=lm_layers,
                notes=(
                    f"LLaVA-OneVision branch-only LoRA scope: SigLIP vision last {branch_v_layers} layers and "
                    f"Qwen2 language last {branch_l_layers} layers receive LoRA; fusion path is frozen."
                ),
            )

        if resolved_scope in {"llava_onevision_fusion_lora", "llava_onevision_fusion_early4"}:
            _require_nonempty(fusion_modules, "LLaVA-OneVision fusion_lora scope discovery returned no fusion LoRA target modules.")
            return TrainableScopeConfig(
                model_family="llava_onevision",
                scope_name="llava_onevision_fusion_lora" if resolved_scope == "llava_onevision_fusion_lora" else resolved_scope,
                fusion_module_names=fusion_modules,
                lora_target_modules=fusion_modules,
                unfreeze_prefixes=fusion_modules,
                num_lm_layers=lm_layers,
                notes="LLaVA-OneVision fusion-only LoRA scope: projector plus early Qwen2 fusion-sensitive layers receive LoRA.",
            )

        if resolved_scope in {"llava_onevision_branch_lora_fusion_full", "branch_lora_fusion_full"}:
            _reject_fusion_full_scope()

        matched_aliases = {
            "llava_onevision_matched_lora",
            "vlm_matched_lora",
            "matched_lora",
            "llava_onevision_branch_lora_fusion_lora",
            "branch_lora_fusion_lora",
        }
        if resolved_scope not in matched_aliases:
            raise ValueError(
                f"Unsupported LLaVA-OneVision trainable_scope={trainable_scope!r}. Use "
                "llava_onevision_branch_lora, llava_onevision_fusion_lora, "
                "llava_onevision_matched_lora, or llava_onevision_branch_lora_fusion_lora."
            )

        targets = _deduplicate(branch_modules + fusion_modules)
        _require_nonempty(targets, "LLaVA-OneVision matched LoRA scope discovery returned no target modules.")
        return TrainableScopeConfig(
            model_family="llava_onevision",
            scope_name="llava_onevision_matched_lora",
            fusion_module_names=fusion_modules,
            branch_visual_module_names=branch_visual,
            branch_text_module_names=branch_text,
            lora_target_modules=targets,
            unfreeze_prefixes=targets,
            num_lm_layers=lm_layers,
            notes=(
                f"Matched VLM PEFT scope for LLaVA-OneVision: SigLIP vision last {branch_v_layers} layers, "
                f"Qwen2 language last {branch_l_layers} layers, projector, and early {lm_layers} fusion-sensitive LM layers all receive LoRA."
            ),
        )

    if family != "internvl":
        raise ValueError(f"Unsupported model family for trainable scope resolution: {model_family!r}")

    resolved_scope = scope if scope not in {"", "auto", "default"} else "internvl3_matched_lora"
    lm_layers = int(num_lm_fusion_layers or 4)
    branch_visual = tuple(_discover_internvl3_vision_branch_modules(model, branch_v_layers))
    branch_text = tuple(_discover_internvl3_text_branch_modules(model, branch_l_layers))
    branch_modules = _deduplicate(branch_visual + branch_text)
    projector_modules = tuple(_discover_internvl3_projector_modules(model))

    if resolved_scope == "internvl3_fusion_minimal":
        fusion_modules = projector_modules
        return TrainableScopeConfig(
            model_family="internvl",
            scope_name=resolved_scope,
            fusion_module_names=fusion_modules,
            lora_target_modules=fusion_modules,
            unfreeze_prefixes=fusion_modules,
            num_lm_layers=0,
            notes="Legacy InternVL3 projector-only fusion LoRA scope for low-memory debugging.",
        )

    fusion_modules = tuple(
        list(projector_modules)
        + _discover_internvl3_lm_fusion_modules(model, num_lm_fusion_layers=lm_layers)
    )

    if resolved_scope in {"internvl3_branch_lora", "branch_lora"}:
        _require_nonempty(branch_modules, "InternVL3 branch_lora scope discovery returned no branch LoRA target modules.")
        return TrainableScopeConfig(
            model_family="internvl",
            scope_name="internvl3_branch_lora",
            fusion_module_names=fusion_modules,
            branch_visual_module_names=branch_visual,
            branch_text_module_names=branch_text,
            lora_target_modules=branch_modules,
            unfreeze_prefixes=(),
            num_lm_layers=lm_layers,
            notes=(
                f"InternVL3 branch-only LoRA scope: InternViT last {branch_v_layers} layers and "
                f"Qwen2 language last {branch_l_layers} layers receive LoRA; fusion path is frozen."
            ),
        )

    if resolved_scope in {"internvl3_fusion_lora", "internvl3_fusion_early4"}:
        _require_nonempty(fusion_modules, "InternVL3 fusion_lora scope discovery returned no fusion LoRA target modules.")
        return TrainableScopeConfig(
            model_family="internvl",
            scope_name="internvl3_fusion_lora" if resolved_scope == "internvl3_fusion_lora" else resolved_scope,
            fusion_module_names=fusion_modules,
            lora_target_modules=fusion_modules,
            unfreeze_prefixes=fusion_modules,
            num_lm_layers=lm_layers,
            notes="InternVL3 fusion-only LoRA scope: mlp1 projector plus early language-model fusion-sensitive layers receive LoRA.",
        )

    if resolved_scope == "internvl3_broader":
        broader_layers = int(max(4, num_lm_fusion_layers or 4))
        broader_fusion_modules = tuple(
            list(projector_modules)
            + _discover_internvl3_lm_fusion_modules(model, num_lm_fusion_layers=broader_layers)
        )
        return TrainableScopeConfig(
            model_family="internvl",
            scope_name=resolved_scope,
            fusion_module_names=broader_fusion_modules,
            lora_target_modules=broader_fusion_modules,
            unfreeze_prefixes=broader_fusion_modules,
            num_lm_layers=broader_layers,
            notes="Legacy broader InternVL3 fusion-only LoRA scope. Vision modules remain frozen.",
        )

    if resolved_scope in {"internvl3_branch_lora_fusion_full", "branch_lora_fusion_full"}:
        _reject_fusion_full_scope()

    matched_aliases = {
        "internvl3_matched_lora",
        "vlm_matched_lora",
        "matched_lora",
        "internvl3_branch_lora_fusion_lora",
        "branch_lora_fusion_lora",
    }
    if resolved_scope not in matched_aliases:
        raise ValueError(
            f"Unsupported InternVL3 trainable_scope={trainable_scope!r}. Use internvl3_branch_lora, "
            "internvl3_fusion_lora, internvl3_matched_lora, or internvl3_branch_lora_fusion_lora."
        )

    targets = _deduplicate(branch_modules + fusion_modules)
    _require_nonempty(targets, "InternVL3 matched LoRA scope discovery returned no target modules.")
    return TrainableScopeConfig(
        model_family="internvl",
        scope_name="internvl3_matched_lora",
        fusion_module_names=fusion_modules,
        branch_visual_module_names=branch_visual,
        branch_text_module_names=branch_text,
        lora_target_modules=targets,
        unfreeze_prefixes=targets,
        num_lm_layers=lm_layers,
        notes=(
            f"Matched VLM PEFT scope for InternVL3: InternViT last {branch_v_layers} layers, "
            f"Qwen2 language last {branch_l_layers} layers, mlp1 projector, and early {lm_layers} fusion-sensitive LM layers all receive LoRA."
        ),
    )

def freeze_all_parameters(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_modules_by_prefix(model: nn.Module, prefixes: Sequence[str]) -> List[str]:
    activated: List[str] = []
    normalized_prefixes = tuple(normalize_scope_name(prefix) for prefix in prefixes if str(prefix))
    if not normalized_prefixes:
        return activated
    for name, param in model.named_parameters():
        normalized = normalize_scope_name(name)
        if any(normalized == prefix or normalized.startswith(prefix + ".") for prefix in normalized_prefixes):
            param.requires_grad = True
            activated.append(name)
    return activated


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = 0
    trainable = 0
    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def _extract_lm_layer_index(name: str) -> Optional[int]:
    normalized = normalize_scope_name(name)
    for pattern in (r"^language_model\.model\.layers\.(\d+)\.", r"^model\.layers\.(\d+)\."):
        match = re.search(pattern, normalized)
        if match:
            return int(match.group(1))
    return None


def _contains_embedding(name: str) -> bool:
    normalized = normalize_scope_name(name)
    return (
        normalized.startswith("language_model.model.embed_tokens")
        or normalized.startswith("model.embed_tokens")
        or ".embeddings." in normalized
        or normalized.endswith(".embeddings")
    )


def _contains_lm_head(name: str) -> bool:
    normalized = normalize_scope_name(name)
    return normalized.startswith("language_model.lm_head") or normalized.startswith("lm_head")


def _hits_vision_model(name: str) -> bool:
    normalized = normalize_scope_name(name)
    return (
        normalized.startswith("vision_model")
        or normalized.startswith("vision_tower")
        or normalized.startswith("visual")
    )


def collect_lora_target_module_hits(model: nn.Module) -> List[str]:
    hits: List[str] = []
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A"):
            continue
        lora_a = getattr(module, "lora_A", None)
        try:
            active = len(lora_a) > 0
        except Exception:
            active = bool(lora_a)
        if active:
            hits.append(normalize_scope_name(name))
    return sorted(set(hits))


def collect_trainable_parameter_report(
    model: nn.Module,
    *,
    expected_modules: Optional[Sequence[str]] = None,
    preview_limit: int = 50,
    lm_layer_boundary: Optional[int] = None,
) -> Dict[str, object]:
    counts = count_parameters(model)
    total = int(counts["total"])
    trainable = int(counts["trainable"])
    trainable_ratio = float(trainable / total) if total else 0.0

    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    normalized_trainable = [normalize_scope_name(name) for name in trainable_names]
    actual_lora_hits = collect_lora_target_module_hits(model)
    expected = set(normalize_scope_name(name) for name in expected_modules or [])

    flags = {
        "hits_vision_model": any(_hits_vision_model(name) for name in normalized_trainable + actual_lora_hits),
        "hits_embedding": any(_contains_embedding(name) for name in normalized_trainable + actual_lora_hits),
        "hits_lm_head": any(_contains_lm_head(name) for name in normalized_trainable + actual_lora_hits),
    }
    if lm_layer_boundary is not None:
        flags["hits_lm_layers_after_boundary"] = any(
            (layer_idx is not None and layer_idx >= int(lm_layer_boundary))
            for layer_idx in (_extract_lm_layer_index(name) for name in normalized_trainable + actual_lora_hits)
        )
    else:
        flags["hits_lm_layers_after_boundary"] = False

    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": int(counts["frozen"]),
        "trainable_ratio": trainable_ratio,
        "trainable_parameter_names_preview": trainable_names[: max(1, int(preview_limit))],
        "actual_lora_target_modules": actual_lora_hits,
        "expected_target_modules": sorted(expected),
        "missing_expected_modules": sorted(expected - set(actual_lora_hits)) if expected else [],
        "unexpected_lora_modules": sorted(set(actual_lora_hits) - expected) if expected else [],
        "flags": flags,
    }


def enforce_internvl_scope_safety(
    report: Dict[str, object],
    *,
    scope_name: str,
    lm_layer_boundary: int,
    adapter_expected: bool,
    allow_lm_head: bool = False,
) -> None:
    flags = dict(report.get("flags", {}))
    scope_text = str(scope_name).lower()
    branch_scope = ("matched_lora" in scope_text) or ("branch_lora" in scope_text)
    if flags.get("hits_vision_model") and not branch_scope:
        raise RuntimeError(f"{scope_name} unexpectedly activated InternVL3 vision_model parameters.")
    if flags.get("hits_embedding"):
        raise RuntimeError(f"{scope_name} unexpectedly activated InternVL3 embedding parameters.")
    if flags.get("hits_lm_layers_after_boundary") and not branch_scope:
        raise RuntimeError(
            f"{scope_name} unexpectedly activated language_model.model.layers.{lm_layer_boundary} and above."
        )
    if flags.get("hits_lm_head") and not allow_lm_head:
        raise RuntimeError(f"{scope_name} unexpectedly activated lm_head parameters.")

    if adapter_expected:
        actual_hits = list(report.get("actual_lora_target_modules", []))
        unexpected_hits = list(report.get("unexpected_lora_modules", []))
        if not actual_hits:
            raise RuntimeError(f"{scope_name} did not inject any LoRA target modules.")
        if unexpected_hits:
            raise RuntimeError(
                f"{scope_name} injected LoRA into unexpected modules: {unexpected_hits[:16]}"
                f"{' ...' if len(unexpected_hits) > 16 else ''}"
            )


def enforce_llava_onevision_scope_safety(
    report: Dict[str, object],
    *,
    scope_name: str,
    lm_layer_boundary: int,
    adapter_expected: bool,
    allow_lm_head: bool = False,
) -> None:
    flags = dict(report.get("flags", {}))
    scope_text = str(scope_name).lower()
    branch_scope = ("matched_lora" in scope_text) or ("branch_lora" in scope_text)
    if flags.get("hits_vision_model") and not branch_scope:
        raise RuntimeError(f"{scope_name} unexpectedly activated LLaVA-OneVision vision_tower parameters.")
    if flags.get("hits_embedding"):
        raise RuntimeError(f"{scope_name} unexpectedly activated LLaVA-OneVision embedding parameters.")
    if flags.get("hits_lm_layers_after_boundary") and not branch_scope:
        raise RuntimeError(
            f"{scope_name} unexpectedly activated language_model.model.layers.{lm_layer_boundary} and above."
        )
    if flags.get("hits_lm_head") and not allow_lm_head:
        raise RuntimeError(f"{scope_name} unexpectedly activated LLaVA-OneVision lm_head parameters.")

    if adapter_expected:
        actual_hits = list(report.get("actual_lora_target_modules", []))
        unexpected_hits = list(report.get("unexpected_lora_modules", []))
        if not actual_hits:
            raise RuntimeError(f"{scope_name} did not inject any LoRA target modules.")
        if unexpected_hits:
            raise RuntimeError(
                f"{scope_name} injected LoRA into unexpected modules: {unexpected_hits[:16]}"
                f"{' ...' if len(unexpected_hits) > 16 else ''}"
            )
