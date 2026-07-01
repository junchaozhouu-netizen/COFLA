from __future__ import annotations

import os
from contextlib import nullcontext
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, CLIPTextModel, CLIPVisionModel

try:
    from torch.nn.utils.stateless import functional_call as _functional_call
except Exception:  # pragma: no cover - older/newer torch fallback
    from torch.func import functional_call as _functional_call


def _require_existing_dir(path: str, *, name: str) -> str:
    if not path:
        raise ValueError(f"{name} path is empty.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} path does not exist: {path}")
    return path


def _capture_rng_state(device: torch.device) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def _restore_rng_state(state: Optional[Dict[str, torch.Tensor]], device: torch.device) -> None:
    if not state:
        return
    cpu_state = state.get("cpu")
    if cpu_state is not None:
        torch.set_rng_state(cpu_state)
    cuda_state = state.get("cuda")
    if cuda_state is not None and device.type == "cuda":
        torch.cuda.set_rng_state(cuda_state, device)


def _prefer_pooled_or_cls(outputs) -> torch.Tensor:
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is None:
        pooled = getattr(outputs, "pooled_output", None)
    if pooled is not None:
        return pooled

    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        raise RuntimeError("Encoder output does not contain pooled_output/pooler_output/last_hidden_state.")
    return hidden[:, 0]


def _mean_pool_hidden(hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if attention_mask is None:
        return hidden[:, 0]
    mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype, device=hidden.device)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden * mask).sum(dim=1) / denom



def _normalize_branch_tuning_mode(value: str | None) -> str:
    mode = str(value or "frozen").strip().lower().replace("-", "_")
    aliases = {
        "freeze": "frozen",
        "frozen_branch": "frozen",
        "frozen_branches": "frozen",
        "branch_lora": "branch_lora_full_fusion",
        "branch_lora_fullfusion": "branch_lora_full_fusion",
        "unimodal_lora_full_fusion": "branch_lora_full_fusion",
        "branch_lora_full_fusion": "branch_lora_full_fusion",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"frozen", "branch_lora_full_fusion"}:
        raise ValueError(
            f"Unsupported branch_tuning_mode={value!r}. "
            "Use 'frozen' or 'branch_lora_full_fusion'."
        )
    return mode


def _split_target_modules(value: str | Sequence[str] | None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "auto":
            return []
        return [item.strip() for item in text.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _auto_lora_targets(encoder_kind: str) -> List[str]:
    kind = str(encoder_kind).strip().lower()
    if kind in {"clip_vision", "clip_text"}:
        return ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    if kind == "roberta":
        return ["query", "key", "value", "dense"]
    return ["query", "key", "value", "q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "dense"]


def _attach_lora_to_encoder(
    module: nn.Module,
    *,
    encoder_kind: str,
    target_modules: str | Sequence[str] | None,
    r: int,
    alpha: int,
    dropout: float,
) -> Tuple[nn.Module, List[str]]:
    resolved_targets = _split_target_modules(target_modules) or _auto_lora_targets(encoder_kind)
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(
            "Branch LoRA requires the `peft` package. Install it with `pip install peft` "
            "or use --branch_tuning_mode frozen."
        ) from exc

    for param in module.parameters():
        param.requires_grad_(False)

    # Do not set a task_type here.  The generic PEFT wrapper preserves arbitrary
    # encoder forward signatures such as CLIPVisionModel(pixel_values=...) and
    # CLIPTextModel/RobertaModel(input_ids=..., attention_mask=...).
    lora_cfg = LoraConfig(
        r=int(r),
        lora_alpha=int(alpha),
        target_modules=resolved_targets,
        lora_dropout=float(dropout),
        bias="none",
    )
    wrapped = get_peft_model(module, lora_cfg)
    trainable_count = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
    if trainable_count <= 0:
        raise RuntimeError(
            f"Branch LoRA did not create any trainable parameters for {encoder_kind}. "
            f"Tried target_modules={resolved_targets}."
        )
    return wrapped, resolved_targets

@dataclass
class ControlledLateFusionOutput:
    z_v: torch.Tensor
    z_t: torch.Tensor
    z_f: torch.Tensor
    logits: torch.Tensor
    loss: Optional[torch.Tensor]


class ConcatMLPFusion(nn.Module):
    def __init__(self, vision_dim: int, text_dim: int, fusion_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.visual_proj = nn.Linear(vision_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(self, z_v: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        v = self.visual_proj(z_v)
        t = self.text_proj(z_t)
        fused = self.mlp(torch.cat([v, t], dim=-1))
        return self.norm(fused)


class GatedMLPFusion(nn.Module):
    def __init__(self, vision_dim: int, text_dim: int, fusion_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.visual_proj = nn.Linear(vision_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.gate = nn.Linear(vision_dim + text_dim, fusion_dim)
        self.post_mlp = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(self, z_v: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        v = self.visual_proj(z_v)
        t = self.text_proj(z_t)
        gate = torch.sigmoid(self.gate(torch.cat([z_v, z_t], dim=-1)))
        fused = gate * v + (1.0 - gate) * t
        fused = fused + self.post_mlp(fused)
        return self.norm(fused)


class ControlledLateFusionModel(nn.Module):
    # Keep the original Hateful Memes and MM-IMDb task branches unchanged.
    # NLVR2 and ScienceQA are explicit, isolated task types used only by the
    # dedicated perturbation-evaluation entrypoint.
    SUPPORTED_DATASETS = {"hateful_memes", "mmimdb", "nlvr2", "scienceqa"}
    SUPPORTED_TEXT_ENCODERS = {"clip_text", "roberta"}
    SUPPORTED_FUSIONS = {"concat_mlp", "gated_mlp"}

    def __init__(
        self,
        *,
        dataset_type: str,
        text_encoder_type: str,
        fusion_type: str,
        clip_path: str,
        roberta_path: Optional[str] = None,
        fusion_dim: int = 512,
        dropout: float = 0.1,
        num_labels: Optional[int] = None,
        freeze_branches: bool = True,
        branch_tuning_mode: str = "frozen",
        branch_lora_r: int = 8,
        branch_lora_alpha: int = 16,
        branch_lora_dropout: float = 0.05,
        branch_lora_target_modules: str = "auto",
        local_files_only: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.dataset_type = str(dataset_type).strip().lower()
        self.text_encoder_type = str(text_encoder_type).strip().lower()
        self.fusion_type = str(fusion_type).strip().lower()
        # Backward-compatible behavior: freeze_branches=True maps to the old frozen-branch
        # controlled setting.  Passing branch_tuning_mode="branch_lora_full_fusion" freezes
        # the original unimodal branch weights, injects LoRA adapters into the branches,
        # and trains those adapters together with the full fusion block and classifier.
        if bool(freeze_branches) and str(branch_tuning_mode).strip().lower() in {"", "none"}:
            branch_tuning_mode = "frozen"
        self.branch_tuning_mode = _normalize_branch_tuning_mode(branch_tuning_mode)
        self.freeze_branches = self.branch_tuning_mode == "frozen"
        self.branch_lora_r = int(branch_lora_r)
        self.branch_lora_alpha = int(branch_lora_alpha)
        self.branch_lora_dropout = float(branch_lora_dropout)
        self.branch_lora_target_modules = str(branch_lora_target_modules or "auto")
        self.branch_lora_target_modules_resolved: Dict[str, List[str]] = {}
        self.geometry_eps = float(eps)

        if self.dataset_type not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")
        if self.text_encoder_type not in self.SUPPORTED_TEXT_ENCODERS:
            raise ValueError(f"Unsupported text_encoder_type: {text_encoder_type}")
        if self.fusion_type not in self.SUPPORTED_FUSIONS:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        clip_path = _require_existing_dir(clip_path, name="CLIP model")
        if self.text_encoder_type == "roberta":
            _require_existing_dir(roberta_path or "", name="RoBERTa model")

        self.visual_encoder = CLIPVisionModel.from_pretrained(
            clip_path,
            local_files_only=local_files_only,
        )
        if self.text_encoder_type == "clip_text":
            self.text_encoder = CLIPTextModel.from_pretrained(
                clip_path,
                local_files_only=local_files_only,
            )
        else:
            self.text_encoder = AutoModel.from_pretrained(
                roberta_path,
                local_files_only=local_files_only,
            )

        vision_dim = int(getattr(self.visual_encoder.config, "hidden_size"))
        text_dim = int(getattr(self.text_encoder.config, "hidden_size"))
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.fusion_dim = int(fusion_dim)

        if self.branch_tuning_mode == "branch_lora_full_fusion":
            self._setup_branch_lora_modules()

        if self.fusion_type == "concat_mlp":
            self.fusion_module = ConcatMLPFusion(
                vision_dim=vision_dim,
                text_dim=text_dim,
                fusion_dim=fusion_dim,
                dropout=dropout,
            )
        else:
            self.fusion_module = GatedMLPFusion(
                vision_dim=vision_dim,
                text_dim=text_dim,
                fusion_dim=fusion_dim,
                dropout=dropout,
            )

        if self.dataset_type == "hateful_memes":
            # Original Hateful Memes behavior: one binary logit.
            self.num_labels = 1
        elif self.dataset_type == "mmimdb":
            # Original MM-IMDb behavior is intentionally kept unchanged.
            if num_labels is None or int(num_labels) <= 0:
                raise ValueError("MM-IMDb requires a positive num_labels value.")
            self.num_labels = int(num_labels)
        elif self.dataset_type == "nlvr2":
            # Dedicated NLVR2 branch: binary classification with one logit.
            self.num_labels = 1
        elif self.dataset_type == "scienceqa":
            # Dedicated ScienceQA branch: masked single-choice classification.
            if num_labels is None or int(num_labels) <= 0:
                raise ValueError("ScienceQA requires a positive num_labels value.")
            self.num_labels = int(num_labels)
        else:  # guarded by SUPPORTED_DATASETS; retained for defensive clarity
            raise ValueError(f"Unsupported dataset_type: {self.dataset_type}")
        self.classifier = nn.Linear(self.fusion_dim, self.num_labels)

        if self.freeze_branches:
            self._freeze_branch_modules()

    def branch_parameters_are_trainable(self) -> bool:
        return any(param.requires_grad for param in self.visual_encoder.parameters()) or any(
            param.requires_grad for param in self.text_encoder.parameters()
        )

    def _setup_branch_lora_modules(self) -> None:
        # LoRA is attached independently to the visual and textual branches.  The base
        # branch weights remain frozen; only LoRA adapter parameters are trainable.
        self.visual_encoder, visual_targets = _attach_lora_to_encoder(
            self.visual_encoder,
            encoder_kind="clip_vision",
            target_modules=self.branch_lora_target_modules,
            r=self.branch_lora_r,
            alpha=self.branch_lora_alpha,
            dropout=self.branch_lora_dropout,
        )
        text_kind = "clip_text" if self.text_encoder_type == "clip_text" else "roberta"
        self.text_encoder, text_targets = _attach_lora_to_encoder(
            self.text_encoder,
            encoder_kind=text_kind,
            target_modules=self.branch_lora_target_modules,
            r=self.branch_lora_r,
            alpha=self.branch_lora_alpha,
            dropout=self.branch_lora_dropout,
        )
        self.branch_lora_target_modules_resolved = {
            "visual_encoder": list(visual_targets),
            "text_encoder": list(text_targets),
        }

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        if self.freeze_branches:
            self.visual_encoder.eval()
            self.text_encoder.eval()
        return self

    def _freeze_branch_modules(self) -> None:
        for module in [self.visual_encoder, self.text_encoder]:
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)

    def get_fusion_named_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        return list(self.fusion_module.named_parameters())

    def get_classifier_named_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        return list(self.classifier.named_parameters())

    def get_fusion_head_named_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        named: List[Tuple[str, nn.Parameter]] = []
        named.extend((f"fusion_module.{name}", param) for name, param in self.fusion_module.named_parameters() if param.requires_grad)
        named.extend((f"classifier.{name}", param) for name, param in self.classifier.named_parameters() if param.requires_grad)
        return named

    def get_branch_lora_named_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        named: List[Tuple[str, nn.Parameter]] = []
        if self.branch_tuning_mode != "branch_lora_full_fusion":
            return named
        for prefix, module in [("visual_encoder", self.visual_encoder), ("text_encoder", self.text_encoder)]:
            for name, param in module.named_parameters():
                if param.requires_grad:
                    named.append((f"{prefix}.{name}", param))
        return named

    def get_trainable_named_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        named: List[Tuple[str, nn.Parameter]] = []
        named.extend(self.get_branch_lora_named_parameters())
        named.extend(self.get_fusion_head_named_parameters())
        return named

    def get_fusion_trainable_named_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        # Compatibility alias used by the generic method classes.  In the
        # controlled branch-LoRA/full-fusion setting, the fusion path is the
        # explicit fusion module plus task head.
        return self.get_fusion_head_named_parameters()

    def _apply_param_perturb(self, perturb_dict: Dict[str, torch.Tensor], sign: int) -> None:
        """In-place perturbation helper for branch-side SAM-style baselines.

        The controlled model usually evaluates fusion perturbations with
        functional_call, but branch-side LoRA perturbations must be applied to
        the encoder adapters before recomputing branch activations.  This helper
        only touches trainable parameters whose full names appear in
        perturb_dict and restores them when called with sign=-1.
        """
        if not perturb_dict:
            return
        params = dict(self.get_trainable_named_parameters())
        with torch.no_grad():
            for name, delta in perturb_dict.items():
                param = params.get(name)
                if param is None:
                    continue
                param.add_(delta.to(device=param.device, dtype=param.dtype), alpha=float(sign))

    def get_fusion_parameters(self) -> List[nn.Parameter]:
        return [param for _, param in self.get_fusion_named_parameters()]

    def get_fusion_head_parameters(self) -> List[nn.Parameter]:
        return [param for _, param in self.get_fusion_head_named_parameters()]

    def get_branch_lora_parameters(self) -> List[nn.Parameter]:
        return [param for _, param in self.get_branch_lora_named_parameters()]

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        return [param for _, param in self.get_trainable_named_parameters()]

    def build_optimizer_param_groups(
        self,
        *,
        learning_rate: float,
        weight_decay: float,
        branch_lora_lr: Optional[float] = None,
    ) -> List[Dict[str, object]]:
        fusion_head_params = self.get_fusion_head_parameters()
        branch_params = self.get_branch_lora_parameters()
        groups: List[Dict[str, object]] = []
        if fusion_head_params:
            groups.append(
                {
                    "name": "fusion_head",
                    "params": fusion_head_params,
                    "lr": float(learning_rate),
                    "weight_decay": float(weight_decay),
                }
            )
        if branch_params:
            groups.append(
                {
                    "name": "branch_lora",
                    "params": branch_params,
                    "lr": float(learning_rate if branch_lora_lr is None else branch_lora_lr),
                    "weight_decay": float(weight_decay),
                }
            )
        return groups

    def describe_trainable_state(self, preview_limit: int = 20) -> Dict[str, object]:
        named = self.get_trainable_named_parameters()
        preview = [name for name, _ in named[: max(0, int(preview_limit))]]
        total = sum(param.numel() for _, param in self.named_parameters())
        trainable = sum(param.numel() for _, param in named)
        return {
            "dataset_type": self.dataset_type,
            "text_encoder_type": self.text_encoder_type,
            "fusion_type": self.fusion_type,
            "vision_dim": self.vision_dim,
            "text_dim": self.text_dim,
            "fusion_dim": self.fusion_dim,
            "num_labels": self.num_labels,
            "branch_tuning_mode": self.branch_tuning_mode,
            "branch_lora_r": self.branch_lora_r,
            "branch_lora_alpha": self.branch_lora_alpha,
            "branch_lora_dropout": self.branch_lora_dropout,
            "branch_lora_target_modules": self.branch_lora_target_modules_resolved,
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "branch_lora_trainable_parameters": int(sum(p.numel() for _, p in self.get_branch_lora_named_parameters())),
            "fusion_head_trainable_parameters": int(sum(p.numel() for _, p in self.get_fusion_head_named_parameters())),
            "trainable_parameter_names_preview": preview,
        }

    def export_trainable_state(self, metadata: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        trainable_state = {
            name: param.detach().cpu()
            for name, param in self.get_trainable_named_parameters()
        }
        return {
            "fusion_module": self.fusion_module.state_dict(),
            "classifier": self.classifier.state_dict(),
            "trainable_parameter_state": trainable_state,
            "metadata": metadata or {},
        }

    def load_trainable_state(self, state: Dict[str, object], strict: bool = True) -> Dict[str, object]:
        fusion_result = self.fusion_module.load_state_dict(state.get("fusion_module", {}), strict=False)
        classifier_result = self.classifier.load_state_dict(state.get("classifier", {}), strict=False)
        loaded_trainable = 0
        missing_trainable: List[str] = []
        unexpected_trainable: List[str] = []
        trainable_state = state.get("trainable_parameter_state", {})
        if isinstance(trainable_state, dict):
            current = dict(self.get_trainable_named_parameters())
            for name, value in trainable_state.items():
                param = current.get(name)
                if param is None:
                    unexpected_trainable.append(name)
                    continue
                tensor = value.to(device=param.device, dtype=param.dtype) if torch.is_tensor(value) else torch.as_tensor(value, device=param.device, dtype=param.dtype)
                if tuple(tensor.shape) != tuple(param.shape):
                    unexpected_trainable.append(name)
                    continue
                with torch.no_grad():
                    param.copy_(tensor)
                loaded_trainable += 1
            for name in current:
                if name not in trainable_state:
                    missing_trainable.append(name)
        return {
            "fusion_missing": list(fusion_result.missing_keys),
            "fusion_unexpected": list(fusion_result.unexpected_keys),
            "classifier_missing": list(classifier_result.missing_keys),
            "classifier_unexpected": list(classifier_result.unexpected_keys),
            "loaded_trainable_parameters": loaded_trainable,
            "missing_trainable_parameter_names": missing_trainable[:50],
            "unexpected_trainable_parameter_names": unexpected_trainable[:50],
            "metadata": state.get("metadata", {}),
        }

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        ctx = torch.no_grad() if self.freeze_branches else nullcontext()
        with ctx:
            outputs = self.visual_encoder(pixel_values=pixel_values)
        return _prefer_pooled_or_cls(outputs)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        ctx = torch.no_grad() if self.freeze_branches else nullcontext()
        with ctx:
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        if self.text_encoder_type == "roberta":
            return _mean_pool_hidden(outputs.last_hidden_state, attention_mask)
        return _prefer_pooled_or_cls(outputs)

    def extract_branch_features(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if "pixel_values" not in batch:
            raise KeyError("Batch is missing 'pixel_values'.")
        if "input_ids" not in batch:
            raise KeyError("Batch is missing 'input_ids'.")
        z_v = self.encode_image(batch["pixel_values"])
        z_t = self.encode_text(batch["input_ids"], batch.get("attention_mask"))
        return z_v, z_t

    def _prepare_branch_activation(self, z: torch.Tensor, requires_grad: bool) -> torch.Tensor:
        if self.freeze_branches:
            out = z.detach()
            if requires_grad:
                out.requires_grad_(True)
            return out
        # In Branch-LoRA mode, keep the computation graph so the task loss can update
        # branch LoRA adapters.  z is also a valid differentiable activation for FCF probes.
        if requires_grad and not z.requires_grad:
            z.requires_grad_(True)
        return z

    def detach_branch_activation_if_frozen(self, z: torch.Tensor) -> torch.Tensor:
        return z.detach() if self.freeze_branches else z

    def prepare_branch_activations(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        requires_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_v, z_t = self.extract_branch_features(batch)
        return self._prepare_branch_activation(z_v, requires_grad), self._prepare_branch_activation(z_t, requires_grad)

    @staticmethod
    def _build_override_map(
        named_params: Sequence[Tuple[str, nn.Parameter]],
        perturb: Optional[Dict[str, torch.Tensor]],
    ) -> Optional[OrderedDict[str, torch.Tensor]]:
        if perturb is None:
            return None
        mapped: OrderedDict[str, torch.Tensor] = OrderedDict()
        for name, param in named_params:
            delta = perturb.get(name)
            if delta is None:
                mapped[name] = param
            else:
                mapped[name] = param + delta.to(device=param.device, dtype=param.dtype)
        return mapped

    def _fusion_forward(
        self,
        z_v: torch.Tensor,
        z_t: torch.Tensor,
        fusion_param_overrides: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if fusion_param_overrides is None:
            return self.fusion_module(z_v, z_t)
        override_map = self._build_override_map(self.get_fusion_named_parameters(), fusion_param_overrides)
        return _functional_call(self.fusion_module, override_map, (z_v, z_t))

    def _classifier_forward(
        self,
        z_f: torch.Tensor,
        classifier_param_overrides: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if classifier_param_overrides is None:
            return self.classifier(z_f)
        override_map = self._build_override_map(self.get_classifier_named_parameters(), classifier_param_overrides)
        return _functional_call(self.classifier, override_map, (z_f,))

    def compute_task_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        choice_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.dataset_type == "hateful_memes":
            # Original Hateful Memes loss path: unchanged.
            logits_flat = logits.squeeze(-1)
            labels_flat = labels.to(device=logits.device, dtype=logits.dtype).view(-1)
            return F.binary_cross_entropy_with_logits(logits_flat, labels_flat)

        if self.dataset_type == "mmimdb":
            # Original MM-IMDb multi-label BCE path: unchanged.
            labels_float = labels.to(device=logits.device, dtype=logits.dtype)
            return F.binary_cross_entropy_with_logits(logits, labels_float)

        if self.dataset_type == "nlvr2":
            # NLVR2 is binary classification, kept separate from Hateful Memes
            # so future task-specific changes cannot alter the original branch.
            logits_flat = logits.squeeze(-1)
            labels_flat = labels.to(device=logits.device, dtype=logits.dtype).view(-1)
            return F.binary_cross_entropy_with_logits(logits_flat, labels_flat)

        if self.dataset_type == "scienceqa":
            # ScienceQA is a single-choice task. Invalid padded choices must not
            # participate in either the clean task loss or any perturbation loss.
            targets = labels.to(device=logits.device)
            if targets.ndim > 1:
                # Backward-compatible acceptance of legacy one-hot batches; the
                # dedicated collator now emits integer class indices.
                targets = targets.argmax(dim=-1)
            targets = targets.to(dtype=torch.long).view(-1)

            masked_logits = logits
            if choice_mask is not None:
                valid_choices = choice_mask.to(device=logits.device, dtype=torch.bool)
                if valid_choices.shape != logits.shape:
                    raise ValueError(
                        "ScienceQA choice_mask shape must match logits: "
                        f"mask={tuple(valid_choices.shape)}, logits={tuple(logits.shape)}"
                    )
                masked_logits = logits.masked_fill(
                    ~valid_choices,
                    torch.finfo(logits.dtype).min,
                )
            # Compute CE in fp32 for stable perturbation differences while
            # preserving gradients back to the original logits.
            return F.cross_entropy(masked_logits.float(), targets)

        raise ValueError(f"Unsupported dataset_type: {self.dataset_type}")

    def forward_from_activations(
        self,
        z_v: torch.Tensor,
        z_t: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
        choice_mask: Optional[torch.Tensor] = None,
        fusion_param_overrides: Optional[Dict[str, torch.Tensor]] = None,
        classifier_param_overrides: Optional[Dict[str, torch.Tensor]] = None,
    ) -> ControlledLateFusionOutput:
        z_f = self._fusion_forward(z_v, z_t, fusion_param_overrides=fusion_param_overrides)
        logits = self._classifier_forward(z_f, classifier_param_overrides=classifier_param_overrides)
        loss = None if labels is None else self.compute_task_loss(
            logits,
            labels,
            choice_mask=choice_mask,
        )
        return ControlledLateFusionOutput(z_v=z_v, z_t=z_t, z_f=z_f, logits=logits, loss=loss)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        branch_requires_grad: bool = False,
        labels: Optional[torch.Tensor] = None,
        fusion_param_overrides: Optional[Dict[str, torch.Tensor]] = None,
        classifier_param_overrides: Optional[Dict[str, torch.Tensor]] = None,
        z_v_override: Optional[torch.Tensor] = None,
        z_t_override: Optional[torch.Tensor] = None,
    ) -> ControlledLateFusionOutput:
        if z_v_override is None or z_t_override is None:
            z_v, z_t = self.prepare_branch_activations(batch, requires_grad=branch_requires_grad)
        else:
            z_v, z_t = z_v_override, z_t_override
        task_labels = batch.get("labels") if labels is None else labels
        return self.forward_from_activations(
            z_v,
            z_t,
            labels=task_labels,
            choice_mask=batch.get("choice_mask") if self.dataset_type == "scienceqa" else None,
            fusion_param_overrides=fusion_param_overrides,
            classifier_param_overrides=classifier_param_overrides,
        )

    def predict_from_logits(self, logits: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = torch.sigmoid(logits)
        if self.dataset_type in {"hateful_memes", "nlvr2"}:
            probs = probs.squeeze(-1)
            preds = (probs >= threshold).long()
            return probs, preds
        preds = (probs >= threshold).long()
        return probs, preds

    def compute_geometry_probe(self, batch: Dict[str, torch.Tensor]) -> Dict[str, object]:
        z_v, z_t = self.prepare_branch_activations(batch, requires_grad=True)
        rng_state = _capture_rng_state(z_v.device) if self.training else None
        outputs = self.forward_from_activations(
            z_v,
            z_t,
            labels=batch["labels"],
            choice_mask=batch.get("choice_mask") if self.dataset_type == "scienceqa" else None,
        )
        if outputs.loss is None:
            raise RuntimeError("Geometry probe requires labels to compute the task loss.")
        return {
            "z_v": z_v,
            "z_t": z_t,
            "z_f": outputs.z_f,
            "logits": outputs.logits,
            "loss": outputs.loss,
            "rng_state": rng_state,
        }

    def _build_activation_delta(
        self,
        grad: Optional[torch.Tensor],
        reference: torch.Tensor,
        rho: float,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if grad is None:
            return torch.zeros_like(reference), torch.tensor(0.0, device=device)
        grad_detached = grad.detach()
        norm = grad_detached.float().norm(p=2)
        if float(norm.detach().cpu().item()) == 0.0 or rho <= 0.0:
            return torch.zeros_like(grad_detached), norm.to(device=device)
        delta = rho * grad_detached / (norm.to(device=grad_detached.device) + self.geometry_eps)
        return delta.to(dtype=grad_detached.dtype), norm.to(device=device)

    def _build_fusion_param_delta(
        self,
        grads: Sequence[Optional[torch.Tensor]],
        rho_f: float,
        device: torch.device,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        fusion_named = self.get_fusion_named_parameters()
        sq = 0.0
        for grad in grads:
            if grad is None:
                continue
            sq += grad.detach().float().pow(2).sum().item()
        norm = torch.tensor(sq ** 0.5, device=device, dtype=torch.float32)
        if rho_f <= 0.0 or float(norm.detach().cpu().item()) == 0.0:
            return {}, norm
        scale = float(rho_f) / (float(norm.detach().cpu().item()) + self.geometry_eps)
        delta: Dict[str, torch.Tensor] = {}
        for (name, param), grad in zip(fusion_named, grads):
            if grad is None:
                continue
            delta[name] = grad.detach().to(device=param.device, dtype=param.dtype) * scale
        return delta, norm

    def _compute_linear_geometry_terms(
        self,
        batch: Dict[str, torch.Tensor],
        probe: Dict[str, object],
        *,
        rho_v: float,
        rho_t: float,
        rho_f: float,
        retain_graph: bool,
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        z_v = probe["z_v"]
        z_t = probe["z_t"]
        loss0 = probe["loss"]
        device = z_v.device

        fusion_named = self.get_fusion_named_parameters()
        grad_inputs: List[torch.Tensor] = [z_v, z_t]
        grad_inputs.extend(param for _, param in fusion_named)
        grads = torch.autograd.grad(
            outputs=loss0,
            inputs=grad_inputs,
            retain_graph=retain_graph,
            create_graph=False,
            allow_unused=True,
        )
        grad_zv = grads[0]
        grad_zt = grads[1]
        fusion_grads = grads[2:]

        delta_v, gv_norm = self._build_activation_delta(grad_zv, z_v, rho_v, device)
        delta_t, gt_norm = self._build_activation_delta(grad_zt, z_t, rho_t, device)
        delta_f, gf_norm = self._build_fusion_param_delta(fusion_grads, rho_f, device)
        branch_linear_proxy = 0.5 * (
            float(rho_v) * gv_norm.to(device=device, dtype=loss0.dtype)
            + float(rho_t) * gt_norm.to(device=device, dtype=loss0.dtype)
        )
        return {
            "delta_v": delta_v,
            "delta_t": delta_t,
            "delta_f": delta_f,
            "gv_norm": gv_norm.to(device=device, dtype=loss0.dtype),
            "gt_norm": gt_norm.to(device=device, dtype=loss0.dtype),
            "gf_norm": gf_norm.to(device=device, dtype=loss0.dtype),
            "branch_linear_proxy": branch_linear_proxy,
        }

    def compute_linear_geometry_stats(
        self,
        batch: Dict[str, torch.Tensor],
        rho_v: float,
        rho_t: float,
        *,
        retain_graph: bool = False,
    ) -> Dict[str, torch.Tensor]:
        probe = self.compute_geometry_probe(batch)
        terms = self._compute_linear_geometry_terms(
            batch,
            probe,
            rho_v=rho_v,
            rho_t=rho_t,
            rho_f=0.0,
            retain_graph=retain_graph,
        )
        return {
            "loss": probe["loss"],
            "gv_norm": terms["gv_norm"],
            "gt_norm": terms["gt_norm"],
            "gf_norm": terms["gf_norm"],
            "branch_linear_proxy": terms["branch_linear_proxy"],
        }


    def compute_fast_fcf_metrics(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        rho_v: float,
        rho_t: float,
        rho_f: float,
        retain_graph: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Compute FAST-FCF geometry metrics from first-order clean-loss gradients.

        FAST-FCF is an evaluation metric, not an optimizer. It approximates the
        exact perturbation-based FCF by using the clean task-loss gradient norms
        with respect to visual activations, text activations, and fusion-path
        parameters. It does not perturb parameters and does not perform updates.
        """
        probe = self.compute_geometry_probe(batch)
        loss0 = probe["loss"]
        device = loss0.device
        terms = self._compute_linear_geometry_terms(
            batch,
            probe,
            rho_v=rho_v,
            rho_t=rho_t,
            rho_f=rho_f,
            retain_graph=retain_graph,
        )
        gv_norm = terms["gv_norm"].to(device=device, dtype=loss0.dtype)
        gt_norm = terms["gt_norm"].to(device=device, dtype=loss0.dtype)
        gf_norm = terms["gf_norm"].to(device=device, dtype=loss0.dtype)
        eps = torch.tensor(self.geometry_eps, device=device, dtype=loss0.dtype)
        fast_s_v = float(rho_v) * gv_norm
        fast_s_t = float(rho_t) * gt_norm
        fast_s_branch = 0.5 * (fast_s_v + fast_s_t)
        fast_s_f = float(rho_f) * gf_norm
        fast_fcf = (fast_s_f + eps) / (fast_s_branch + eps)
        fast_rfcf = torch.log(fast_fcf)
        return {
            "loss": loss0,
            "fast_s_v": fast_s_v,
            "fast_s_t": fast_s_t,
            "fast_s_branch": fast_s_branch,
            "fast_s_f": fast_s_f,
            "fast_fcf": fast_fcf,
            "fast_rfcf": fast_rfcf,
            "fast_grad_zv_norm": gv_norm,
            "fast_grad_zt_norm": gt_norm,
            "fast_grad_fusion_norm": gf_norm,
        }

    def compute_geometry_metrics_from_probe(
        self,
        batch: Dict[str, torch.Tensor],
        probe: Dict[str, object],
        *,
        rho_v: float,
        rho_t: float,
        rho_f: float,
        retain_graph: bool = True,
    ) -> Dict[str, torch.Tensor]:
        z_v = probe["z_v"]
        z_t = probe["z_t"]
        loss0 = probe["loss"]
        rng_state = probe.get("rng_state")
        device = loss0.device
        terms = self._compute_linear_geometry_terms(
            batch,
            probe,
            rho_v=rho_v,
            rho_t=rho_t,
            rho_f=rho_f,
            retain_graph=retain_graph,
        )
        delta_v = terms["delta_v"]
        delta_t = terms["delta_t"]
        delta_f = terms["delta_f"]
        gv_norm = terms["gv_norm"]
        gt_norm = terms["gt_norm"]
        gf_norm = terms["gf_norm"]

        if self.training and rng_state is not None:
            _restore_rng_state(rng_state, device)
        scienceqa_choice_mask = batch.get("choice_mask") if self.dataset_type == "scienceqa" else None
        loss_v = self.forward_from_activations(
            z_v + delta_v,
            z_t,
            labels=batch["labels"],
            choice_mask=scienceqa_choice_mask,
        ).loss
        if self.training and rng_state is not None:
            _restore_rng_state(rng_state, device)
        loss_t = self.forward_from_activations(
            z_v,
            z_t + delta_t,
            labels=batch["labels"],
            choice_mask=scienceqa_choice_mask,
        ).loss
        if self.training and rng_state is not None:
            _restore_rng_state(rng_state, device)
        loss_f = self.forward_from_activations(
            z_v,
            z_t,
            labels=batch["labels"],
            choice_mask=scienceqa_choice_mask,
            fusion_param_overrides=delta_f if delta_f else None,
        ).loss
        if loss_v is None or loss_t is None or loss_f is None:
            raise RuntimeError("Geometry evaluation failed to produce perturbed losses.")

        # Clamp tiny negative values caused by finite precision around the base loss.
        s_v = (loss_v - loss0).clamp_min(0.0)
        s_t = (loss_t - loss0).clamp_min(0.0)
        s_f = (loss_f - loss0).clamp_min(0.0)
        s_branch = 0.5 * (s_v + s_t)
        eps = torch.tensor(self.geometry_eps, device=device, dtype=loss0.dtype)
        fcf = (s_f + eps) / (s_branch + eps)
        rfcf = torch.log(fcf)

        branch_linear = 0.5 * (float(rho_v) * gv_norm.to(device=device, dtype=loss0.dtype) + float(rho_t) * gt_norm.to(device=device, dtype=loss0.dtype))
        r_lin_fcf = torch.log(
            (float(rho_f) * gf_norm.to(device=device, dtype=loss0.dtype) + eps) /
            (branch_linear + eps)
        )

        return {
            "loss": loss0,
            "loss_v_plus": loss_v,
            "loss_t_plus": loss_t,
            "loss_f_plus": loss_f,
            # Explicit exact-FCF fields for optional diagnostics.
            "exact_s_v": s_v,
            "exact_s_t": s_t,
            "exact_s_branch": s_branch,
            "exact_s_f": s_f,
            "exact_fcf": fcf,
            "exact_rfcf": rfcf,
            "exact_r_lin_fcf": r_lin_fcf,
            "exact_grad_zv_norm": gv_norm.to(device=device, dtype=loss0.dtype),
            "exact_grad_zt_norm": gt_norm.to(device=device, dtype=loss0.dtype),
            "exact_grad_fusion_norm": gf_norm.to(device=device, dtype=loss0.dtype),
            # Backward-compatible aliases used internally by the COFLA training step.
            "s_v": s_v,
            "s_t": s_t,
            "s_branch": s_branch,
            "s_f": s_f,
            "fcf": fcf,
            "rfcf": rfcf,
            "r_lin_fcf": r_lin_fcf,
            "grad_zv_norm": gv_norm.to(device=device, dtype=loss0.dtype),
            "grad_zt_norm": gt_norm.to(device=device, dtype=loss0.dtype),
            "grad_fusion_norm": gf_norm.to(device=device, dtype=loss0.dtype),
        }

    def compute_geometry_metrics(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        rho_v: float,
        rho_t: float,
        rho_f: float,
        retain_graph: bool = True,
    ) -> Dict[str, torch.Tensor]:
        probe = self.compute_geometry_probe(batch)
        return self.compute_geometry_metrics_from_probe(
            batch,
            probe,
            rho_v=rho_v,
            rho_t=rho_t,
            rho_f=rho_f,
            retain_graph=retain_graph,
        )
