from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class StepOutput:
    loss: torch.Tensor
    logs: Dict[str, float]


class BaseMethod:
    name = "base"

    def __init__(self, args):
        self.args = args

    def training_step(self, wrapper, batch) -> StepOutput:
        raise NotImplementedError

    def after_backward(self, wrapper) -> Dict[str, float]:
        return {}

    def validation_geometry(self, wrapper, batch) -> Dict[str, float]:
        with torch.enable_grad():
            probe = wrapper.compute_geometry_probe(batch)
            geom = wrapper.compute_geometry_metrics_from_probe(batch, probe)
        return {k: float(v.detach().cpu().item()) for k, v in geom.items() if torch.is_tensor(v) and v.dim() == 0}

    @staticmethod
    def _named_trainable_params(wrapper) -> List[Tuple[str, torch.nn.Parameter]]:
        return wrapper.get_fusion_trainable_named_parameters()

    @staticmethod
    def _named_branch_params(wrapper) -> List[Tuple[str, torch.nn.Parameter]]:
        if hasattr(wrapper, "get_branch_lora_named_parameters"):
            return list(wrapper.get_branch_lora_named_parameters())
        return []

    @staticmethod
    def _named_fusion_params(wrapper) -> List[Tuple[str, torch.nn.Parameter]]:
        if hasattr(wrapper, "get_fusion_trainable_named_parameters"):
            return list(wrapper.get_fusion_trainable_named_parameters())
        return []

    @staticmethod
    def _split_branch_named_params(
        named_params: List[Tuple[str, torch.nn.Parameter]],
    ) -> Tuple[List[Tuple[str, torch.nn.Parameter]], List[Tuple[str, torch.nn.Parameter]]]:
        visual: List[Tuple[str, torch.nn.Parameter]] = []
        text: List[Tuple[str, torch.nn.Parameter]] = []
        for name, param in named_params:
            lowered = str(name).lower()
            if (
                lowered.startswith("visual_encoder.")
                or "vision_tower" in lowered
                or "vision_model" in lowered
                or ".visual." in lowered
                or lowered.startswith("visual.")
                or "visual_branch" in lowered
                or "image_branch" in lowered
            ):
                visual.append((name, param))
            elif (
                lowered.startswith("text_encoder.")
                or "language_model" in lowered
                or ".model.layers" in lowered
                or lowered.startswith("model.layers")
                or "text_branch" in lowered
            ):
                text.append((name, param))
        return visual, text

    @classmethod
    def _grouped_branch_params(
        cls,
        wrapper,
    ) -> Tuple[List[Tuple[str, torch.nn.Parameter]], List[Tuple[str, torch.nn.Parameter]]]:
        """Return visual/text branch-LoRA groups using the wrapper scope when available."""
        if hasattr(wrapper, "get_branch_lora_grouped_named_parameters"):
            visual, text = wrapper.get_branch_lora_grouped_named_parameters()
            return list(visual), list(text)
        return cls._split_branch_named_params(cls._named_branch_params(wrapper))

    @staticmethod
    def _grad_map(
        named_params: List[Tuple[str, torch.nn.Parameter]],
        grads: Tuple[Optional[torch.Tensor], ...],
    ) -> Dict[str, torch.Tensor]:
        grad_map: Dict[str, torch.Tensor] = {}
        for (name, _), grad in zip(named_params, grads):
            if grad is None:
                continue
            grad_map[name] = grad.detach().float()
        return grad_map

    @staticmethod
    def _merge_grad_maps_sum(*maps: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        merged: Dict[str, torch.Tensor] = {}
        for grad_map in maps:
            for name, grad in grad_map.items():
                if grad is None:
                    continue
                if name in merged:
                    merged[name] = merged[name] + grad.detach().float()
                else:
                    merged[name] = grad.detach().float().clone()
        return merged

    @staticmethod
    def _grad_l2_norm(grad_map: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
        sq = 0.0
        for grad in grad_map.values():
            sq += grad.pow(2).sum().item()
        return torch.tensor(math.sqrt(sq), device=device)

    @classmethod
    def _build_param_perturb_from_grad_map(
        cls,
        named_params: List[Tuple[str, torch.nn.Parameter]],
        grad_map: Dict[str, torch.Tensor],
        radius: float,
        device: torch.device,
        eps: float = 1e-12,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        norm = cls._grad_l2_norm(grad_map, device=device)
        norm_value = float(norm.detach().cpu().item())
        if radius <= 0.0 or norm_value == 0.0:
            return {}, torch.tensor(0.0, device=device)

        scale = radius / (norm_value + eps)
        perturb = {
            name: grad_map[name].to(dtype=param.dtype, device=param.device) * scale
            for name, param in named_params
            if name in grad_map
        }
        return perturb, norm

    @staticmethod
    def _forward_training_loss(
        wrapper,
        batch,
        *,
        capture_layer0: bool = False,
        hidden_perturb: Optional[torch.Tensor] = None,
        param_perturb: Optional[Dict[str, torch.Tensor]] = None,
    ):
        if batch["dataset_name"] == "mmimdb":
            return wrapper.forward_task_head(
                batch,
                use_full_inputs=True,
                compute_loss=True,
                capture_layer0=capture_layer0,
                hidden_perturb=hidden_perturb,
                param_perturb=param_perturb,
            )
        return wrapper.forward_lm(
            batch,
            use_full_inputs=True,
            labels=batch["lm_labels"],
            capture_layer0=capture_layer0,
            hidden_perturb=hidden_perturb,
            param_perturb=param_perturb,
        )

    @staticmethod
    def _compute_per_sample_losses(batch, outputs) -> torch.Tensor:
        if batch["dataset_name"] == "mmimdb":
            logits = outputs.logits.float()
            labels = batch["labels"].to(logits.device, dtype=torch.float32)
            return F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(dim=-1)

        logits = outputs.logits.float()
        labels = batch["lm_labels"].to(logits.device)
        vocab_size = logits.size(-1)
        token_loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(labels)
        valid = (labels != -100).float()
        denom = valid.sum(dim=-1).clamp_min(1.0)
        return (token_loss * valid).sum(dim=-1) / denom

    @staticmethod
    def _select_topk_mean(losses: torch.Tensor, keep_ratio: float) -> Tuple[torch.Tensor, int]:
        batch_size = int(losses.size(0))
        if batch_size == 0:
            raise ValueError("Cannot select samples from an empty loss tensor.")
        keep_ratio = float(max(0.0, min(1.0, keep_ratio)))
        if keep_ratio >= 1.0 or batch_size == 1:
            return losses.mean(), batch_size
        keep_count = max(1, int(math.ceil(batch_size * keep_ratio)))
        topk = torch.topk(losses, k=keep_count, largest=True).values
        return topk.mean(), keep_count

    @staticmethod
    def _dot_and_norm_stats(
        named_params: List[Tuple[str, torch.nn.Parameter]],
        grad_a: Dict[str, torch.Tensor],
        grad_b: Dict[str, torch.Tensor],
    ) -> Tuple[float, float, float]:
        dot = 0.0
        norm_a_sq = 0.0
        norm_b_sq = 0.0
        for name, _ in named_params:
            ga = grad_a.get(name)
            gb = grad_b.get(name)
            if ga is not None:
                norm_a_sq += ga.pow(2).sum().item()
            if gb is not None:
                norm_b_sq += gb.pow(2).sum().item()
            if ga is not None and gb is not None:
                dot += (ga * gb).sum().item()
        return dot, norm_a_sq, norm_b_sq

    @classmethod
    def _cosine_between_grad_maps(
        cls,
        named_params: List[Tuple[str, torch.nn.Parameter]],
        grad_a: Dict[str, torch.Tensor],
        grad_b: Dict[str, torch.Tensor],
        device: torch.device,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        dot, norm_a_sq, norm_b_sq = cls._dot_and_norm_stats(named_params, grad_a, grad_b)
        if norm_a_sq <= 0.0 or norm_b_sq <= 0.0:
            return torch.tensor(0.0, device=device)
        return torch.tensor(dot / (math.sqrt(norm_a_sq) * math.sqrt(norm_b_sq) + eps), device=device)

    @classmethod
    def _project_conflicting_grad_maps(
        cls,
        named_params: List[Tuple[str, torch.nn.Parameter]],
        grad_a: Dict[str, torch.Tensor],
        grad_b: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], bool]:
        dot, norm_a_sq, norm_b_sq = cls._dot_and_norm_stats(named_params, grad_a, grad_b)
        if dot >= 0.0 or norm_a_sq <= 0.0 or norm_b_sq <= 0.0:
            return dict(grad_a), dict(grad_b), False

        proj_a: Dict[str, torch.Tensor] = {}
        proj_b: Dict[str, torch.Tensor] = {}
        scale_a = dot / (norm_b_sq + 1e-12)
        scale_b = dot / (norm_a_sq + 1e-12)
        for name, _ in named_params:
            ga = grad_a.get(name)
            gb = grad_b.get(name)
            if ga is not None:
                if gb is not None:
                    proj_a[name] = ga - gb * scale_a
                else:
                    proj_a[name] = ga
            if gb is not None:
                if ga is not None:
                    proj_b[name] = gb - ga * scale_b
                else:
                    proj_b[name] = gb
        return proj_a, proj_b, True

    @staticmethod
    def _float(value) -> float:
        if torch.is_tensor(value):
            return float(torch.nan_to_num(value.detach().float(), nan=0.0, posinf=0.0, neginf=0.0).cpu().item())
        return float(value)

    @staticmethod
    def _hidden_drop_perturb(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return -hidden.detach() * mask.to(device=hidden.device, dtype=hidden.dtype)

    def _capture_loss_hidden_masks(self, wrapper, batch):
        outputs, cache = self._forward_training_loss(wrapper, batch, capture_layer0=True)
        hidden = cache.get("hidden") if isinstance(cache, dict) else None
        if hidden is None:
            raise RuntimeError(
                f"{self.name} requires captured layer-0 hidden states for VLM modality-dropout proxy losses."
            )
        if not hasattr(wrapper, "_build_branch_masks"):
            raise RuntimeError(f"{self.name} requires wrapper._build_branch_masks for VLM branch proxy dropout.")
        image_mask, text_mask = wrapper._build_branch_masks(batch, hidden)
        return outputs, hidden, image_mask.detach(), text_mask.detach()

    def _compute_modality_dropout_losses_from_hidden(
        self,
        wrapper,
        batch,
        hidden: torch.Tensor,
        image_mask: torch.Tensor,
        text_mask: torch.Tensor,
    ):
        """Return visual-only, text-only, and branch-empty losses for VLM token-proxy branches.

        The VLM wrappers do not expose separate pre-fusion branch activations like the
        controlled late-fusion model. We therefore implement the same modality-dropout
        idea on the early mixed hidden states used by the geometry probe: zero the text
        proxy region for visual-only, zero the image proxy region for text-only, and zero
        both proxy regions for the empty reference.
        """
        drop_v = self._hidden_drop_perturb(hidden, image_mask)
        drop_t = self._hidden_drop_perturb(hidden, text_mask)
        out_v, _ = self._forward_training_loss(wrapper, batch, capture_layer0=False, hidden_perturb=drop_t)
        out_t, _ = self._forward_training_loss(wrapper, batch, capture_layer0=False, hidden_perturb=drop_v)
        out_empty, _ = self._forward_training_loss(
            wrapper,
            batch,
            capture_layer0=False,
            hidden_perturb=drop_v + drop_t,
        )
        if out_v.loss is None or out_t.loss is None or out_empty.loss is None:
            raise RuntimeError(f"{self.name} modality-dropout proxy losses failed to produce scalar losses.")
        return out_v.loss, out_t.loss, out_empty.loss

    @staticmethod
    def _two_modality_shapley_weights(
        *,
        loss_fuse: torch.Tensor,
        loss_v: torch.Tensor,
        loss_t: torch.Tensor,
        loss_empty: torch.Tensor,
        eps: float,
    ) -> Tuple[float, float, float, float]:
        lf = BaseMethod._float(loss_fuse)
        lv = BaseMethod._float(loss_v)
        lt = BaseMethod._float(loss_t)
        le = BaseMethod._float(loss_empty)
        phi_v = 0.5 * ((le - lv) + (lt - lf))
        phi_t = 0.5 * ((le - lt) + (lv - lf))
        pv = max(0.0, float(phi_v))
        pt = max(0.0, float(phi_t))
        denom = pv + pt + float(eps)
        if denom <= float(eps):
            return 0.5, 0.5, float(phi_v), float(phi_t)
        return pv / denom, pt / denom, float(phi_v), float(phi_t)

    def _install_grad_replacement(
        self,
        wrapper,
        grad_map: Dict[str, torch.Tensor],
        *,
        clean_grad_map: Optional[Dict[str, torch.Tensor]] = None,
        clear_missing: bool = False,
    ) -> int:
        """Replace the current micro-batch branch gradient without breaking accumulation.

        The training loop calls ``loss / gradient_accumulation_steps`` before
        ``backward()``, so ``param.grad`` already contains the previous accumulated
        gradients plus the clean branch gradient of the current micro-batch scaled
        by ``1 / gradient_accumulation_steps``. A direct assignment such as
        ``param.grad = corrected / accum`` would drop gradients accumulated from
        earlier micro-batches.

        When ``clean_grad_map`` is provided, we instead perform an in-place
        micro-batch replacement:

            grad <- grad - clean_grad / accum + corrected_grad / accum

        This preserves all previous micro-batch contributions while replacing only
        the current clean branch gradient. For backward compatibility, if the
        caller does not provide ``clean_grad_map``, we fall back to the old direct
        assignment behavior.
        """
        scale = 1.0 / max(1, int(getattr(self.args, "gradient_accumulation_steps", 1)))
        clean_grad_map = clean_grad_map or {}
        replaced = 0
        for name, param in self._named_branch_params(wrapper):
            corrected = grad_map.get(name)
            clean = clean_grad_map.get(name)

            if corrected is None:
                if clear_missing:
                    if clean is not None and param.grad is not None:
                        param.grad.add_(clean.to(device=param.device, dtype=param.dtype), alpha=-scale)
                    else:
                        param.grad = None
                continue

            corrected = corrected.to(device=param.device, dtype=param.dtype)
            if param.grad is None:
                param.grad = corrected.clone().mul_(scale)
            else:
                if clean is not None:
                    param.grad.add_(clean.to(device=param.device, dtype=param.dtype), alpha=-scale)
                param.grad.add_(corrected, alpha=scale)
            replaced += 1
        return replaced
