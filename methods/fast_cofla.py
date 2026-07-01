from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch

from .cofla import COFLAMethod, bounded_positive_fcf_gate
from .base_method import StepOutput


class FastCOFLAMethod(COFLAMethod):
    """COFLA-F: one-probe fusion-exact / branch-calibrated finite-step variant.

    The revised COFLA-F keeps the finite-step fusion sharpness
    S_f=[L_f^+-L_0]_+ as the projection objective, but replaces the two exact
    branch probes with a detached first-order branch calibration term.  This
    preserves S_f-guided projection, avoids differentiating through gradient-norm
    sharpness, and reduces COFLA-E's three perturbed probes to one fusion probe.
    """

    name = "fast_cofla"

    def __init__(self, args):
        super().__init__(args)
        self._fast_projection_stats: Dict[str, float] = {}

    @staticmethod
    def _safe_eps(args) -> float:
        return float(getattr(args, "fcf_eps", getattr(args, "eps", 1e-8)))

    @staticmethod
    def _masked_norm(x: Optional[torch.Tensor], mask: torch.Tensor, eps: float, *, device, dtype) -> torch.Tensor:
        if x is None or mask is None or not bool(mask.any()):
            return torch.zeros((), device=device, dtype=dtype)
        return torch.sqrt((x.detach().float() * mask.detach().float()).pow(2).sum() + float(eps)).to(device=device, dtype=dtype)

    @staticmethod
    def _grad_norm_from_list(grads: List[Optional[torch.Tensor]], eps: float, *, device, dtype) -> torch.Tensor:
        sq = 0.0
        found = False
        for grad in grads:
            if grad is None:
                continue
            found = True
            sq = sq + grad.detach().float().pow(2).sum()
        if not found:
            return torch.zeros((), device=device, dtype=dtype)
        return torch.sqrt(sq + float(eps)).to(device=device, dtype=dtype)

    @staticmethod
    def _grad_map_local(
        named_params: List[Tuple[str, torch.nn.Parameter]],
        grads: Tuple[Optional[torch.Tensor], ...] | List[Optional[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for (name, param), grad in zip(named_params, grads):
            if grad is None:
                continue
            out[name] = grad.detach().to(device=param.device, dtype=param.dtype)
        return out

    @staticmethod
    def _build_param_perturb_from_grad_map(
        named_params: List[Tuple[str, torch.nn.Parameter]],
        grad_map: Dict[str, torch.Tensor],
        radius: float,
        eps: float,
    ) -> Dict[str, torch.Tensor]:
        if float(radius) <= 0.0:
            return {}
        sq = 0.0
        for name, _ in named_params:
            grad = grad_map.get(name)
            if grad is not None:
                sq += float(grad.detach().float().pow(2).sum().cpu().item())
        norm = math.sqrt(max(sq, 0.0))
        if norm <= float(eps):
            return {}
        scale = float(radius) / (norm + float(eps))
        return {
            name: grad_map[name].detach().to(device=param.device, dtype=param.dtype) * scale
            for name, param in named_params
            if name in grad_map
        }

    def training_step(self, wrapper, batch) -> StepOutput:
        if not all(hasattr(wrapper, name) for name in ("compute_base_loss", "_build_branch_masks")):
            raise RuntimeError(
                "fast_cofla requires wrapper.compute_base_loss(..., capture_layer0=True) "
                "and wrapper._build_branch_masks(batch, hidden)."
            )

        # Clean loss and first-order directions.  create_graph=False is crucial:
        # gradients are used only to build detached finite-step perturbations and
        # branch calibration, never as differentiable objectives.
        outputs, cache = wrapper.compute_base_loss(batch, capture_layer0=True)
        base_loss = outputs.loss.float()
        hidden = cache.get("hidden")
        if hidden is None:
            raise RuntimeError("Failed to capture layer-0 hidden states for COFLA-F.")

        eps = self._safe_eps(self.args)
        device = base_loss.device
        dtype = base_loss.dtype

        fusion_named = self._named_fusion_params(wrapper)
        grad_inputs: List[torch.Tensor] = [hidden] + [param for _, param in fusion_named]
        grads = torch.autograd.grad(
            outputs=base_loss,
            inputs=grad_inputs,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        hidden_grad = grads[0]
        if hidden_grad is None:
            hidden_grad = torch.zeros_like(hidden)
        fusion_grads = list(grads[1:])

        image_mask, text_mask = wrapper._build_branch_masks(batch, hidden)
        # Keep the COFLA-F branch calibration on the same activation-norm
        # convention as the wrapper's COFLA-E / geometry-probe implementation.
        # Most VLM wrappers use token-normalized masked_frobenius_norm to avoid
        # sequence-length-dependent branch scales; fall back to a raw masked norm
        # only for wrappers that do not expose this helper.
        if hasattr(wrapper, "masked_frobenius_norm"):
            gv_norm = wrapper.masked_frobenius_norm(hidden_grad.detach(), mask=image_mask, eps=eps).to(device=device, dtype=dtype)
            gt_norm = wrapper.masked_frobenius_norm(hidden_grad.detach(), mask=text_mask, eps=eps).to(device=device, dtype=dtype)
        else:
            gv_norm = self._masked_norm(hidden_grad, image_mask, eps, device=device, dtype=dtype)
            gt_norm = self._masked_norm(hidden_grad, text_mask, eps, device=device, dtype=dtype)
        gf_norm = self._grad_norm_from_list(fusion_grads, eps, device=device, dtype=dtype)

        # Detached branch-side calibration denominator.
        fast_s_v = float(getattr(self.args, "rho_v", 0.0)) * gv_norm
        fast_s_t = float(getattr(self.args, "rho_t", 0.0)) * gt_norm
        if hasattr(wrapper, "_aggregate_branch_sharpness"):
            branch_calibration = wrapper._aggregate_branch_sharpness(fast_s_v, fast_s_t).detach()
        else:
            branch_calibration = (0.5 * (fast_s_v + fast_s_t)).detach()

        # One exact finite-step fusion probe: L_f^+ = L(theta_f + delta_f).
        fusion_grad_map = self._grad_map_local(fusion_named, fusion_grads)
        param_perturb = self._build_param_perturb_from_grad_map(
            fusion_named,
            fusion_grad_map,
            float(getattr(self.args, "rho_f", 0.0)),
            eps,
        )
        perturbed_outputs, _ = self._forward_training_loss(
            wrapper,
            batch,
            capture_layer0=False,
            hidden_perturb=None,
            param_perturb=param_perturb if param_perturb else None,
        )
        if perturbed_outputs.loss is None:
            raise RuntimeError("COFLA-F one-probe fusion forward did not return a loss.")
        loss_f_plus = perturbed_outputs.loss.float()
        s_f = (loss_f_plus - base_loss).clamp_min(0.0)

        eps_tensor = torch.tensor(float(eps), device=device, dtype=dtype)
        fcf = (s_f + eps_tensor) / (branch_calibration + eps_tensor)
        rfcf = torch.log(fcf.clamp_min(float(eps)))
        j_bf = torch.relu(rfcf)
        fcf_gate = bounded_positive_fcf_gate(rfcf)
        total_loss = (1.0 - fcf_gate) * base_loss + fcf_gate * loss_f_plus

        # Projection is applied only to the fusion side and is driven by the
        # finite-step objective J_BF^F, not by gradient-norm FAST-FCF.
        self._branch_correction_map = {}
        self._fusion_correction_map, stats_f = self._build_projection_for_objective(
            named_params=fusion_named,
            loss_task=total_loss,
            objective=j_bf,
            objective_value=self._float(j_bf),
        )
        self._alpha_star = float(stats_f["alpha_star"])
        self._projection_active = bool(self._fusion_correction_map)
        self._j_bf = self._float(j_bf)

        logs = {
            "train_loss": self._float(total_loss),
            "task_loss": self._float(base_loss),
            "loss_align": self._float(j_bf),
            "loss_fuse": self._float(s_f),
            "s_v_b": self._float(fast_s_v),
            "s_t_b": self._float(fast_s_t),
            "s_f_b": self._float(s_f),
            "s_branch_b": self._float(branch_calibration),
            "s_v": self._float(fast_s_v),
            "s_t": self._float(fast_s_t),
            "s_f": self._float(s_f),
            "s_branch": self._float(branch_calibration),
            "fcf": self._float(fcf),
            "rfcf": self._float(rfcf),
            "rlin_fcf": self._float(rfcf),
            "gv_norm": self._float(gv_norm),
            "gt_norm": self._float(gt_norm),
            "gf_norm": self._float(gf_norm),
            "cofla_train_uses_fast_fcf": 1.0,
            "cofla_f_no_second_order": 1.0,
            "cofla_f_one_probe_fusion_exact": 1.0,
            "cofla_f_branch_calibrated": 1.0,
            "cofla_gate": self._float(fcf_gate),
            "cofla_perturbed_loss": self._float(loss_f_plus),
            "cofla_robust_delta": self._float(loss_f_plus - base_loss),
            "cofla_j_bf": float(self._j_bf),
            "cofla_alpha_star": float(self._alpha_star),
            "cofla_projection_active": 1.0 if self._projection_active else 0.0,
            "cofla_projection_dot": float(stats_f["projection_dot"]),
            "cofla_projection_norm_task": float(stats_f["projection_norm_task"]),
            "cofla_projection_norm_bf": float(stats_f["projection_norm_obj"]),
            "cofla_branch_v_alpha_star": 0.0,
            "cofla_branch_t_alpha_star": 0.0,
            "cofla_branch_v_projection_active": 0.0,
            "cofla_branch_t_projection_active": 0.0,
            "cofla_branch_projection_params": 0.0,
        }
        self._last_logs = dict(logs)
        return StepOutput(loss=total_loss, logs=logs)
