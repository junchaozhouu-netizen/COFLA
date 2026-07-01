from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch

from .base_method import BaseMethod, StepOutput


def bounded_positive_fcf_gate(value: torch.Tensor) -> torch.Tensor:
    """Parameter-free bounded gate used by the robust COFLA objective.

    The gate is activated only by positive branch--fusion mismatch and is
    detached.  Therefore it controls the robust objective strength without
    becoming an extra learnable or manually weighted loss term.
    """
    positive = torch.relu(value.float())
    gate = positive / (1.0 + positive)
    return gate.detach().to(device=value.device, dtype=value.dtype)


class COFLAMethod(BaseMethod):
    """COFLA-E for VLM PEFT.

    This mirrors the updated controlled late-fusion implementation:
    1) compute exact FCF/RFCF geometry,
    2) train with the FCF-gated robust objective,
    3) retain branch-side and fusion-side projection corrections as safeguards.

    In the VLM setting, all trainable parameters are matched LoRA parameters;
    fusion-side correction is applied to fusion-sensitive LoRA only, not to the
    task head.
    """

    name = "cofla"

    def __init__(self, args):
        super().__init__(args)
        self._fusion_correction_map: Dict[str, torch.Tensor] = {}
        self._branch_correction_map: Dict[str, torch.Tensor] = {}
        self._alpha_star: float = 0.0
        self._projection_active: bool = False
        self._j_bf: float = 0.0
        self._last_logs: Dict[str, float] = {}

    @staticmethod
    def _grad_map(
        named_params: List[Tuple[str, torch.nn.Parameter]],
        grads: Tuple[Optional[torch.Tensor], ...] | List[Optional[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        grad_map: Dict[str, torch.Tensor] = {}
        for (name, param), grad in zip(named_params, grads):
            if grad is None:
                continue
            grad_map[name] = grad.detach().to(device=param.device, dtype=param.dtype)
        return grad_map

    @staticmethod
    def _dot_and_norm_b(
        named_params: List[Tuple[str, torch.nn.Parameter]],
        grad_task: Dict[str, torch.Tensor],
        grad_obj: Dict[str, torch.Tensor],
    ) -> Tuple[float, float, float]:
        dot = 0.0
        norm_task_sq = 0.0
        norm_obj_sq = 0.0
        for name, _ in named_params:
            gt = grad_task.get(name)
            go = grad_obj.get(name)
            if gt is not None:
                gt_f = gt.detach().float()
                norm_task_sq += float(gt_f.pow(2).sum().cpu().item())
            if go is not None:
                go_f = go.detach().float()
                norm_obj_sq += float(go_f.pow(2).sum().cpu().item())
            if gt is not None and go is not None:
                dot += float((gt.detach().float() * go.detach().float()).sum().cpu().item())
        return float(dot), float(norm_task_sq), float(norm_obj_sq)

    def _build_projection_for_objective(
        self,
        *,
        named_params: List[Tuple[str, torch.nn.Parameter]],
        loss_task: torch.Tensor,
        objective: torch.Tensor,
        objective_value: float,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        params = [param for _, param in named_params]
        if not params:
            return {}, {
                "alpha_star": 0.0,
                "projection_active": 0.0,
                "projection_dot": 0.0,
                "projection_norm_task": 0.0,
                "projection_norm_obj": 0.0,
            }

        grads_task = torch.autograd.grad(
            outputs=loss_task,
            inputs=params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        grads_obj = torch.autograd.grad(
            outputs=objective,
            inputs=params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        grad_task_map = self._grad_map(named_params, grads_task)
        grad_obj_map = self._grad_map(named_params, grads_obj)
        dot, norm_task_sq, norm_obj_sq = self._dot_and_norm_b(named_params, grad_task_map, grad_obj_map)
        eps = float(getattr(self.args, "fcf_eps", getattr(self.args, "eps", 1e-8)))

        alpha_star = 0.0
        if objective_value > 0.0 and norm_obj_sq > 0.0 and math.isfinite(dot):
            alpha_star = max(0.0, -dot / (norm_obj_sq + eps))

        correction: Dict[str, torch.Tensor] = {}
        if alpha_star > 0.0 and math.isfinite(alpha_star):
            for name, param in named_params:
                go = grad_obj_map.get(name)
                if go is None:
                    continue
                correction[name] = (go * alpha_star).to(device=param.device, dtype=param.dtype)

        return correction, {
            "alpha_star": float(alpha_star),
            "projection_active": 1.0 if correction else 0.0,
            "projection_dot": float(dot),
            "projection_norm_task": math.sqrt(max(float(norm_task_sq), 0.0)),
            "projection_norm_obj": math.sqrt(max(float(norm_obj_sq), 0.0)),
        }

    @staticmethod
    def _float(value) -> float:
        if torch.is_tensor(value):
            return float(torch.nan_to_num(value.detach().float(), nan=0.0, posinf=0.0, neginf=0.0).cpu().item())
        return float(value)

    @staticmethod
    def _perturbed_loss_from_geom(geom: Dict[str, torch.Tensor], base_loss: torch.Tensor) -> torch.Tensor:
        if all(key in geom for key in ("loss_v_plus", "loss_t_plus", "loss_f_plus")):
            return (geom["loss_v_plus"] + geom["loss_t_plus"] + geom["loss_f_plus"]) / 3.0
        # Backward-compatible fallback for older wrappers that only return sharpness increments.
        s_v = geom.get("s_v", geom.get("s_v_b", base_loss.new_tensor(0.0)))
        s_t = geom.get("s_t", geom.get("s_t_b", base_loss.new_tensor(0.0)))
        s_f = geom.get("s_f", geom.get("s_f_b", base_loss.new_tensor(0.0)))
        return base_loss + (s_v + s_t + s_f) / 3.0

    def _compute_training_geometry(self, wrapper, batch) -> Tuple[Dict[str, torch.Tensor], float]:
        probe = wrapper.compute_geometry_probe(batch)
        geom = wrapper.compute_geometry_metrics_from_probe(batch, probe)
        return geom, 0.0

    def training_step(self, wrapper, batch) -> StepOutput:
        geom, uses_fast = self._compute_training_geometry(wrapper, batch)

        base_loss = geom.get("loss", geom.get("probe_loss"))
        if base_loss is None:
            raise RuntimeError("COFLA requires a scalar task loss in geometry metrics.")

        rfcf = geom["rfcf"]
        j_bf = torch.relu(rfcf)
        fcf_gate = bounded_positive_fcf_gate(rfcf)
        perturbed_loss = self._perturbed_loss_from_geom(geom, base_loss)
        total_loss = (1.0 - fcf_gate) * base_loss + fcf_gate * perturbed_loss

        self._branch_correction_map = {}
        visual_named, text_named = self._grouped_branch_params(wrapper)
        s_v = geom.get("s_v", geom.get("s_v_b", base_loss.new_tensor(0.0)))
        s_t = geom.get("s_t", geom.get("s_t_b", base_loss.new_tensor(0.0)))
        corr_v, stats_v = self._build_projection_for_objective(
            named_params=visual_named,
            loss_task=total_loss,
            objective=s_v,
            objective_value=self._float(s_v),
        )
        corr_t, stats_t = self._build_projection_for_objective(
            named_params=text_named,
            loss_task=total_loss,
            objective=s_t,
            objective_value=self._float(s_t),
        )
        self._branch_correction_map.update(corr_v)
        self._branch_correction_map.update(corr_t)

        fusion_named = self._named_fusion_params(wrapper)
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
            "loss_fuse": self._float(geom.get("s_f", geom.get("s_f_b", base_loss.new_tensor(0.0)))),
            "s_v_b": self._float(geom.get("s_v_b", geom.get("s_v", base_loss.new_tensor(0.0)))),
            "s_t_b": self._float(geom.get("s_t_b", geom.get("s_t", base_loss.new_tensor(0.0)))),
            "s_f_b": self._float(geom.get("s_f_b", geom.get("s_f", base_loss.new_tensor(0.0)))),
            "s_branch_b": self._float(geom.get("s_branch_b", geom.get("s_branch", base_loss.new_tensor(0.0)))),
            "fcf": self._float(geom["fcf"]),
            "rfcf": self._float(rfcf),
            "rlin_fcf": self._float(geom.get("rlin_fcf", geom.get("r_lin_fcf", base_loss.new_tensor(0.0)))),
            "gv_norm": self._float(geom.get("gv_norm", geom.get("grad_zv_norm", base_loss.new_tensor(0.0)))),
            "gt_norm": self._float(geom.get("gt_norm", geom.get("grad_zt_norm", base_loss.new_tensor(0.0)))),
            "gf_norm": self._float(geom.get("gf_norm", geom.get("grad_fusion_norm", base_loss.new_tensor(0.0)))),
            "cofla_train_uses_fast_fcf": float(uses_fast),
            "cofla_gate": self._float(fcf_gate),
            "cofla_perturbed_loss": self._float(perturbed_loss),
            "cofla_robust_delta": self._float(perturbed_loss - base_loss),
            "cofla_j_bf": float(self._j_bf),
            "cofla_alpha_star": float(self._alpha_star),
            "cofla_projection_active": 1.0 if self._projection_active else 0.0,
            "cofla_projection_dot": float(stats_f["projection_dot"]),
            "cofla_projection_norm_task": float(stats_f["projection_norm_task"]),
            "cofla_projection_norm_bf": float(stats_f["projection_norm_obj"]),
            "cofla_branch_v_alpha_star": float(stats_v["alpha_star"]),
            "cofla_branch_t_alpha_star": float(stats_t["alpha_star"]),
            "cofla_branch_v_projection_active": float(stats_v["projection_active"]),
            "cofla_branch_t_projection_active": float(stats_t["projection_active"]),
            "cofla_branch_projection_params": float(len(self._branch_correction_map)),
        }
        self._last_logs = dict(logs)
        return StepOutput(loss=total_loss, logs=logs)

    def after_backward(self, wrapper):
        scale = 1.0 / max(1, int(getattr(self.args, "gradient_accumulation_steps", 1)))
        branch_applied = 0
        if self._branch_correction_map:
            for name, param in self._named_branch_params(wrapper):
                correction = self._branch_correction_map.get(name)
                if correction is None:
                    continue
                correction = correction.to(device=param.device, dtype=param.dtype)
                if param.grad is None:
                    param.grad = correction.clone().mul_(scale)
                else:
                    param.grad.add_(correction, alpha=scale)
                branch_applied += 1

        fusion_applied = 0
        if self._fusion_correction_map:
            for name, param in self._named_fusion_params(wrapper):
                correction = self._fusion_correction_map.get(name)
                if correction is None:
                    continue
                correction = correction.to(device=param.device, dtype=param.dtype)
                if param.grad is None:
                    param.grad = correction.clone().mul_(scale)
                else:
                    param.grad.add_(correction, alpha=scale)
                fusion_applied += 1

        self._branch_correction_map = {}
        self._fusion_correction_map = {}
        return {
            "cofla_alpha_star": float(self._alpha_star),
            "cofla_projection_active": 1.0 if self._projection_active else 0.0,
            "cofla_projection_params": float(fusion_applied),
            "cofla_branch_projection_params": float(branch_applied),
            "cofla_j_bf": float(self._j_bf),
        }
