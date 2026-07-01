from __future__ import annotations

import torch

from .base_method import BaseMethod, StepOutput


class SAMLoRAMethod(BaseMethod):
    name = "sam_lora"

    def __init__(self, args):
        super().__init__(args)
        self._branch_grad_map = {}
        self._clean_branch_grad_map = {}
        self._last_replaced = 0

    def training_step(self, wrapper, batch) -> StepOutput:
        """Branch-only SAM-LoRA baseline aligned with controlled late-fusion.

        The returned loss is the clean task loss, so fusion-sensitive LoRA and the
        task head keep clean-loss gradients. Only branch-LoRA gradients are replaced
        in after_backward() by adversarial gradients computed under branch-LoRA SAM
        perturbation. This avoids giving SAM an implicit fusion-robust update.
        """
        named_params = self._named_branch_params(wrapper)
        params = [p for _, p in named_params]

        outputs, _ = self._forward_training_loss(wrapper, batch, capture_layer0=False)
        base_loss = outputs.loss
        if base_loss is None:
            raise RuntimeError("sam_lora step failed to produce a base loss.")

        grads = torch.autograd.grad(
            outputs=base_loss,
            inputs=params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        ) if named_params else []
        grad_map = self._grad_map(named_params, grads)
        self._clean_branch_grad_map = grad_map
        perturb, grad_norm = self._build_param_perturb_from_grad_map(
            named_params,
            grad_map,
            radius=float(getattr(self.args, "sam_rho", getattr(self.args, "rho_f", 0.0))),
            device=base_loss.device,
        )

        if perturb:
            wrapper._apply_param_perturb(perturb, sign=1)
        try:
            adv_outputs, _ = self._forward_training_loss(wrapper, batch, capture_layer0=False)
            adv_loss = adv_outputs.loss
            if adv_loss is None:
                raise RuntimeError("sam_lora perturbed forward failed to produce an adversarial loss.")
            adv_grads = torch.autograd.grad(
                outputs=adv_loss,
                inputs=params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            ) if named_params else []
        finally:
            if perturb:
                wrapper._apply_param_perturb(perturb, sign=-1)

        self._branch_grad_map = self._grad_map(named_params, adv_grads)
        logs = {
            "train_loss": float(base_loss.detach().cpu().item()),
            "task_loss": float(base_loss.detach().cpu().item()),
            "sam_base_loss": float(base_loss.detach().cpu().item()),
            "sam_adv_loss": float(adv_loss.detach().cpu().item()),
            "sam_grad_norm": float(grad_norm.detach().cpu().item()),
            "sam_perturb_scope": 1.0 if perturb else 0.0,
            "sam_branch_grad_replacement_pending": float(len(self._branch_grad_map)),
            "sam_branch_only_update": 1.0,
        }
        return StepOutput(loss=base_loss, logs=logs)

    def after_backward(self, wrapper):
        replaced = self._install_grad_replacement(
            wrapper,
            self._branch_grad_map,
            clean_grad_map=self._clean_branch_grad_map,
        )
        self._branch_grad_map = {}
        self._clean_branch_grad_map = {}
        self._last_replaced = replaced
        return {
            "sam_replaced_branch_grad_count": float(replaced),
            "sam_branch_only_update": 1.0,
        }
