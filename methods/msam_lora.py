from __future__ import annotations

import torch

from .base_method import BaseMethod, StepOutput


class MSAMLoRAMethod(BaseMethod):
    name = "msam_lora"

    def __init__(self, args):
        super().__init__(args)
        self._pending_perturb = {}
        self._pending_scope = ""

    def training_step(self, wrapper, batch) -> StepOutput:
        """M-SAM-LoRA using VLM token-proxy modality dropout.

        This mirrors the controlled late-fusion implementation: estimate two-modality
        Shapley weights with visual-only/text-only/empty proxy losses, choose the
        dominant branch, and apply SAM only to that branch-LoRA group.
        """
        base_outputs, hidden, image_mask, text_mask = self._capture_loss_hidden_masks(wrapper, batch)
        base_loss = base_outputs.loss
        if base_loss is None:
            raise RuntimeError("msam_lora step failed to produce a base loss.")

        loss_v_uni, loss_t_uni, loss_empty = self._compute_modality_dropout_losses_from_hidden(
            wrapper, batch, hidden, image_mask, text_mask
        )
        w_v, w_t, phi_v, phi_t = self._two_modality_shapley_weights(
            loss_fuse=base_loss,
            loss_v=loss_v_uni,
            loss_t=loss_t_uni,
            loss_empty=loss_empty,
            eps=float(getattr(self.args, "msam_shapley_eps", 1e-8)),
        )
        dominant = "visual" if w_v >= w_t else "text"
        branch_v, branch_t = self._grouped_branch_params(wrapper)
        dominant_named = branch_v if dominant == "visual" else branch_t
        dominant_params = [param for _, param in dominant_named]

        modulated_loss = base_loss + w_v * loss_v_uni + w_t * loss_t_uni
        grads_dom = torch.autograd.grad(
            outputs=modulated_loss,
            inputs=dominant_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        ) if dominant_named else []
        grad_map = self._grad_map(dominant_named, grads_dom)
        perturb, grad_norm = self._build_param_perturb_from_grad_map(
            dominant_named,
            grad_map,
            radius=float(getattr(self.args, "sam_rho", getattr(self.args, "rho_f", 0.0))),
            device=base_loss.device,
        )

        if perturb:
            wrapper._apply_param_perturb(perturb, sign=1)
        try:
            adv_outputs, adv_hidden, adv_image_mask, adv_text_mask = self._capture_loss_hidden_masks(wrapper, batch)
            adv_base_loss = adv_outputs.loss
            if adv_base_loss is None:
                raise RuntimeError("msam_lora perturbed forward failed to produce a loss.")
            adv_loss_v_uni, adv_loss_t_uni, _ = self._compute_modality_dropout_losses_from_hidden(
                wrapper, batch, adv_hidden, adv_image_mask, adv_text_mask
            )
            total_loss = adv_base_loss + w_v * adv_loss_v_uni + w_t * adv_loss_t_uni
        except Exception:
            if perturb:
                wrapper._apply_param_perturb(perturb, sign=-1)
            self._pending_perturb = {}
            self._pending_scope = ""
            raise

        self._pending_perturb = perturb
        self._pending_scope = f"msam_dominant_{dominant}"
        logs = {
            "task_loss": float(base_loss.detach().cpu().item()),
            "train_loss": float(total_loss.detach().cpu().item()),
            "msam_phi_v": float(phi_v),
            "msam_phi_t": float(phi_t),
            "msam_w_v": float(w_v),
            "msam_w_t": float(w_t),
            "msam_dominant_is_visual": 1.0 if dominant == "visual" else 0.0,
            "msam_loss_v_uni": float(loss_v_uni.detach().cpu().item()),
            "msam_loss_t_uni": float(loss_t_uni.detach().cpu().item()),
            "msam_loss_empty": float(loss_empty.detach().cpu().item()),
            "msam_grad_norm": float(grad_norm.detach().cpu().item()),
            "msam_adv_loss": float(total_loss.detach().cpu().item()),
            "msam_perturb_scope": 1.0 if perturb else 0.0,
        }
        return StepOutput(loss=total_loss, logs=logs)

    def after_backward(self, wrapper):
        restored = 0
        if self._pending_perturb:
            wrapper._apply_param_perturb(self._pending_perturb, sign=-1)
            restored = len(self._pending_perturb)
        scope = self._pending_scope
        self._pending_perturb = {}
        self._pending_scope = ""
        return {
            "restored_param_perturb": float(restored),
            "restored_param_perturb_scope": scope,
        }
