from __future__ import annotations

import torch

from .base_method import BaseMethod, StepOutput


class MASAMLoRAMethod(BaseMethod):
    name = "masam_lora"

    def __init__(self, args):
        super().__init__(args)
        self._branch_grad_map = {}
        self._clean_branch_grad_map = {}
        self._dominant_modality = ""
        self._ma_v = None
        self._ma_t = None
        self._prev_v = None
        self._prev_t = None

    def training_step(self, wrapper, batch) -> StepOutput:
        """MASAM-LoRA aligned with the controlled late-fusion update style.

        It uses VLM token-proxy modality dropout for unimodal losses, APS to select
        the dominant modality, MDPS cosine scaling for the branch-LoRA SAM radius,
        and replaces only branch-LoRA gradients after clean backward. Fusion LoRA
        and the task head therefore receive clean task-loss gradients.
        """
        base_outputs, hidden, image_mask, text_mask = self._capture_loss_hidden_masks(wrapper, batch)
        base_loss = base_outputs.loss
        if base_loss is None:
            raise RuntimeError("masam_lora step failed to produce a base loss.")

        loss_v_uni, loss_t_uni, _ = self._compute_modality_dropout_losses_from_hidden(
            wrapper, batch, hidden, image_mask, text_mask
        )
        branch_v, branch_t = self._grouped_branch_params(wrapper)
        params_v = [param for _, param in branch_v]
        params_t = [param for _, param in branch_t]

        grads_fuse_v = torch.autograd.grad(
            outputs=base_loss,
            inputs=params_v,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        ) if params_v else []
        grads_fuse_t = torch.autograd.grad(
            outputs=base_loss,
            inputs=params_t,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        ) if params_t else []
        grads_uni_v = torch.autograd.grad(
            outputs=loss_v_uni,
            inputs=params_v,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        ) if params_v else []
        grads_uni_t = torch.autograd.grad(
            outputs=loss_t_uni,
            inputs=params_t,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        ) if params_t else []

        fuse_v_map = self._grad_map(branch_v, grads_fuse_v)
        fuse_t_map = self._grad_map(branch_t, grads_fuse_t)
        self._clean_branch_grad_map = self._merge_grad_maps_sum(fuse_v_map, fuse_t_map)
        uni_v_map = self._grad_map(branch_v, grads_uni_v)
        uni_t_map = self._grad_map(branch_t, grads_uni_t)
        gamma_v = float(
            self._cosine_between_grad_maps(branch_v, fuse_v_map, uni_v_map, device=base_loss.device).detach().cpu().item()
        )
        gamma_t = float(
            self._cosine_between_grad_maps(branch_t, fuse_t_map, uni_t_map, device=base_loss.device).detach().cpu().item()
        )

        beta = max(0.0, min(0.9999, float(getattr(self.args, "masam_ma_beta", 0.9))))
        alpha = max(0.0, min(1.0, float(getattr(self.args, "masam_aps_alpha", 0.5))))
        lv = float(loss_v_uni.detach().cpu().item())
        lt = float(loss_t_uni.detach().cpu().item())
        if self._ma_v is None:
            self._ma_v = lv
            self._ma_t = lt
            self._prev_v = lv
            self._prev_t = lt
        ma_v = beta * float(self._ma_v) + (1.0 - beta) * lv
        ma_t = beta * float(self._ma_t) + (1.0 - beta) * lt
        decay_v = max(0.0, float(self._prev_v) - ma_v)
        decay_t = max(0.0, float(self._prev_t) - ma_t)
        self._ma_v = ma_v
        self._ma_t = ma_t
        self._prev_v = lv
        self._prev_t = lt

        aps_v = alpha * decay_v + (1.0 - alpha) * gamma_v
        aps_t = alpha * decay_t + (1.0 - alpha) * gamma_t
        dominant = "visual" if aps_v >= aps_t else "text"

        if dominant == "visual":
            dominant_named = branch_v
            dominant_fuse_grad_map = fuse_v_map
            gamma = max(0.0, gamma_v)
        else:
            dominant_named = branch_t
            dominant_fuse_grad_map = fuse_t_map
            gamma = max(0.0, gamma_t)

        perturb, grad_norm = self._build_param_perturb_from_grad_map(
            dominant_named,
            dominant_fuse_grad_map,
            radius=float(getattr(self.args, "sam_rho", getattr(self.args, "rho_f", 0.0))) * float(gamma),
            device=base_loss.device,
        )

        if perturb:
            wrapper._apply_param_perturb(perturb, sign=1)
        try:
            adv_outputs, _ = self._forward_training_loss(wrapper, batch, capture_layer0=False)
            adv_loss = adv_outputs.loss
            if adv_loss is None:
                raise RuntimeError("masam_lora perturbed forward failed to produce a loss.")
            adv_grads_dom = torch.autograd.grad(
                outputs=adv_loss,
                inputs=[param for _, param in dominant_named],
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            ) if dominant_named else []
        finally:
            if perturb:
                wrapper._apply_param_perturb(perturb, sign=-1)

        dom_map = self._grad_map(dominant_named, adv_grads_dom)
        if dominant == "visual":
            branch_grad_map = self._merge_grad_maps_sum(dom_map, uni_v_map, fuse_t_map, uni_t_map)
        else:
            branch_grad_map = self._merge_grad_maps_sum(fuse_v_map, uni_v_map, dom_map, uni_t_map)

        self._branch_grad_map = branch_grad_map
        self._dominant_modality = dominant
        logs = {
            "train_loss": float(base_loss.detach().cpu().item()),
            "task_loss": float(base_loss.detach().cpu().item()),
            "masam_loss_v_uni": lv,
            "masam_loss_t_uni": lt,
            "masam_gamma_v": float(gamma_v),
            "masam_gamma_t": float(gamma_t),
            "masam_decay_v": float(decay_v),
            "masam_decay_t": float(decay_t),
            "masam_aps_v": float(aps_v),
            "masam_aps_t": float(aps_t),
            "masam_dominant_is_visual": 1.0 if dominant == "visual" else 0.0,
            "masam_mdps_gamma": float(gamma),
            "masam_adaptive_rho": float(getattr(self.args, "sam_rho", getattr(self.args, "rho_f", 0.0))) * float(gamma),
            "masam_grad_norm": float(grad_norm.detach().cpu().item()),
            "masam_perturb_scope": 1.0 if perturb else 0.0,
            "masam_adv_loss": float(adv_loss.detach().cpu().item()),
            "masam_branch_grad_replacement_pending": float(len(self._branch_grad_map)),
        }
        return StepOutput(loss=base_loss, logs=logs)

    def after_backward(self, wrapper):
        replaced = self._install_grad_replacement(
            wrapper,
            self._branch_grad_map,
            clean_grad_map=self._clean_branch_grad_map,
        )
        dominant = self._dominant_modality
        self._branch_grad_map = {}
        self._clean_branch_grad_map = {}
        self._dominant_modality = ""
        return {
            "masam_replaced_branch_grad_count": float(replaced),
            "masam_dominant_modality": dominant,
        }
