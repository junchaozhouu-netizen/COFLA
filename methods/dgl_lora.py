from __future__ import annotations

import gc

import torch

from .base_method import BaseMethod, StepOutput


class DGLLoRAMethod(BaseMethod):
    name = "dgl_lora"

    def __init__(self, args):
        super().__init__(args)
        self._branch_grad_map = {}
        self._clean_branch_grad_map = {}
        self._last_conflict = 0.0
        self._last_replaced = 0

    def _is_llava_ov_0p5b_hm_verbalizer(self, batch) -> bool:
        """Limit the memory-safe fallback to LLaVA-OneVision-Qwen2-0.5B on HM."""
        model_name = str(getattr(self.args, "model_name", "")).lower().replace("-", "_")
        model_family = str(getattr(self.args, "model_family", "")).lower().replace("-", "_")
        task_format = str(getattr(self.args, "task_format", "")).lower()
        dataset_name = str(batch.get("dataset_name", getattr(self.args, "dataset", ""))).lower()
        is_llava_ov = "llava_onevision" in model_family or "llava_onevision" in model_name or "llava_ov" in model_name
        is_0p5b = "0_5b" in model_name or "0.5b" in model_name or "0p5b" in model_name
        is_hm = dataset_name == "hateful_memes" or str(getattr(self.args, "dataset", "")).lower() == "hateful_memes"
        is_verbalizer = "verbalizer" in task_format
        return bool(is_llava_ov and is_0p5b and is_hm and is_verbalizer)

    @staticmethod
    def _release_cuda_cache():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def training_step(self, wrapper, batch) -> StepOutput:
        """DGL-LoRA aligned with controlled late-fusion semantics.

        Default path is unchanged. For LLaVA-OneVision-Qwen2-0.5B on Hateful
        Memes verbalizer classification, use a memory-safe path that avoids
        keeping clean, visual-only, text-only and empty-modality graphs alive at
        the same time. The update semantics stay the same: fusion-sensitive
        LoRA/task head receive clean multimodal task gradients, while branch-LoRA
        gradients are replaced after backward by unimodal proxy gradients.
        """
        if self._is_llava_ov_0p5b_hm_verbalizer(batch):
            return self._training_step_llava_ov_0p5b_memory_safe(wrapper, batch)
        return self._training_step_default(wrapper, batch)

    def _training_step_default(self, wrapper, batch) -> StepOutput:
        base_outputs, hidden, image_mask, text_mask = self._capture_loss_hidden_masks(wrapper, batch)
        base_loss = base_outputs.loss
        if base_loss is None:
            raise RuntimeError("dgl_lora step failed to produce a base loss.")

        loss_v_uni, loss_t_uni, _ = self._compute_modality_dropout_losses_from_hidden(
            wrapper, batch, hidden, image_mask, text_mask
        )
        branch_v, branch_t = self._grouped_branch_params(wrapper)

        grads_clean_v = torch.autograd.grad(
            outputs=base_loss,
            inputs=[param for _, param in branch_v],
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        ) if branch_v else []
        grads_clean_t = torch.autograd.grad(
            outputs=base_loss,
            inputs=[param for _, param in branch_t],
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        ) if branch_t else []
        clean_v_map = self._grad_map(branch_v, grads_clean_v)
        clean_t_map = self._grad_map(branch_t, grads_clean_t)
        self._clean_branch_grad_map = self._merge_grad_maps_sum(clean_v_map, clean_t_map)

        grads_uni_v = torch.autograd.grad(
            outputs=loss_v_uni,
            inputs=[param for _, param in branch_v],
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        ) if branch_v else []
        grads_uni_t = torch.autograd.grad(
            outputs=loss_t_uni,
            inputs=[param for _, param in branch_t],
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        ) if branch_t else []

        uni_v_map = self._grad_map(branch_v, grads_uni_v)
        uni_t_map = self._grad_map(branch_t, grads_uni_t)
        self._last_conflict = float(
            self._cosine_between_grad_maps(
                branch_v + branch_t,
                self._merge_grad_maps_sum(uni_v_map),
                self._merge_grad_maps_sum(uni_t_map),
                device=base_loss.device,
            ).detach().cpu().item()
        )
        self._branch_grad_map = self._merge_grad_maps_sum(uni_v_map, uni_t_map)

        logs = {
            "train_loss": float(base_loss.detach().cpu().item()),
            "task_loss": float(base_loss.detach().cpu().item()),
            "dgl_multimodal_clean_loss": float(base_loss.detach().cpu().item()),
            "dgl_unimodal_v_loss": float(loss_v_uni.detach().cpu().item()),
            "dgl_unimodal_t_loss": float(loss_t_uni.detach().cpu().item()),
            "dgl_branch_param_count": float(len(branch_v) + len(branch_t)),
            "conflict_cosine": self._last_conflict,
            "dgl_branch_grad_replacement_pending": float(len(self._branch_grad_map)),
        }
        return StepOutput(loss=base_loss, logs=logs)

    def _training_step_llava_ov_0p5b_memory_safe(self, wrapper, batch) -> StepOutput:
        branch_v, branch_t = self._grouped_branch_params(wrapper)
        params_v = [param for _, param in branch_v]
        params_t = [param for _, param in branch_t]
        branch_all = branch_v + branch_t
        params_all = [param for _, param in branch_all]

        # 1) Capture hidden/masks without retaining a training graph. Hidden-dropout
        #    perturbations use hidden.detach(), so gradients through this capture are
        #    not needed.
        with torch.no_grad():
            probe_outputs, hidden, image_mask, text_mask = self._capture_loss_hidden_masks(wrapper, batch)
            probe_clean_value = float(probe_outputs.loss.detach().cpu().item()) if probe_outputs.loss is not None else 0.0
        hidden = hidden.detach()
        image_mask = image_mask.detach()
        text_mask = text_mask.detach()
        self._release_cuda_cache()

        # 2) Visual-only proxy branch gradient. This graph is freed immediately.
        drop_t = self._hidden_drop_perturb(hidden, text_mask)
        out_v, _ = self._forward_training_loss(wrapper, batch, capture_layer0=False, hidden_perturb=drop_t)
        loss_v_uni = out_v.loss
        if loss_v_uni is None:
            raise RuntimeError("dgl_lora memory-safe visual-unimodal proxy failed to produce a loss.")
        loss_v_value = float(loss_v_uni.detach().cpu().item())
        grads_uni_v = torch.autograd.grad(
            outputs=loss_v_uni,
            inputs=params_v,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        ) if branch_v else []
        uni_v_map = self._grad_map(branch_v, grads_uni_v)
        del out_v, loss_v_uni, grads_uni_v, drop_t
        self._release_cuda_cache()

        # 3) Text-only proxy branch gradient. This graph is also freed immediately.
        drop_v = self._hidden_drop_perturb(hidden, image_mask)
        out_t, _ = self._forward_training_loss(wrapper, batch, capture_layer0=False, hidden_perturb=drop_v)
        loss_t_uni = out_t.loss
        if loss_t_uni is None:
            raise RuntimeError("dgl_lora memory-safe text-unimodal proxy failed to produce a loss.")
        loss_t_value = float(loss_t_uni.detach().cpu().item())
        grads_uni_t = torch.autograd.grad(
            outputs=loss_t_uni,
            inputs=params_t,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        ) if branch_t else []
        uni_t_map = self._grad_map(branch_t, grads_uni_t)
        del out_t, loss_t_uni, grads_uni_t, drop_v, hidden, image_mask, text_mask
        self._release_cuda_cache()

        self._last_conflict = float(
            self._cosine_between_grad_maps(
                branch_all,
                self._merge_grad_maps_sum(uni_v_map),
                self._merge_grad_maps_sum(uni_t_map),
                device=next((p.device for _, p in branch_all), torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            ).detach().cpu().item()
        ) if branch_all else 0.0
        self._branch_grad_map = self._merge_grad_maps_sum(uni_v_map, uni_t_map)

        # 4) Final clean forward for the main training-loop backward. This preserves
        #    clean fusion/task-head gradients. We keep only this clean graph alive.
        base_outputs, _ = self._forward_training_loss(wrapper, batch, capture_layer0=False)
        base_loss = base_outputs.loss
        if base_loss is None:
            raise RuntimeError("dgl_lora memory-safe final clean forward failed to produce a base loss.")
        clean_grads = torch.autograd.grad(
            outputs=base_loss,
            inputs=params_all,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        ) if branch_all else []
        self._clean_branch_grad_map = self._grad_map(branch_all, clean_grads)

        logs = {
            "train_loss": float(base_loss.detach().cpu().item()),
            "task_loss": float(base_loss.detach().cpu().item()),
            "dgl_multimodal_clean_loss": float(base_loss.detach().cpu().item()),
            "dgl_probe_clean_loss": probe_clean_value,
            "dgl_unimodal_v_loss": loss_v_value,
            "dgl_unimodal_t_loss": loss_t_value,
            "dgl_branch_param_count": float(len(branch_v) + len(branch_t)),
            "conflict_cosine": self._last_conflict,
            "dgl_branch_grad_replacement_pending": float(len(self._branch_grad_map)),
            "llava_ov_0p5b_memory_safe": 1.0,
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
            "dgl_replaced_branch_grad_count": float(replaced),
        }
