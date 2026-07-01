from __future__ import annotations

import gc
import math
from typing import Dict, List, Optional, Tuple

import torch

from .base_method import BaseMethod, StepOutput


class ESAMLoRAMethod(BaseMethod):
    name = "esam_lora"

    def __init__(self, args):
        super().__init__(args)
        self._branch_grad_map = {}
        self._clean_branch_grad_map = {}

    def _is_llava_ov_0p5b_hm_verbalizer(self, batch) -> bool:
        """Limit the memory-safe fallback to LLaVA-OneVision-Qwen2-0.5B on HM.

        This keeps Qwen / InternVL3 and other existing results on the original
        implementation path, so those experiments do not need to be rerun.
        """
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

    def _build_swp_perturb(
        self,
        named_params: List[Tuple[str, torch.nn.Parameter]],
        grads: Tuple[Optional[torch.Tensor], ...],
        *,
        radius: float,
        device: torch.device,
        eps: float = 1e-12,
    ):
        """ESAM SWP: sample branch tensors and rescale by 1 / beta."""
        beta = max(float(eps), min(1.0, float(getattr(self.args, "esam_swp_prob", 0.6))))
        candidates = [(item, grad) for item, grad in zip(named_params, grads) if grad is not None]
        selected = []
        for item, grad in candidates:
            if torch.rand((), device=device).item() <= beta:
                selected.append((item, grad))
        if not selected and candidates:
            idx = int(torch.randint(low=0, high=len(candidates), size=(), device=device).item())
            selected.append(candidates[idx])

        grad_sq = 0.0
        for _, grad in selected:
            grad_sq += float(grad.detach().float().pow(2).sum().cpu().item())
        grad_norm_value = math.sqrt(max(grad_sq, 0.0))
        grad_norm = torch.tensor(grad_norm_value, device=device, dtype=torch.float32)
        if float(radius) <= 0.0 or grad_norm_value <= float(eps):
            return {}, grad_norm, 0, beta

        scale = float(radius) / (beta * (grad_norm_value + float(eps)))
        perturb: Dict[str, torch.Tensor] = {}
        for (name, param), grad in selected:
            perturb[name] = grad.detach().to(device=param.device, dtype=param.dtype) * scale
        return perturb, grad_norm, len(perturb), beta

    def training_step(self, wrapper, batch) -> StepOutput:
        """Branch-only ESAM-LoRA aligned with controlled late-fusion.

        Default path is unchanged. For LLaVA-OneVision-Qwen2-0.5B on Hateful
        Memes verbalizer classification, use a memory-safe three-forward path
        that avoids keeping the clean and adversarial graphs alive at the same
        time. The update semantics remain the same: fusion/task-head parameters
        receive clean gradients, while branch-LoRA gradients are replaced by the
        ESAM adversarial branch gradients in ``after_backward``.
        """
        if self._is_llava_ov_0p5b_hm_verbalizer(batch):
            return self._training_step_llava_ov_0p5b_memory_safe(wrapper, batch)
        return self._training_step_default(wrapper, batch)

    def _training_step_default(self, wrapper, batch) -> StepOutput:
        named_params = self._named_branch_params(wrapper)
        params = [p for _, p in named_params]

        outputs, _ = self._forward_training_loss(wrapper, batch, capture_layer0=False)
        base_loss = outputs.loss
        if base_loss is None:
            raise RuntimeError("esam_lora step failed to produce a base loss.")
        base_per_sample = self._compute_per_sample_losses(batch, outputs)

        grads = torch.autograd.grad(
            outputs=base_loss,
            inputs=params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        ) if named_params else []
        self._clean_branch_grad_map = self._grad_map(named_params, grads)
        perturb, grad_norm, swp_selected_count, swp_beta = self._build_swp_perturb(
            named_params,
            grads,
            radius=float(getattr(self.args, "sam_rho", getattr(self.args, "rho_f", 0.0))),
            device=base_loss.device,
            eps=float(getattr(self.args, "eps", 1e-12)),
        )

        if perturb:
            wrapper._apply_param_perturb(perturb, sign=1)
        try:
            adv_outputs, _ = self._forward_training_loss(wrapper, batch, capture_layer0=False)
            adv_per_sample = self._compute_per_sample_losses(batch, adv_outputs)
            sharpness_increase = (adv_per_sample - base_per_sample).detach()

            keep_ratio = max(0.0, min(1.0, float(getattr(self.args, "esam_keep_ratio", 0.5))))
            batch_size = int(base_per_sample.size(0))
            keep_count = batch_size if batch_size <= 1 else max(1, int(math.ceil(batch_size * keep_ratio)))
            if keep_count >= batch_size:
                adv_branch_loss = adv_per_sample.mean()
            else:
                selected = torch.topk(sharpness_increase, k=keep_count, largest=True).indices
                adv_branch_loss = adv_per_sample.index_select(0, selected).mean()

            adv_grads = torch.autograd.grad(
                outputs=adv_branch_loss,
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
            "esam_base_loss": float(base_loss.detach().cpu().item()),
            "esam_adv_loss": float(adv_branch_loss.detach().cpu().item()),
            "esam_selected_count": int(keep_count),
            "esam_selected_ratio": float(keep_count / max(1, batch_size)),
            "esam_grad_norm": float(grad_norm.detach().cpu().item()),
            "esam_swp_beta": float(swp_beta),
            "esam_swp_selected_tensors": float(swp_selected_count),
            "esam_mean_sharpness_increase": float(sharpness_increase.mean().detach().cpu().item()),
            "esam_perturb_scope": 1.0 if perturb else 0.0,
            "esam_branch_grad_replacement_pending": float(len(self._branch_grad_map)),
            "sam_branch_only_update": 1.0,
        }
        return StepOutput(loss=base_loss, logs=logs)

    def _training_step_llava_ov_0p5b_memory_safe(self, wrapper, batch) -> StepOutput:
        named_params = self._named_branch_params(wrapper)
        params = [p for _, p in named_params]

        # 1) Probe clean forward: only used to build ESAM perturbation and SDS scores.
        #    Do not retain this graph for the final backward.
        probe_outputs, _ = self._forward_training_loss(wrapper, batch, capture_layer0=False)
        probe_loss = probe_outputs.loss
        if probe_loss is None:
            raise RuntimeError("esam_lora memory-safe probe forward failed to produce a base loss.")
        base_per_sample = self._compute_per_sample_losses(batch, probe_outputs).detach()
        probe_base_loss_value = float(probe_loss.detach().cpu().item())

        probe_grads = torch.autograd.grad(
            outputs=probe_loss,
            inputs=params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        ) if named_params else []
        perturb, grad_norm, swp_selected_count, swp_beta = self._build_swp_perturb(
            named_params,
            probe_grads,
            radius=float(getattr(self.args, "sam_rho", getattr(self.args, "rho_f", 0.0))),
            device=probe_loss.device,
            eps=float(getattr(self.args, "eps", 1e-12)),
        )
        del probe_outputs, probe_loss, probe_grads
        self._release_cuda_cache()

        # 2) Adversarial forward: compute replacement branch gradients only.
        if perturb:
            wrapper._apply_param_perturb(perturb, sign=1)
        try:
            adv_outputs, _ = self._forward_training_loss(wrapper, batch, capture_layer0=False)
            adv_per_sample = self._compute_per_sample_losses(batch, adv_outputs)
            sharpness_increase = (adv_per_sample.detach() - base_per_sample).detach()
            keep_ratio = max(0.0, min(1.0, float(getattr(self.args, "esam_keep_ratio", 0.5))))
            batch_size = int(base_per_sample.size(0))
            keep_count = batch_size if batch_size <= 1 else max(1, int(math.ceil(batch_size * keep_ratio)))
            if keep_count >= batch_size:
                adv_branch_loss = adv_per_sample.mean()
            else:
                selected = torch.topk(sharpness_increase, k=keep_count, largest=True).indices
                adv_branch_loss = adv_per_sample.index_select(0, selected).mean()
            adv_loss_value = float(adv_branch_loss.detach().cpu().item())
            adv_grads = torch.autograd.grad(
                outputs=adv_branch_loss,
                inputs=params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            ) if named_params else []
        finally:
            if perturb:
                wrapper._apply_param_perturb(perturb, sign=-1)
        self._branch_grad_map = self._grad_map(named_params, adv_grads)
        mean_sharpness_value = float(sharpness_increase.mean().detach().cpu().item())
        del adv_outputs, adv_per_sample, adv_branch_loss, adv_grads
        self._release_cuda_cache()

        # 3) Final clean forward: this is the loss that the main training loop will
        #    backward through, so fusion/task-head clean gradients are unchanged.
        final_outputs, _ = self._forward_training_loss(wrapper, batch, capture_layer0=False)
        base_loss = final_outputs.loss
        if base_loss is None:
            raise RuntimeError("esam_lora memory-safe final clean forward failed to produce a base loss.")
        clean_grads = torch.autograd.grad(
            outputs=base_loss,
            inputs=params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        ) if named_params else []
        self._clean_branch_grad_map = self._grad_map(named_params, clean_grads)

        logs = {
            "train_loss": float(base_loss.detach().cpu().item()),
            "task_loss": float(base_loss.detach().cpu().item()),
            "esam_base_loss": float(base_loss.detach().cpu().item()),
            "esam_probe_base_loss": probe_base_loss_value,
            "esam_adv_loss": adv_loss_value,
            "esam_selected_count": int(keep_count),
            "esam_selected_ratio": float(keep_count / max(1, batch_size)),
            "esam_grad_norm": float(grad_norm.detach().cpu().item()),
            "esam_swp_beta": float(swp_beta),
            "esam_swp_selected_tensors": float(swp_selected_count),
            "esam_mean_sharpness_increase": mean_sharpness_value,
            "esam_perturb_scope": 1.0 if perturb else 0.0,
            "esam_branch_grad_replacement_pending": float(len(self._branch_grad_map)),
            "sam_branch_only_update": 1.0,
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
        return {
            "esam_replaced_branch_grad_count": float(replaced),
            "sam_branch_only_update": 1.0,
        }
