from __future__ import annotations


# VLM-online-aligned: VLM PEFT baseline: trains the matched branch-side LoRA and fusion-sensitive LoRA scope through wrapper.compute_base_loss().
from .base_method import BaseMethod, StepOutput


class VanillaLoRAMethod(BaseMethod):
    name = "vanilla_lora"

    def training_step(self, wrapper, batch) -> StepOutput:
        outputs, _ = wrapper.compute_base_loss(batch, capture_layer0=False)
        loss = outputs.loss
        return StepOutput(loss=loss, logs={"train_loss": float(loss.detach().cpu().item())})
