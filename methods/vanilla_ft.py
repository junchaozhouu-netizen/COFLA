from __future__ import annotations


# VLM-online-aligned: Legacy/full-tuning baseline entry. It is kept for CLI compatibility; VLM PEFT experiments should normally use vanilla_lora.
from .base_method import BaseMethod, StepOutput


class VanillaFTMethod(BaseMethod):
    name = "vanilla_ft"

    def training_step(self, wrapper, batch) -> StepOutput:
        outputs, _ = wrapper.compute_base_loss(batch, capture_layer0=False)
        loss = outputs.loss
        return StepOutput(loss=loss, logs={"train_loss": float(loss.detach().cpu().item())})
