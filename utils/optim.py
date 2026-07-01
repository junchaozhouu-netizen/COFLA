from __future__ import annotations

from typing import Iterable

import torch
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

try:
    import bitsandbytes as bnb
except Exception:
    bnb = None


class NullScheduler:
    def step(self):
        return None



def build_optimizer(args, model):
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer.lower() == "adamw":
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer.lower() == "adamw8bit":
        if bnb is None:
            raise ImportError(
                "Optimizer 'adamw8bit' requires bitsandbytes to be installed. "
                "Install bitsandbytes or switch to --optimizer adamw."
            )
        return bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer.lower() == "sgd":
        return torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")



def build_scheduler(args, optimizer, num_training_steps: int):
    if args.scheduler.lower() == "none":
        return NullScheduler()
    warmup_steps = int(num_training_steps * args.warmup_ratio)
    if args.scheduler.lower() == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)
    if args.scheduler.lower() == "linear":
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")



def compute_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.detach().float().norm(2)
        total += param_norm.item() ** 2
    return float(total ** 0.5)
