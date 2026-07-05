#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect the Qwen2-VL-7B matched-LoRA scope used by train_7B.py.
This builds the architecture on the meta device when possible; it should not load full weights.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="./external_models/qwen2_vl_7b_instruct")
    ap.add_argument("--num_fusion_layers", type=int, default=2)
    ap.add_argument("--num_lm_fusion_layers", type=int, default=2)
    ap.add_argument("--num_branch_vision_layers", type=int, default=1)
    ap.add_argument("--num_branch_lm_layers", type=int, default=1)
    ap.add_argument("--out", default="./outputs/qwen2vl7b_scope_train7B.txt")
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig
    try:
        from accelerate import init_empty_weights
    except Exception:
        init_empty_weights = None

    from models.trainable_scope import resolve_trainable_scope

    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    ctx = init_empty_weights() if init_empty_weights is not None else torch.device("meta")
    with ctx:
        try:
            from transformers import Qwen2VLForConditionalGeneration
            model = Qwen2VLForConditionalGeneration(cfg)
        except Exception:
            from transformers import AutoModelForVision2Seq
            model = AutoModelForVision2Seq.from_config(cfg, trust_remote_code=True)

    scope = resolve_trainable_scope(
        model,
        model_family="qwen",
        trainable_scope="qwen_matched_lora",
        num_fusion_layers=args.num_fusion_layers,
        num_lm_fusion_layers=args.num_lm_fusion_layers,
        num_branch_vision_layers=args.num_branch_vision_layers,
        num_branch_lm_layers=args.num_branch_lm_layers,
    )

    lines = []
    lines.append("Qwen2-VL-7B train_7B.py matched-LoRA scope")
    lines.append("=" * 80)
    lines.append(f"model_path: {Path(args.model_path)}")
    lines.append(f"scope_name: {scope.scope_name}")
    lines.append(f"notes: {scope.notes}")
    lines.append("")
    lines.append(f"fusion_module_names ({len(scope.fusion_module_names)}):")
    lines.extend(f"  {x}" for x in scope.fusion_module_names)
    lines.append("")
    lines.append(f"branch_visual_module_names ({len(scope.branch_visual_module_names)}):")
    lines.extend(f"  {x}" for x in scope.branch_visual_module_names)
    lines.append("")
    lines.append(f"branch_text_module_names ({len(scope.branch_text_module_names)}):")
    lines.extend(f"  {x}" for x in scope.branch_text_module_names)
    lines.append("")
    lines.append(f"lora_target_modules ({len(scope.lora_target_modules)}):")
    lines.extend(f"  {x}" for x in scope.lora_target_modules)
    text = "\n".join(lines) + "\n"
    print(text)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"[OK] wrote {args.out}")


if __name__ == "__main__":
    main()
