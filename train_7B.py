#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2-VL-7B safe entrypoint for COFLA experiments.

Place this file at the repository root next to train.py.
It does NOT modify train.py or other existing files. It only adds a qwen2vl7b alias
at runtime and supplies conservative 24GB-friendly defaults when a flag is omitted.

Recommended call pattern:
  python train_7B.py --device 0 --method vanilla_lora --dataset hateful_memes ...
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
# This file may be placed at the repository root or inside a helper subdirectory.
# Prefer COFLA_PROJECT_ROOT if set;
# otherwise infer the project root by looking for train.py.
PROJECT_ROOT = Path(os.environ.get("COFLA_PROJECT_ROOT", str(THIS_FILE.parent))).resolve()
if not (PROJECT_ROOT / "train.py").exists() and (THIS_FILE.parent.parent / "train.py").exists():
    PROJECT_ROOT = THIS_FILE.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_MODEL_PATH = "./external_models/qwen2_vl_7b_instruct"


def _add_default(argv: list[str], flag: str, value: str) -> None:
    if flag not in argv:
        argv.extend([flag, str(value)])


def _patch_qwen2vl7b_alias() -> None:
    """Add a local qwen2vl7b alias without editing models/model_factory.py."""
    try:
        import models.model_factory as mf
    except Exception as exc:
        raise RuntimeError(
            "Could not import models.model_factory. Run train_7B.py from the COFLA project root "
            "or place it next to train.py."
        ) from exc

    mf.MODEL_ALIASES.update({
        "qwen2vl7b": "qwen2vl7b",
        "qwen2_vl_7b": "qwen2vl7b",
        "qwen2-vl-7b": "qwen2vl7b",
        "qwen2-vl-7b-instruct": "qwen2vl7b",
        "qwen2vl7b-instruct": "qwen2vl7b",
    })
    mf.MODEL_FAMILY["qwen2vl7b"] = "qwen"


def _patch_7b_datasets() -> None:
    """Route only Qwen2-VL-7B runs to 7B-specific NLVR2/ScienceQA datasets.

    The original files remain untouched:
      datasets/base.py
      datasets/nlvr2.py
      datasets/scienceqa.py

    This patch modifies only the in-memory attributes imported by train.py when
    this 7B entrypoint is used. 0.5B/2B/control-late-fusion runs that call the
    original train.py keep using the original dataset classes.
    """
    try:
        import datasets as ds
        from datasets.nlvr2_7B import NLVR2Dataset as NLVR2Dataset7B
        from datasets.nlvr2_7B import NLVR2Collator as NLVR2Collator7B
        from datasets.scienceqa_7B import ScienceQADataset as ScienceQADataset7B
        from datasets.scienceqa_7B import ScienceQACollator as ScienceQACollator7B
    except Exception as exc:
        raise RuntimeError(
            "Could not import 7B-specific dataset files. Make sure these files exist under "
            "the COFLA project root: datasets/base_7B.py, datasets/nlvr2_7B.py, "
            "datasets/scienceqa_7B.py."
        ) from exc

    ds.NLVR2Dataset = NLVR2Dataset7B
    ds.NLVR2Collator = NLVR2Collator7B
    ds.ScienceQADataset = ScienceQADataset7B
    ds.ScienceQACollator = ScienceQACollator7B


def main() -> None:
    if not (PROJECT_ROOT / "train.py").exists():
        raise RuntimeError(
            f"Cannot find train.py under inferred COFLA project root: {PROJECT_ROOT}. "
            "Place this helper code inside the project tree or set COFLA_PROJECT_ROOT to the project root.."
        )
    os.chdir(PROJECT_ROOT)

    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.8",
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Make qwen2vl7b usable as --model_name without touching existing files.
    _patch_qwen2vl7b_alias()

    # Use 7B-only robust NLVR2/ScienceQA dataset classes without modifying original dataset files.
    _patch_7b_datasets()

    argv = sys.argv

    # Conservative defaults for a single 24GB GPU. Explicit command-line flags override these.
    _add_default(argv, "--device", "0")
    _add_default(argv, "--model_name", "qwen2vl7b")
    _add_default(argv, "--model_family", "qwen")
    _add_default(argv, "--model_path", DEFAULT_MODEL_PATH)
    _add_default(argv, "--local_files_only", "true")
    _add_default(argv, "--trust_remote_code", "true")

    _add_default(argv, "--precision", "fp16")
    _add_default(argv, "--quantization", "4bit")
    _add_default(argv, "--bnb_4bit_quant_type", "nf4")
    _add_default(argv, "--bnb_4bit_use_double_quant", "true")
    _add_default(argv, "--gradient_checkpointing", "true")
    _add_default(argv, "--gradient_checkpointing_use_reentrant", "auto")

    _add_default(argv, "--per_device_train_batch_size", "1")
    _add_default(argv, "--per_device_eval_batch_size", "1")
    _add_default(argv, "--gradient_accumulation_steps", "8")
    _add_default(argv, "--num_workers", "0")

    # Smaller LoRA/scope than the 2B runs to reduce trainable states and optimizer memory.
    _add_default(argv, "--trainable_scope", "qwen_matched_lora")
    _add_default(argv, "--lora_r", "4")
    _add_default(argv, "--lora_alpha", "8")
    _add_default(argv, "--lora_dropout", "0.05")
    _add_default(argv, "--bias", "none")
    _add_default(argv, "--num_fusion_layers", "2")
    _add_default(argv, "--num_lm_fusion_layers", "2")
    _add_default(argv, "--num_branch_vision_layers", "1")
    _add_default(argv, "--num_branch_lm_layers", "1")

    # Avoid validation/test geometry on 7B during main task runs; it is memory-expensive.
    _add_default(argv, "--val_compute_geometry", "false")
    _add_default(argv, "--test_compute_geometry", "false")

    # Let PyTorch/Transformers choose memory-efficient attention when available.
    _add_default(argv, "--deterministic", "false")

    import train
    train.main()


if __name__ == "__main__":
    main()
