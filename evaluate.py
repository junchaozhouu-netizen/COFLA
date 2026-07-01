from __future__ import annotations

import argparse
import json
import os
from typing import Any, Optional

from torch.utils.data import DataLoader

from datasets import (
    HatefulMemesCollator,
    HatefulMemesDataset,
    MMIMDbCollator,
    MMIMDbDataset,
    NLVR2Collator,
    NLVR2Dataset,
    ScienceQACollator,
    ScienceQADataset,
)
from methods import (
    COFLAMethod,
    DGLLoRAMethod,
    ESAMLoRAMethod,
    FastCOFLAMethod,
    MASAMLoRAMethod,
    MSAMLoRAMethod,
    SAMLoRAMethod,
    VanillaFTMethod,
    VanillaLoRAMethod,
)

# Robust import:
# Some code versions do not export build_model_wrapper in models/__init__.py.
# This evaluate.py first tries the package export, then models/model_factory.py,
# and finally falls back to a local lightweight selector.
try:
    from models import build_model_wrapper  # type: ignore
except Exception:
    try:
        from models.model_factory import build_model_wrapper  # type: ignore
    except Exception:
        def _infer_model_key(args: argparse.Namespace) -> str:
            values = [
                getattr(args, "model_name", ""),
                getattr(args, "model_family", ""),
                getattr(args, "model_path", ""),
            ]
            text = " ".join(str(v).lower() for v in values if v is not None)
            if "internvl" in text:
                return "internvl3_1b"
            if "llava-onevision" in text or "llava_onevision" in text:
                return "llava_onevision_qwen2_0_5b"
            if "qwen" in text:
                return "qwen"
            raise ImportError(
                "Cannot import build_model_wrapper and cannot infer model type. "
                "Please make sure models/model_factory.py exists or set model_name/model_family/model_path in config.json."
            )

        def build_model_wrapper(args: argparse.Namespace):  # type: ignore
            key = _infer_model_key(args)
            if key == "internvl3_1b":
                from models.internvl3_wrapper import InternVL3Wrapper

                args.model_name = "internvl3_1b"
                args.model_family = "internvl"
                return InternVL3Wrapper(args)
            if key == "llava_onevision_qwen2_0_5b":
                from models.llava_onevision_wrapper import LlavaOnevisionWrapper

                args.model_name = "llava_onevision_qwen2_0_5b"
                args.model_family = "llava_onevision"
                return LlavaOnevisionWrapper(args)

            from models.qwen25vl_wrapper import Qwen25VLWrapper

            # Keep original qwen model_name if it already exists.
            args.model_family = "qwen"
            return Qwen25VLWrapper(args)

from train import evaluate
from utils.io_utils import save_json, str2bool
from utils.logging_utils import setup_logger
from utils.seed import build_dataloader_generator, seed_worker, set_seed


METHOD_MAP = {
    "vanilla_ft": VanillaFTMethod,
    "vanilla_lora": VanillaLoRAMethod,
    "sam_lora": SAMLoRAMethod,
    "esam_lora": ESAMLoRAMethod,
    "msam_lora": MSAMLoRAMethod,
    "masam_lora": MASAMLoRAMethod,
    "dgl_lora": DGLLoRAMethod,
    "cofla": COFLAMethod,
    "fast_cofla": FastCOFLAMethod,
}


def remap_paths_in_obj(obj: Any, old_prefix: Optional[str], new_prefix: Optional[str]) -> Any:
    """Recursively replace an old server path prefix in config values.

    Example:
      <old_prefix>/...  ->  <new_prefix>/...

    This is useful when results were trained on another server and copied here.
    """
    if not old_prefix or not new_prefix or old_prefix == new_prefix:
        return obj
    if isinstance(obj, dict):
        return {k: remap_paths_in_obj(v, old_prefix, new_prefix) for k, v in obj.items()}
    if isinstance(obj, list):
        return [remap_paths_in_obj(v, old_prefix, new_prefix) for v in obj]
    if isinstance(obj, str):
        return obj.replace(old_prefix, new_prefix)
    return obj


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", type=str, required=True)
    p.add_argument("--method", type=str, required=True, choices=list(METHOD_MAP.keys()))
    p.add_argument("--dataset", type=str, default="hateful_memes")
    p.add_argument("--split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--checkpoint_type", type=str, default="best", choices=["best", "latest"])

    # Keep this name for compatibility with run_compute_fcf_*.py scripts.
    # In this fixed evaluate.py it controls both test and val geometry.
    p.add_argument("--test_compute_geometry", type=str2bool, default=False)

    # Optional direct path mapping, useful for standalone evaluate.py calls.
    # The batch scripts already patch config.json before calling evaluate.py,
    # but these flags make evaluate.py itself robust too.
    p.add_argument("--path_map_old", type=str, default=None)
    p.add_argument("--path_map_new", type=str, default=None)
    return p.parse_args()


def get_attr(args: argparse.Namespace, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def main():
    cli = parse_args()
    config_path = os.path.join(cli.exp_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Optional safety remap. This does not modify config.json on disk.
    config = remap_paths_in_obj(config, cli.path_map_old, cli.path_map_new)

    args = argparse.Namespace(**{k: v for k, v in config.items() if k != "runtime"})
    args.method = cli.method
    args.dataset = cli.dataset

    # Important fix:
    # train.evaluate usually checks val_compute_geometry for val split and
    # test_compute_geometry for test split. The old evaluate.py only set
    # test_compute_geometry, so val geometry could silently not be computed.
    args.test_compute_geometry = cli.test_compute_geometry
    args.val_compute_geometry = cli.test_compute_geometry

    if not hasattr(args, "deterministic"):
        args.deterministic = True
    if not hasattr(args, "num_workers"):
        args.num_workers = 0
    if not hasattr(args, "per_device_eval_batch_size"):
        args.per_device_eval_batch_size = 1
    if not hasattr(args, "seed"):
        args.seed = 42

    if args.dataset == "mmimdb":
        if str(getattr(args, "task_format", "")).lower() != "task_head_cls":
            raise ValueError("MMIMDb must use --task_format task_head_cls because it is a multi-label task.")
        label_names = MMIMDbDataset.parse_label_names(getattr(args, "mmimdb_label_names", ""))
        if not label_names:
            label_names = MMIMDbDataset.infer_label_names(args.data_root, split_file=args.mmimdb_split_file)
            args.mmimdb_label_names = ",".join(label_names)
        if not label_names:
            raise ValueError(f"Could not infer MMIMDb label names from {args.data_root}")
        args.num_labels = len(label_names)

    set_seed(args.seed, deterministic=args.deterministic)
    logger = setup_logger(os.path.join(cli.exp_dir, "logs", f"evaluate_{cli.split}.log"), name=f"eval_{cli.split}")

    # Do not hard-code Qwen25VLWrapper here.
    # build_model_wrapper(args) should select Qwen/InternVL/LLaVA according to config.json.
    wrapper = build_model_wrapper(args)
    wrapper.apply_method_setup(args.method)

    ckpt_dir = os.path.join(cli.exp_dir, "checkpoints", cli.checkpoint_type)
    ckpt_path = os.path.join(ckpt_dir, "trainable_state.pt")
    checkpoint_mode = str(getattr(args, "checkpoint_mode", "both")).lower()
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint '{cli.checkpoint_type}' not found at {ckpt_path}. "
            f"This experiment was trained with checkpoint_mode='{checkpoint_mode}'. "
            "Re-run training with --checkpoint_mode best/latest/both if you need standalone evaluation."
        )
    load_info = wrapper.load_trainable_checkpoint(ckpt_dir, strict=False)
    logger.info(f"Loaded checkpoint: {load_info}")

    method = METHOD_MAP[args.method](args)

    max_val_samples = get_attr(args, "max_val_samples", None)
    max_test_samples = get_attr(args, "max_test_samples", None)
    max_samples = max_val_samples if cli.split == "val" else max_test_samples

    if args.dataset == "hateful_memes":
        split_name = args.hm_val_split if cli.split == "val" else args.hm_test_split
        ds = HatefulMemesDataset(args.data_root, split_name, max_samples=max_samples, seed=args.seed)
        collate_fn = HatefulMemesCollator(wrapper, split=cli.split)
    elif args.dataset == "nlvr2":
        split_name = args.nlvr2_val_split if cli.split == "val" else args.nlvr2_test_split
        ds = NLVR2Dataset(
            args.data_root,
            split_name,
            max_samples=max_samples,
            seed=args.seed,
            variant=args.nlvr2_variant,
        )
        collate_fn = NLVR2Collator(wrapper, split=cli.split, data_root=args.data_root)
    elif args.dataset == "scienceqa":
        split_name = args.scienceqa_val_split if cli.split == "val" else args.scienceqa_test_split
        ds = ScienceQADataset(args.data_root, split_name, max_samples=max_samples, seed=args.seed)
        collate_fn = ScienceQACollator(
            wrapper,
            split=cli.split,
            data_root=args.data_root,
            include_hint=getattr(args, "scienceqa_include_hint", True),
        )
    elif args.dataset == "mmimdb":
        split_name = args.mmimdb_val_split if cli.split == "val" else args.mmimdb_test_split
        ds = MMIMDbDataset(
            args.data_root,
            split_name,
            max_samples=max_samples,
            seed=args.seed,
            label_names=MMIMDbDataset.parse_label_names(args.mmimdb_label_names),
            split_file=args.mmimdb_split_file,
        )
        collate_fn = MMIMDbCollator(wrapper, split=cli.split, data_root=args.data_root)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    loader = DataLoader(
        ds,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=build_dataloader_generator(args.seed + 1),
        persistent_workers=args.num_workers > 0,
    )

    metrics, preds = evaluate(wrapper, method, loader, args, split=cli.split, logger=logger)
    logger.info(f"{cli.split} metrics: {metrics}")
    save_json(
        {"metrics": metrics, "predictions": preds},
        os.path.join(cli.exp_dir, "predictions", f"{cli.split}_predictions.json"),
    )


if __name__ == "__main__":
    main()
