#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize COFLA grid-search results under the seven-method setting.

COFLA is the only proposed training method. FAST-FCF / FAST-RFCF are geometry
metrics parsed for every method, not optimization algorithms.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This script requires pandas. Please install it with `pip install pandas`.") from exc

DEFAULT_METHODS = [
    "cofla",
    "vanilla_lora",
    "sam_lora",
    "esam_lora",
    "msam_lora",
    "masam_lora",
    "dgl_lora",
]
PROPOSED_METHOD = "cofla"

METHOD_ALIASES = {
    "cofla": "cofla",
    "vanilla": "vanilla_lora",
    "vanilla_lora": "vanilla_lora",
    "vanilla lora": "vanilla_lora",
    "sam": "sam_lora",
    "sam_lora": "sam_lora",
    "sam lora": "sam_lora",
    "esam": "esam_lora",
    "esam_lora": "esam_lora",
    "esam lora": "esam_lora",
    "msam": "msam_lora",
    "m_sam": "msam_lora",
    "m sam": "msam_lora",
    "m-sam": "msam_lora",
    "msam_lora": "msam_lora",
    "m-sam-lora": "msam_lora",
    "masam": "masam_lora",
    "masam_lora": "masam_lora",
    "dgl": "dgl_lora",
    "dgl_lora": "dgl_lora",
    "dgl lora": "dgl_lora",
}

SUMMARY_KEYS = {
    "dataset": "dataset",
    "visual branch": "visual_branch",
    "text branch": "text_branch",
    "encoder setup": "encoder_setup",
    "fusion type": "fusion_type",
    "method": "method",
    "seed": "seed",
    "best epoch": "best_epoch",
    "auroc": "test_auroc",
    "accuracy": "test_accuracy",
    "macro-f1": "test_macro_f1",
    "macro f1": "test_macro_f1",
    "micro-f1": "test_micro_f1",
    "micro f1": "test_micro_f1",
    "mean fast s v": "mean_fast_s_v",
    "mean fast s t": "mean_fast_s_t",
    "mean fast s branch": "mean_fast_s_branch",
    "mean fast s f": "mean_fast_s_f",
    "mean fast fcf": "mean_fast_fcf",
    "mean fast rfcf": "mean_fast_rfcf",
    "mean exact s v": "mean_exact_s_v",
    "mean exact s t": "mean_exact_s_t",
    "mean exact s branch": "mean_exact_s_branch",
    "mean exact s f": "mean_exact_s_f",
    "mean exact fcf": "mean_exact_fcf",
    "mean exact rfcf": "mean_exact_rfcf",
    "trainable parameter count": "trainable_params",
    "peak gpu memory mb": "peak_gpu_memory_mb",
}

FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def normalize_text_key(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower().replace("_", " "))


def normalize_method(s: Any) -> str | None:
    if s is None:
        return None
    raw = str(s).strip()
    key = normalize_text_key(raw).replace("-", "_")
    key_space = normalize_text_key(raw).replace("_", " ")
    return METHOD_ALIASES.get(key) or METHOD_ALIASES.get(key_space) or raw.lower()


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    m = FLOAT_RE.search(str(value).replace(",", ""))
    if not m:
        return None
    try:
        x = float(m.group(0))
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def parse_int(value: Any) -> int | None:
    x = parse_float(value)
    return None if x is None else int(round(x))


def flatten_json(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{prefix}.{k}" if prefix else str(k)
            out[new_key] = v
            out.update(flatten_json(v, new_key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten_json(v, f"{prefix}[{i}]"))
    return out


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        return flatten_json(json.loads(path.read_text(encoding="utf-8", errors="replace")))
    except Exception:
        return {}


def normalized_json_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", key.lower())


def find_json_value(flat: dict[str, Any], candidate_keys: Iterable[str]) -> Any | None:
    wanted = {normalized_json_key(k) for k in candidate_keys}
    for k, v in flat.items():
        tail = k.split(".")[-1]
        if normalized_json_key(tail) in wanted:
            return v
    for k, v in flat.items():
        kk = normalized_json_key(k)
        if any(w in kk for w in wanted):
            return v
    return None


def read_auxiliary_json(summary_path: Path) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    candidates = [
        summary_path.parent / "config.json",
        summary_path.parent / "final_metrics.json",
        summary_path.parent / "metrics" / "final_metrics.json",
        summary_path.parent / "metrics" / "metrics.json",
    ]
    for path in candidates:
        flat.update(load_json_if_exists(path))
    return flat


def parse_summary_txt(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"best validation metric\s*\(([^)]+)\)\s*:\s*([^\n\r]+)", text, re.I)
    if m:
        data["val_main_metric_name"] = m.group(1).strip()
        data["val_main_metric"] = parse_float(m.group(2))
    m = re.search(r"test metric\s*\(([^)]+)\)\s*:\s*([^\n\r]+)", text, re.I)
    if m:
        data["test_main_metric_name"] = m.group(1).strip()
        data["test_main_metric"] = parse_float(m.group(2))
    m = re.search(r"total wall-clock time\s*:\s*([^\(\n\r]+)\s*\(([^\)]+)\)", text, re.I)
    if m:
        data["wall_clock_hms"] = m.group(1).strip()
        data["wall_clock_seconds"] = parse_float(m.group(2))
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        out_key = SUMMARY_KEYS.get(normalize_text_key(key))
        if not out_key:
            continue
        if out_key in {"seed", "best_epoch", "trainable_params"}:
            data[out_key] = parse_int(value)
        elif out_key.startswith("mean_") or out_key.startswith("test_") or out_key.endswith("_mb"):
            data[out_key] = parse_float(value)
        else:
            data[out_key] = value.strip()
    if "method" in data:
        data["method"] = normalize_method(data["method"])
    dataset = str(data.get("dataset", "")).lower()
    if "val_main_metric_name" not in data:
        data["val_main_metric_name"] = "AUROC" if "hateful" in dataset else "Macro-F1"
    if "test_main_metric_name" not in data:
        data["test_main_metric_name"] = "AUROC" if "hateful" in dataset else "Macro-F1"
    if data.get("test_main_metric") is None:
        metric_name = str(data.get("test_main_metric_name", "")).lower()
        if "auroc" in metric_name:
            data["test_main_metric"] = data.get("test_auroc")
        elif "macro" in metric_name:
            data["test_main_metric"] = data.get("test_macro_f1")
        elif "micro" in metric_name:
            data["test_main_metric"] = data.get("test_micro_f1")
        else:
            data["test_main_metric"] = data.get("test_accuracy")
    return data


def decode_scientific_token(token: str) -> float | None:
    s = token.strip().lower().replace("p", ".")
    s = s.replace("em", "e-")
    try:
        return float(s)
    except ValueError:
        return parse_float(s)


def parse_grid_from_path(path: Path) -> dict[str, Any]:
    text = "/".join(path.parts)
    out: dict[str, Any] = {}
    m = re.search(r"(?:^|[/_\-])g(?:rid)?[_\-]?0*([0-9]{1,3})(?:[/_\-]|$)", text, re.I)
    if m:
        out["grid_id"] = f"g{int(m.group(1)):03d}"
    patterns = {
        "rho_v": [r"(?:rv|rho[_\-]?v)[_\-]?([0-9]+(?:p[0-9]+)?(?:em[0-9]+|e[-+]?\d+)?)"],
        "rho_t": [r"(?:rt|rho[_\-]?t)[_\-]?([0-9]+(?:p[0-9]+)?(?:em[0-9]+|e[-+]?\d+)?)"],
        "K_cal": [r"(?:k|kcal|k_cal|calib|rho[_\-]?f[_\-]?calib[_\-]?batches)[_\-]?([0-9]+)"],
    }
    for key, pats in patterns.items():
        for pat in pats:
            m = re.search(pat, text, re.I)
            if m:
                out[key] = parse_int(m.group(1)) if key == "K_cal" else decode_scientific_token(m.group(1))
                break
    return out


def infer_grid_id(rho_v: Any, rho_t: Any, k_cal: Any) -> str:
    def fmt(x: Any) -> str:
        xf = parse_float(x)
        return "NA" if xf is None else f"{xf:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    k = parse_int(k_cal)
    return f"rv{fmt(rho_v)}_rt{fmt(rho_t)}_k{k if k is not None else 'NA'}"


def collect_summary_files(result_dir: Path) -> list[Path]:
    files = sorted(result_dir.glob("**/summary.txt"))
    if files:
        return files
    return [p for p in sorted(result_dir.glob("**/*summary*.txt")) if not p.name.lower().startswith("grid_")]


def build_result_row(summary_path: Path, result_dir: Path) -> dict[str, Any]:
    row = parse_summary_txt(summary_path)
    aux = read_auxiliary_json(summary_path)
    path_grid = parse_grid_from_path(summary_path)
    row.update({"summary_path": str(summary_path), "result_path": str(summary_path.parent), "exp_name": summary_path.parent.name})
    for key, candidates in {
        "dataset": ["dataset"],
        "fusion_type": ["fusion_type"],
        "method": ["method"],
        "rho_v": ["rho_v", "eval_rho_v"],
        "rho_t": ["rho_t", "eval_rho_t"],
        "rho_f": ["rho_f"],
        "K_cal": ["rho_f_calib_batches", "K_cal", "k_cal"],
        "rho_f_calib_stat": ["rho_f_calib_stat"],
        "auto_rho_f": ["auto_rho_f"],
        "eval_rho_v": ["eval_rho_v"],
        "eval_rho_t": ["eval_rho_t"],
        "eval_rho_f": ["eval_rho_f"],
        "geometry_metric": ["geometry_metric"],
        "rho_f_before_calibration": ["rho_f_before_calibration", "rho_f_before"],
        "rho_f_after_calibration": ["rho_f_after_calibration", "rho_f_after", "calibrated_rho_f"],
        "val_main_metric": ["best_val_metrics.auroc", "best_val_metrics.macro_f1", "val_main_metric"],
        "test_main_metric": ["test_metrics.auroc", "test_metrics.macro_f1", "test_main_metric"],
        "test_auroc": ["test_metrics.auroc", "auroc"],
        "test_accuracy": ["test_metrics.accuracy", "accuracy"],
        "test_macro_f1": ["test_metrics.macro_f1", "macro_f1"],
        "test_micro_f1": ["test_metrics.micro_f1", "micro_f1"],
        "mean_fast_s_v": ["best_val_metrics.fast_s_v", "fast_s_v"],
        "mean_fast_s_t": ["best_val_metrics.fast_s_t", "fast_s_t"],
        "mean_fast_s_branch": ["best_val_metrics.fast_s_branch", "fast_s_branch"],
        "mean_fast_s_f": ["best_val_metrics.fast_s_f", "fast_s_f"],
        "mean_fast_fcf": ["best_val_metrics.fast_fcf", "fast_fcf"],
        "mean_fast_rfcf": ["best_val_metrics.fast_rfcf", "fast_rfcf"],
        "mean_exact_fcf": ["best_val_metrics.exact_fcf", "exact_fcf"],
        "mean_exact_rfcf": ["best_val_metrics.exact_rfcf", "exact_rfcf"],
    }.items():
        if row.get(key) is None:
            value = find_json_value(aux, candidates)
            if value is not None:
                row[key] = value
    for key in ["rho_v", "rho_t", "rho_f", "eval_rho_v", "eval_rho_t", "eval_rho_f", "rho_f_before_calibration", "rho_f_after_calibration", "val_main_metric", "test_main_metric", "test_auroc", "test_accuracy", "test_macro_f1", "test_micro_f1", "mean_fast_s_v", "mean_fast_s_t", "mean_fast_s_branch", "mean_fast_s_f", "mean_fast_fcf", "mean_fast_rfcf", "mean_exact_fcf", "mean_exact_rfcf", "wall_clock_seconds", "peak_gpu_memory_mb"]:
        if key in row:
            row[key] = parse_float(row.get(key))
    for key in ["K_cal", "seed", "best_epoch", "trainable_params"]:
        if key in row:
            row[key] = parse_int(row.get(key))
    for key in ["rho_v", "rho_t", "K_cal", "grid_id"]:
        if row.get(key) is None and path_grid.get(key) is not None:
            row[key] = path_grid[key]
    if not row.get("grid_id"):
        row["grid_id"] = infer_grid_id(row.get("rho_v"), row.get("rho_t"), row.get("K_cal"))
    if not row.get("method"):
        ptxt = str(summary_path).lower()
        for m in DEFAULT_METHODS:
            if m in ptxt:
                row["method"] = m
                break
    row["method"] = normalize_method(row.get("method"))
    row["status"] = "parsed"
    return row


def choose_main_metric(row: dict[str, Any], split: str) -> float | None:
    return parse_float(row.get("val_main_metric" if split == "val" else "test_main_metric"))


def make_group_comparison(df: pd.DataFrame, methods: list[str], split: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    group_cols = [c for c in ["dataset", "fusion_type", "grid_id"] if c in df.columns]
    rows: list[dict[str, Any]] = []
    for group_values, gdf in df.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        grow: dict[str, Any] = dict(zip(group_cols, group_values))
        preferred = gdf[gdf.get("method").eq(PROPOSED_METHOD)] if "method" in gdf.columns else gdf
        if preferred.empty:
            preferred = gdf
        for meta_key in ["rho_v", "rho_t", "K_cal"]:
            vals = preferred[meta_key].dropna().tolist() if meta_key in preferred.columns else []
            if not vals and meta_key in gdf.columns:
                vals = gdf[meta_key].dropna().tolist()
            grow[meta_key] = vals[0] if vals else None
        method_rows = {str(r["method"]): r for _, r in gdf.iterrows() if pd.notna(r.get("method"))}
        for method in methods:
            r = method_rows.get(method)
            if r is None:
                grow[f"{method}_present"] = False
                continue
            grow[f"{method}_present"] = True
            grow[f"{method}_main"] = choose_main_metric(r.to_dict(), split)
            grow[f"{method}_val_main"] = parse_float(r.get("val_main_metric"))
            grow[f"{method}_test_main"] = parse_float(r.get("test_main_metric"))
            grow[f"{method}_auroc"] = parse_float(r.get("test_auroc"))
            grow[f"{method}_accuracy"] = parse_float(r.get("test_accuracy"))
            grow[f"{method}_macro_f1"] = parse_float(r.get("test_macro_f1"))
            grow[f"{method}_micro_f1"] = parse_float(r.get("test_micro_f1"))
            grow[f"{method}_fcf_fastmetric"] = parse_float(r.get("mean_fast_fcf"))
            grow[f"{method}_rfcf_fastmetric"] = parse_float(r.get("mean_fast_rfcf"))
            grow[f"{method}_fast_s_v"] = parse_float(r.get("mean_fast_s_v"))
            grow[f"{method}_fast_s_t"] = parse_float(r.get("mean_fast_s_t"))
            grow[f"{method}_fast_s_branch"] = parse_float(r.get("mean_fast_s_branch"))
            grow[f"{method}_fast_s_f"] = parse_float(r.get("mean_fast_s_f"))
            grow[f"{method}_wall_clock_seconds"] = parse_float(r.get("wall_clock_seconds"))
            grow[f"{method}_peak_gpu_memory_mb"] = parse_float(r.get("peak_gpu_memory_mb"))
        cofla_main = parse_float(grow.get("cofla_main"))
        vanilla_fast_fcf = parse_float(grow.get("vanilla_lora_fcf_fastmetric"))
        cofla_fcf_fastmetric = parse_float(grow.get("cofla_fcf_fastmetric"))
        other_methods = [m for m in methods if m != PROPOSED_METHOD]
        best_other_method = None
        best_other_main = None
        for m in other_methods:
            val = parse_float(grow.get(f"{m}_main"))
            if val is not None and (best_other_main is None or val > best_other_main):
                best_other_main = val
                best_other_method = m
        grow["best_other_method"] = best_other_method
        grow["best_other_main"] = best_other_main
        if cofla_main is not None and best_other_main is not None:
            grow["cofla_margin_vs_best_other"] = cofla_main - best_other_main
        else:
            grow["cofla_margin_vs_best_other"] = None
        if vanilla_fast_fcf is not None and cofla_fcf_fastmetric is not None:
            grow["fast_fcf_improvement_vs_vanilla"] = vanilla_fast_fcf - cofla_fcf_fastmetric
            grow["fast_fcf_reduction_pct_vs_vanilla"] = 100.0 * (vanilla_fast_fcf - cofla_fcf_fastmetric) / vanilla_fast_fcf if vanilla_fast_fcf != 0 else None
        else:
            grow["fast_fcf_improvement_vs_vanilla"] = None
            grow["fast_fcf_reduction_pct_vs_vanilla"] = None
        margin = parse_float(grow.get("cofla_margin_vs_best_other")) or 0.0
        geometry_gain = parse_float(grow.get("fast_fcf_improvement_vs_vanilla")) or 0.0
        grow["selection_score"] = 100.0 * margin + 0.01 * geometry_gain
        grow["candidate_cofla_better"] = bool(margin > 0)
        grow["candidate_fast_fcf_better_than_vanilla"] = bool(geometry_gain > 0)
        grow["num_methods_present"] = sum(bool(grow.get(f"{m}_present")) for m in methods)
        rows.append(grow)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(by=["selection_score", "cofla_margin_vs_best_other"], ascending=[False, False], na_position="last").reset_index(drop=True)
        out.insert(0, "rank", range(1, len(out) + 1))
    return out


def write_report(result_dir: Path, all_df: pd.DataFrame, group_df: pd.DataFrame, best_df: pd.DataFrame, output_path: Path, split: str, methods: list[str], expected_groups: int | None) -> None:
    lines: list[str] = []
    lines.append("Grid grid-search selection report")
    lines.append("=" * 80)
    lines.append(f"Result dir: {result_dir}")
    lines.append(f"Selection split: {split}")
    lines.append(f"Parsed runs: {len(all_df)}")
    lines.append(f"Parsed groups: {len(group_df)}")
    if not group_df.empty and "num_methods_present" in group_df.columns:
        complete_groups = int((group_df["num_methods_present"] == len(methods)).sum())
        lines.append(f"Complete {len(methods)}-method groups: {complete_groups}")
    if expected_groups:
        lines.append(f"Expected groups: {expected_groups}")
        lines.append(f"Expected runs for {len(methods)} methods: {expected_groups * len(methods)}")
    lines.append("")
    if not all_df.empty:
        lines.append("Method counts:")
        for method, count in all_df["method"].value_counts(dropna=False).sort_index().items():
            lines.append(f"  {method}: {count}")
        lines.append("")
        lines.append("Missing-value check:")
        for k in ["val_main_metric", "test_main_metric", "mean_fast_fcf", "mean_fast_rfcf", "mean_fast_s_v", "mean_fast_s_t", "mean_fast_s_branch", "mean_fast_s_f"]:
            lines.append(f"  {k}: {int(all_df[k].isna().sum()) if k in all_df else len(all_df)} missing")
        lines.append("")
    lines.append("Top configurations by selection_score:")
    lines.append("selection_score rewards COFLA validation margin over the strongest baseline and lower COFLA FAST-FCF than vanilla.")
    lines.append("")
    if best_df.empty:
        lines.append("  No complete groups found.")
    else:
        cols = ["rank", "dataset", "fusion_type", "grid_id", "rho_v", "rho_t", "K_cal", "cofla_main", "best_other_method", "best_other_main", "cofla_margin_vs_best_other", "cofla_fcf_fastmetric", "vanilla_lora_fcf_fastmetric", "fast_fcf_reduction_pct_vs_vanilla", "selection_score"]
        for _, r in best_df.head(20).iterrows():
            parts = []
            for c in cols:
                if c not in best_df.columns:
                    continue
                val = r.get(c)
                if isinstance(val, float):
                    val = "" if math.isnan(val) else f"{val:.6g}"
                parts.append(f"{c}={val}")
            lines.append("  " + " | ".join(parts))
    lines.append("")
    lines.append("Notes:")
    lines.append("  - COFLA is the only proposed optimizer in this 7-method setting.")
    lines.append("  - FAST-FCF / FAST-RFCF are geometry metrics evaluated under one protocol for all methods.")
    lines.append("  - For Hateful Memes, the main metric is AUROC; for MM-IMDb, it is Macro-F1.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize seven-method COFLA grid-search results.")
    parser.add_argument("--result_dir", required=True, type=Path, help="Directory containing experiment summary text files.")
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Which main metric split to use for selecting best configs. Default: val")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS), help="Comma-separated expected method list.")
    parser.add_argument("--top_k", type=int, default=8, help="Number of best configurations to write to the best-config CSV file")
    parser.add_argument("--expected_groups", type=int, default=8, help="Expected number of rho_v/rho_t/K_cal groups per result_dir.")
    parser.add_argument("--output_prefix", default="cofla_grid", help="Output prefix for generated CSV summaries.")
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    if not result_dir.exists():
        raise SystemExit(f"Result directory does not exist: {result_dir}")
    methods = [normalize_method(m.strip()) for m in args.methods.split(",") if m.strip()]
    methods = [m for m in methods if m]
    summary_files = collect_summary_files(result_dir)
    if not summary_files:
        raise SystemExit(f"No summary.txt files found under {result_dir}.")
    rows = [build_result_row(p, result_dir) for p in summary_files]
    all_df = pd.DataFrame(rows)
    preferred_cols = [
        "grid_id", "rho_v", "rho_t", "K_cal", "rho_f", "auto_rho_f", "rho_f_calib_stat", "rho_f_before_calibration", "rho_f_after_calibration", "eval_rho_v", "eval_rho_t", "eval_rho_f", "geometry_metric", "dataset", "visual_branch", "text_branch", "encoder_setup", "fusion_type", "method", "seed", "best_epoch", "val_main_metric_name", "val_main_metric", "test_main_metric_name", "test_main_metric", "test_auroc", "test_accuracy", "test_macro_f1", "test_micro_f1", "mean_fast_s_v", "mean_fast_s_t", "mean_fast_s_branch", "mean_fast_s_f", "mean_fast_fcf", "mean_fast_rfcf", "mean_exact_fcf", "mean_exact_rfcf", "trainable_params", "wall_clock_hms", "wall_clock_seconds", "peak_gpu_memory_mb", "exp_name", "result_path", "summary_path", "status",
    ]
    cols = [c for c in preferred_cols if c in all_df.columns] + [c for c in all_df.columns if c not in preferred_cols]
    all_df = all_df[cols]
    group_df = make_group_comparison(all_df, methods=methods, split=args.split)
    best_df = group_df.head(args.top_k).copy() if not group_df.empty else pd.DataFrame()
    out_all = result_dir / f"{args.output_prefix}_all_results.csv"
    out_group = result_dir / f"{args.output_prefix}_group_comparison.csv"
    out_best = result_dir / f"{args.output_prefix}_best_configs.csv"
    out_report = result_dir / f"{args.output_prefix}_selection_report.txt"
    all_df.to_csv(out_all, index=False)
    group_df.to_csv(out_group, index=False)
    best_df.to_csv(out_best, index=False)
    write_report(result_dir, all_df, group_df, best_df, out_report, args.split, methods, args.expected_groups)
    print(f"[summarize] Parsed summary files: {len(summary_files)}")
    print(f"[summarize] Wrote: {out_all}")
    print(f"[summarize] Wrote: {out_group}")
    print(f"[summarize] Wrote: {out_best}")
    print(f"[summarize] Wrote: {out_report}")
    if args.expected_groups:
        expected_runs = args.expected_groups * len(methods)
        if len(all_df) != expected_runs:
            print(f"[summarize][warning] Expected {expected_runs} runs but parsed {len(all_df)}.")
        if len(group_df) != args.expected_groups:
            print(f"[summarize][warning] Expected {args.expected_groups} grid groups but built {len(group_df)}. Check grid_id parsing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
