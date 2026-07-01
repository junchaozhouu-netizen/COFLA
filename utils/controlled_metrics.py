from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# Main controlled geometry protocol: FAST-FCF / FAST-RFCF are the default
# geometry evaluation metrics for all methods. Exact FCF fields are optional
# diagnostics and are only populated when --geometry_metric exact_fcf or both.
GEOMETRY_KEYS = [
    "loss",
    "fast_s_v",
    "fast_s_t",
    "fast_s_branch",
    "fast_s_f",
    "fast_fcf",
    "fast_rfcf",
    "exact_s_v",
    "exact_s_t",
    "exact_s_branch",
    "exact_s_f",
    "exact_fcf",
    "exact_rfcf",
]


def hateful_memes_metrics(labels: Sequence[int], probs: Sequence[float], threshold: float = 0.5) -> Dict[str, float]:
    labels_np = np.asarray(labels).astype(int)
    probs_np = np.asarray(probs).astype(float)
    preds_np = (probs_np >= threshold).astype(int)
    accuracy = float(accuracy_score(labels_np, preds_np)) if labels_np.size else float("nan")

    auroc = float("nan")
    try:
        if labels_np.size and np.unique(labels_np).size > 1:
            auroc = float(roc_auc_score(labels_np, probs_np))
    except Exception:
        auroc = float("nan")

    return {
        "accuracy": accuracy,
        "auroc": auroc,
    }


def mmimdb_metrics(labels, probs, threshold: float = 0.5) -> Dict[str, float]:
    labels_np = np.asarray(labels).astype(int)
    probs_np = np.asarray(probs).astype(float)
    preds_np = (probs_np >= threshold).astype(int)
    return {
        "macro_f1": float(f1_score(labels_np, preds_np, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels_np, preds_np, average="micro", zero_division=0)),
    }


def compute_controlled_metrics(dataset: str, labels, probs, threshold: float = 0.5) -> Dict[str, float]:
    dataset_key = str(dataset).strip().lower()
    if dataset_key == "hateful_memes":
        return hateful_memes_metrics(labels, probs, threshold=threshold)
    if dataset_key == "mmimdb":
        return mmimdb_metrics(labels, probs, threshold=threshold)
    raise ValueError(f"Unsupported controlled dataset: {dataset}")


def select_primary_metric(dataset: str, metrics: Dict[str, float]) -> float:
    dataset_key = str(dataset).strip().lower()
    if dataset_key == "hateful_memes":
        value = metrics.get("auroc")
        if value is None or math.isnan(value):
            value = metrics.get("accuracy", float("-inf"))
        return float(value)
    if dataset_key == "mmimdb":
        return float(metrics.get("macro_f1", metrics.get("micro_f1", float("-inf"))))
    raise ValueError(f"Unsupported controlled dataset: {dataset}")


def primary_metric_name(dataset: str) -> str:
    dataset_key = str(dataset).strip().lower()
    if dataset_key == "hateful_memes":
        return "AUROC"
    if dataset_key == "mmimdb":
        return "Macro-F1"
    raise ValueError(f"Unsupported controlled dataset: {dataset}")


def mean_dict(records: Iterable[Dict[str, float]], keys: Sequence[str]) -> Dict[str, float]:
    sums = {key: 0.0 for key in keys}
    counts = {key: 0 for key in keys}
    for item in records:
        for key in keys:
            value = item.get(key)
            if value is None:
                continue
            try:
                value_float = float(value)
            except (TypeError, ValueError):
                continue
            if math.isnan(value_float):
                continue
            sums[key] += value_float
            counts[key] += 1
    means: Dict[str, float] = {}
    for key in keys:
        means[key] = sums[key] / counts[key] if counts[key] > 0 else float("nan")
    return means
