from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def hateful_memes_metrics(labels: List[int], probs: List[float], threshold: float = 0.5) -> Dict[str, float]:
    labels_np = np.asarray(labels).astype(int)
    probs_np = np.asarray(probs).astype(float)
    preds_np = (probs_np >= threshold).astype(int)

    acc = float(accuracy_score(labels_np, preds_np)) if len(labels_np) else 0.0
    try:
        auroc = float(roc_auc_score(labels_np, probs_np)) if len(np.unique(labels_np)) > 1 else 0.0
    except Exception:
        auroc = 0.0
    return {"accuracy": acc, "auroc": auroc}


def accuracy_metrics(labels: List[int], preds: List[int]) -> Dict[str, float]:
    labels_np = np.asarray(labels).astype(int)
    preds_np = np.asarray(preds).astype(int)
    acc = float(accuracy_score(labels_np, preds_np)) if len(labels_np) else 0.0
    return {"accuracy": acc}


def mmimdb_metrics(labels, probs, threshold: float = 0.5) -> Dict[str, float]:
    labels_np = np.asarray(labels)
    probs_np = np.asarray(probs)
    preds_np = (probs_np >= threshold).astype(int)
    return {
        "macro_f1": float(f1_score(labels_np, preds_np, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels_np, preds_np, average="micro", zero_division=0)),
    }


def compute_metrics_by_dataset(dataset: str, labels, preds, scores, threshold: float = 0.5) -> Dict[str, float]:
    dataset = str(dataset).lower()
    if dataset == "hateful_memes":
        return hateful_memes_metrics(labels, scores)
    if dataset in {"nlvr2", "scienceqa"}:
        return accuracy_metrics(labels, preds)
    if dataset == "mmimdb":
        return mmimdb_metrics(labels, scores, threshold=threshold)
    raise ValueError(f"Unsupported dataset for metrics: {dataset}")


def merge_prefix(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}
