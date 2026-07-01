from __future__ import annotations

import math
import statistics
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch


def safe_float_tensor(x: Any) -> torch.Tensor:
    if torch.is_tensor(x):
        value = x.detach()
        if value.numel() != 1:
            raise ValueError(f"safe_float_tensor expects a scalar tensor, but got shape={tuple(value.shape)}")
        return value.float()
    return torch.tensor(float(x), dtype=torch.float32)


def finite_positive(x: Any) -> bool:
    value = float(safe_float_tensor(x).detach().cpu().item())
    return math.isfinite(value) and value > 0.0


def summarize_values(values: Iterable[float]) -> Dict[str, Any]:
    cleaned = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not cleaned:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "values": [],
        }
    cleaned.sort()
    return {
        "count": len(cleaned),
        "mean": float(statistics.fmean(cleaned)),
        "median": float(statistics.median(cleaned)),
        "min": float(cleaned[0]),
        "max": float(cleaned[-1]),
        "values": list(cleaned),
    }


def grad_norm(
    loss: torch.Tensor,
    params: Iterable[torch.nn.Parameter],
    retain_graph: bool = True,
    allow_unused: bool = True,
) -> torch.Tensor:
    params = list(params)
    device = loss.device
    if not params:
        return torch.tensor(0.0, device=device, dtype=torch.float32)
    grads = torch.autograd.grad(
        outputs=loss,
        inputs=params,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=allow_unused,
    )
    sq = 0.0
    for grad in grads:
        if grad is None:
            continue
        sq += grad.detach().float().pow(2).sum().item()
    return torch.tensor(math.sqrt(sq), device=device, dtype=torch.float32)


def compute_gradient_balanced_weight(
    *,
    task_grad_norm: Any,
    aux_grad_norm: Any,
    coefficient: float,
    min_value: float,
    max_value: float,
    eps: float,
    previous_ema: Optional[float],
    fallback_value: float,
) -> Dict[str, Any]:
    task_value = float(safe_float_tensor(task_grad_norm).detach().cpu().item())
    aux_value = float(safe_float_tensor(aux_grad_norm).detach().cpu().item())
    raw_value: Optional[float] = None
    if math.isfinite(task_value) and task_value >= 0.0 and math.isfinite(aux_value) and aux_value > 0.0:
        candidate = float(coefficient) * task_value / (aux_value + float(eps))
        if math.isfinite(candidate):
            raw_value = float(max(min_value, min(max_value, candidate)))

    fallback = previous_ema if previous_ema is not None and math.isfinite(float(previous_ema)) else float(fallback_value)
    used_fallback = raw_value is None
    return {
        "raw": fallback if raw_value is None else raw_value,
        "valid": raw_value is not None,
        "used_fallback": used_fallback,
        "fallback": fallback,
    }


def update_ema(previous: Optional[float], new_value: float, tau: float) -> float:
    new_value = float(new_value)
    if previous is None or not math.isfinite(float(previous)):
        return new_value
    return float(tau) * float(previous) + (1.0 - float(tau)) * new_value


def calibrate_branch_matched_rho_f(
    *,
    dataloader,
    prepare_batch_fn: Callable[[Any], Any],
    compute_stats_fn: Callable[[Any], Dict[str, Any]],
    rho_f_before: float,
    num_batches: int,
    stat: str = "median",
    logger=None,
) -> Dict[str, Any]:
    if stat not in {"median", "mean"}:
        raise ValueError(f"Unsupported rho_f calibration stat: {stat}")

    candidates: List[float] = []
    requested_batches = max(0, int(num_batches))
    for batch_index, raw_batch in enumerate(dataloader):
        if batch_index >= requested_batches:
            break
        batch = prepare_batch_fn(raw_batch)
        try:
            stats = compute_stats_fn(batch)
        except Exception as exc:
            if logger is not None:
                logger.warning(
                    "[AdaptiveCalibration] Skipping calibration batch %s/%s due to: %s",
                    batch_index + 1,
                    requested_batches,
                    exc,
                )
            continue

        branch_linear_proxy = safe_float_tensor(stats["branch_linear_proxy"])
        gf_norm = safe_float_tensor(stats["gf_norm"])
        if not finite_positive(gf_norm):
            continue

        candidate = float((branch_linear_proxy / (gf_norm + 1e-12)).detach().cpu().item())
        if not math.isfinite(candidate) or candidate <= 0.0:
            continue
        candidates.append(candidate)

    summary = summarize_values(candidates)
    rho_f_after = float(rho_f_before)
    if summary["count"] > 0:
        # Projection-COFLA uses the median/mean calibration value directly.
        stat_value = summary[stat]
        rho_f_after = float(stat_value)
    elif logger is not None:
        logger.warning(
            "[AdaptiveCalibration] No valid rho_f calibration batches were found. Falling back to rho_f=%s",
            rho_f_before,
        )

    return {
        "rho_f_before_calibration": float(rho_f_before),
        "rho_f_after_calibration": float(rho_f_after),
        "rho_f_calib_requested_batches": requested_batches,
        "rho_f_calib_count": int(summary["count"]),
        "rho_f_calib_mean": summary["mean"],
        "rho_f_calib_median": summary["median"],
        "rho_f_calib_min": summary["min"],
        "rho_f_calib_max": summary["max"],
        "rho_f_calib_stat": stat,
        "rho_f_calib_values": summary["values"],
    }
