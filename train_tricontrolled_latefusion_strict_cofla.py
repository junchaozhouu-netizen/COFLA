#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Independent tri-modal controlled late-fusion training code for CMU-MOSI / CMU-MOSEI
processed aligned_50.pkl features.

STRICT COFLA VERSION: paper-aligned COFLA-E/F implementation.
No manually tuned cofla_lambda is used. The robust objective uses
L_rob = L0 + fusegate * (L_pert - L0), where fusegate is computed
adaptively as stopgrad(J_BF / (1 + J_BF)).

Input modalities are pre-extracted features:
  text   : (N, 50, D_text)
  audio  : (N, 50, D_audio)
  vision : (N, 50, D_vision)

Architecture:
  modality feature -> MLP adapter -> 1-layer BiGRU/Transformer -> attention pooling
  three modality representations -> concat_mlp or gated_mlp fusion -> sentiment score

Supported methods:
  vanilla, sam, esam, msam, masam, dgl, cofla_e, cofla_f

Metrics:
  Acc-2(non-zero), Macro-F1(non-zero), MAE, Corr, fast-FCF, fast-RFCF(log ratio)

This script is self-contained and does not modify or import your existing VLM code.
Place this file at the repository root when running tri-modal experiments.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from torch.func import functional_call
except Exception:  # PyTorch < 2.0 fallback
    from torch.nn.utils.stateless import functional_call

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **kwargs: x


# -------------------------
# Utilities
# -------------------------


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def tensor_global_norm(items: Iterable[Optional[torch.Tensor]], normalize_numel: bool = False) -> torch.Tensor:
    sq = None
    n = 0
    device = None
    for t in items:
        if t is None:
            continue
        device = t.device
        val = torch.sum(t.detach().float() ** 2)
        sq = val if sq is None else sq + val
        n += t.numel()
    if sq is None:
        if device is None:
            return torch.tensor(0.0)
        return torch.tensor(0.0, device=device)
    norm = torch.sqrt(sq + 1e-12)
    if normalize_numel and n > 0:
        norm = norm / math.sqrt(float(n))
    return norm


def np_nan_to_num(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


# -------------------------
# Dataset
# -------------------------


class CMUAlignedFeatureDataset(Dataset):
    def __init__(self, root: str | Path, split: str):
        self.root = Path(root)
        self.split = split
        pkl_path = self.root / "aligned_50.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"Cannot find {pkl_path}")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        if split not in data:
            raise KeyError(f"Split {split!r} not found in {pkl_path}; available={list(data.keys())}")
        block = data[split]
        for key in ["text", "audio", "vision", "regression_labels"]:
            if key not in block:
                raise KeyError(f"Required key {key!r} not found in split {split!r}; keys={list(block.keys())}")
        self.text = np_nan_to_num(np.asarray(block["text"], dtype=np.float32))
        self.audio = np_nan_to_num(np.asarray(block["audio"], dtype=np.float32))
        self.vision = np_nan_to_num(np.asarray(block["vision"], dtype=np.float32))
        self.labels = np_nan_to_num(np.asarray(block["regression_labels"], dtype=np.float32).reshape(-1))
        self.ids = block.get("id", [str(i) for i in range(len(self.labels))])
        self.raw_text = block.get("raw_text", None)
        n = len(self.labels)
        assert self.text.shape[0] == n and self.audio.shape[0] == n and self.vision.shape[0] == n

    @property
    def dims(self) -> Dict[str, int]:
        return {
            "text_dim": int(self.text.shape[-1]),
            "audio_dim": int(self.audio.shape[-1]),
            "vision_dim": int(self.vision.shape[-1]),
            "seq_len": int(self.text.shape[1]),
        }

    def summary(self) -> Dict[str, object]:
        labels = self.labels
        return {
            "root": str(self.root),
            "split": self.split,
            "num_samples": int(len(self)),
            "text_shape": list(self.text.shape),
            "audio_shape": list(self.audio.shape),
            "vision_shape": list(self.vision.shape),
            "label_min": float(np.min(labels)),
            "label_max": float(np.max(labels)),
            "label_mean": float(np.mean(labels)),
            "num_label_neg": int(np.sum(labels < 0)),
            "num_label_zero": int(np.sum(labels == 0)),
            "num_label_pos": int(np.sum(labels > 0)),
        }

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return {
            "text": self.text[idx],
            "audio": self.audio[idx],
            "vision": self.vision[idx],
            "label": self.labels[idx],
            "id": self.ids[idx],
        }


def collate_cmu(batch: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "text": torch.tensor(np.stack([b["text"] for b in batch]), dtype=torch.float32),
        "audio": torch.tensor(np.stack([b["audio"] for b in batch]), dtype=torch.float32),
        "vision": torch.tensor(np.stack([b["vision"] for b in batch]), dtype=torch.float32),
        "labels": torch.tensor(np.asarray([b["label"] for b in batch]), dtype=torch.float32),
        "ids": [b["id"] for b in batch],
    }


# -------------------------
# Model
# -------------------------


class AttentionPooling(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,T,D
        a = self.score(x).squeeze(-1)
        w = torch.softmax(a, dim=-1)
        return torch.sum(x * w.unsqueeze(-1), dim=1)


class ModalityBranch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, temporal_encoder: str = "bigru"):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        temporal_encoder = temporal_encoder.lower()
        self.temporal_encoder_name = temporal_encoder
        if temporal_encoder == "bigru":
            if hidden_dim % 2 != 0:
                raise ValueError("hidden_dim must be even for BiGRU")
            self.temporal = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=0.0,
            )
        elif temporal_encoder in {"transformer", "transformer_lite", "transformer-lite"}:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=1)
        else:
            raise ValueError(f"Unsupported temporal_encoder={temporal_encoder}")
        self.pool = AttentionPooling(hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.adapter(x)
        if self.temporal_encoder_name == "bigru":
            z, _ = self.temporal(z)
        else:
            z = self.temporal(z)
        h = self.pool(z)
        return h


class ConcatMLPFusion(nn.Module):
    def __init__(self, hidden_dim: int, fusion_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 3, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1),
        )

    def forward(self, h_text: torch.Tensor, h_audio: torch.Tensor, h_vision: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h_text, h_audio, h_vision], dim=-1)).squeeze(-1)


class GatedMLPFusion(nn.Module):
    def __init__(self, hidden_dim: int, fusion_dim: int, dropout: float):
        super().__init__()
        self.proj_text = nn.Linear(hidden_dim, fusion_dim)
        self.proj_audio = nn.Linear(hidden_dim, fusion_dim)
        self.proj_vision = nn.Linear(hidden_dim, fusion_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 3),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1),
        )

    def forward(self, h_text: torch.Tensor, h_audio: torch.Tensor, h_vision: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([h_text, h_audio, h_vision], dim=-1)
        g = torch.softmax(self.gate(cat), dim=-1)  # B,3
        ht = self.proj_text(h_text)
        ha = self.proj_audio(h_audio)
        hv = self.proj_vision(h_vision)
        fused = g[:, 0:1] * ht + g[:, 1:2] * ha + g[:, 2:3] * hv
        return self.out(fused).squeeze(-1)


class TriModalLateFusionModel(nn.Module):
    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        vision_dim: int,
        branch_hidden_dim: int,
        fusion_dim: int,
        dropout: float,
        fusion_type: str,
        temporal_encoder: str,
    ):
        super().__init__()
        self.text_branch = ModalityBranch(text_dim, branch_hidden_dim, dropout, temporal_encoder)
        self.audio_branch = ModalityBranch(audio_dim, branch_hidden_dim, dropout, temporal_encoder)
        self.vision_branch = ModalityBranch(vision_dim, branch_hidden_dim, dropout, temporal_encoder)
        fusion_type = fusion_type.lower()
        self.fusion_type = fusion_type
        if fusion_type == "concat_mlp":
            self.fusion = ConcatMLPFusion(branch_hidden_dim, fusion_dim, dropout)
        elif fusion_type == "gated_mlp":
            self.fusion = GatedMLPFusion(branch_hidden_dim, fusion_dim, dropout)
        else:
            raise ValueError(f"Unsupported fusion_type={fusion_type}")

    def encode_branches(self, text: torch.Tensor, audio: torch.Tensor, vision: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.text_branch(text), self.audio_branch(audio), self.vision_branch(vision)

    def fuse_from_reps(self, reps: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.fusion(*reps)

    def forward(self, text: torch.Tensor, audio: torch.Tensor, vision: torch.Tensor, return_reps: bool = False):
        reps = self.encode_branches(text, audio, vision)
        pred = self.fuse_from_reps(reps)
        if return_reps:
            return pred, reps
        return pred

    def branch_parameters(self):
        for module in [self.text_branch, self.audio_branch, self.vision_branch]:
            yield from module.parameters()

    def fusion_parameters(self):
        yield from self.fusion.parameters()


# -------------------------
# Metrics and geometry
# -------------------------


def regression_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    preds = preds.reshape(-1).astype(np.float64)
    labels = labels.reshape(-1).astype(np.float64)
    mae = float(np.mean(np.abs(preds - labels)))
    if np.std(preds) < 1e-12 or np.std(labels) < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(preds, labels)[0, 1])
    mask = labels != 0
    if np.sum(mask) == 0:
        return {"mae": mae, "corr": corr, "acc2": 0.0, "macro_f1": 0.0, "num_acc2": 0}
    yt = labels[mask] > 0
    yp = preds[mask] > 0
    acc = float(np.mean(yt == yp) * 100.0)

    f1s = []
    for cls in [False, True]:
        tp = np.sum((yp == cls) & (yt == cls))
        fp = np.sum((yp == cls) & (yt != cls))
        fn = np.sum((yp != cls) & (yt == cls))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s) * 100.0)
    return {"mae": mae, "corr": corr, "acc2": acc, "macro_f1": macro_f1, "num_acc2": int(np.sum(mask))}


def compute_fast_rfcf_batch(
    model: TriModalLateFusionModel,
    batch: Dict[str, torch.Tensor],
    loss_fn: nn.Module,
    device: torch.device,
    rho_text: float,
    rho_audio: float,
    rho_vision: float,
    rho_f: float,
    eps: float = 1e-8,
) -> Dict[str, float]:
    model.zero_grad(set_to_none=True)
    text = batch["text"].to(device)
    audio = batch["audio"].to(device)
    vision = batch["vision"].to(device)
    labels = batch["labels"].to(device)
    pred, reps = model(text, audio, vision, return_reps=True)
    loss = loss_fn(pred, labels)
    rep_grads = torch.autograd.grad(loss, reps, retain_graph=True, allow_unused=True)
    branch_vals = []
    for g, rho in zip(rep_grads, [rho_text, rho_audio, rho_vision]):
        branch_vals.append(float((rho * tensor_global_norm([g], normalize_numel=False)).detach().cpu()))
    branch_mean = float(np.mean(branch_vals))
    fusion_params = [p for p in model.fusion_parameters() if p.requires_grad]
    fusion_grads = torch.autograd.grad(loss, fusion_params, retain_graph=False, allow_unused=True)
    fusion_val = float((rho_f * tensor_global_norm(fusion_grads, normalize_numel=False)).detach().cpu())
    fast_fcf = float((fusion_val + eps) / (branch_mean + eps))
    fast_rfcf = float(math.log(max(fast_fcf, eps)))
    return {
        "branch_fast_sharpness": branch_mean,
        "fusion_fast_sharpness": fusion_val,
        "fast_fcf": fast_fcf,
        "fast_rfcf": fast_rfcf,
        "positive_fast_rfcf": max(0.0, fast_rfcf),
    }


# -------------------------
# Optimizer perturbation helpers
# -------------------------


@dataclass
class PerturbBackup:
    params: List[nn.Parameter]
    values: List[torch.Tensor]


def get_trainable_params(model: nn.Module) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def perturb_parameters(
    params: List[nn.Parameter],
    rho: float,
    mode: str = "full",
    keep_ratio: float = 1.0,
) -> PerturbBackup:
    grads = [p.grad for p in params]
    norm = tensor_global_norm(grads).to(params[0].device if params else "cpu")
    scale = rho / (norm + 1e-12)
    backup = PerturbBackup(params=params, values=[])
    for p in params:
        backup.values.append(p.data.detach().clone())
        if p.grad is None:
            continue
        e = p.grad.detach() * scale
        if mode == "topk" and keep_ratio < 1.0:
            flat = e.abs().flatten()
            if flat.numel() > 0:
                k = max(1, int(math.ceil(flat.numel() * keep_ratio)))
                if k < flat.numel():
                    thresh = torch.topk(flat, k, largest=True).values[-1]
                    e = e * (e.abs() >= thresh).to(e.dtype)
        p.data.add_(e)
    return backup


def restore_parameters(backup: PerturbBackup) -> None:
    for p, v in zip(backup.params, backup.values):
        p.data.copy_(v)


def split_params(model: TriModalLateFusionModel) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    branch = [p for p in model.branch_parameters() if p.requires_grad]
    fusion = [p for p in model.fusion_parameters() if p.requires_grad]
    return branch, fusion


def branch_param_groups(model: TriModalLateFusionModel) -> Dict[str, List[nn.Parameter]]:
    return {
        "text": [p for p in model.text_branch.parameters() if p.requires_grad],
        "audio": [p for p in model.audio_branch.parameters() if p.requires_grad],
        "vision": [p for p in model.vision_branch.parameters() if p.requires_grad],
    }


def safe_grad_list(grads: Iterable[Optional[torch.Tensor]], params: List[nn.Parameter]) -> List[torch.Tensor]:
    out = []
    for g, p in zip(grads, params):
        if g is None:
            out.append(torch.zeros_like(p))
        else:
            out.append(g)
    return out


def grad_dot(a: List[torch.Tensor], b: List[torch.Tensor]) -> torch.Tensor:
    if not a:
        return torch.tensor(0.0)
    total = None
    for x, y in zip(a, b):
        v = torch.sum(x.float() * y.float())
        total = v if total is None else total + v
    if total is None:
        return torch.tensor(0.0, device=a[0].device)
    return total


def grad_norm_sq(a: List[torch.Tensor]) -> torch.Tensor:
    if not a:
        return torch.tensor(0.0)
    total = None
    for x in a:
        v = torch.sum(x.float() * x.float())
        total = v if total is None else total + v
    if total is None:
        return torch.tensor(0.0, device=a[0].device)
    return total


def project_gradient(
    g_rob: List[torch.Tensor],
    g_stab: List[torch.Tensor],
    eps: float,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Apply g <- g_rob + stopgrad(beta*) g_stab.

    beta* = max(0, - <g_rob, g_stab> / (||g_stab||^2 + eps)).
    """
    if not g_rob:
        return g_rob, torch.tensor(0.0)
    denom = grad_norm_sq(g_stab) + eps
    numer = grad_dot(g_rob, g_stab)
    beta = torch.clamp(-numer / denom, min=0.0).detach()
    return [gr + beta.to(gr.device) * gs for gr, gs in zip(g_rob, g_stab)], beta


def set_gradients(params: List[nn.Parameter], grads: List[torch.Tensor]) -> None:
    for p, g in zip(params, grads):
        p.grad = g.detach().clone()


def rep_delta(rep: torch.Tensor, grad: Optional[torch.Tensor], rho: float, eps: float) -> torch.Tensor:
    if grad is None:
        return torch.zeros_like(rep)
    # Frobenius norm over the whole minibatch representation block.
    denom = torch.norm(grad.detach().float()) + eps
    return (rho * grad.detach() / denom).to(rep.dtype)


def fusion_loss_with_finite_step_probe(
    model: TriModalLateFusionModel,
    reps: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    labels: torch.Tensor,
    loss_fn: nn.Module,
    fusion_grads: List[torch.Tensor],
    rho_f: float,
    eps: float,
) -> torch.Tensor:
    """Evaluate L_f^+ using theta_f + delta_f without mutating model parameters."""
    named_params = [(n, p) for n, p in model.fusion.named_parameters() if p.requires_grad]
    params = [p for _, p in named_params]
    norm = torch.sqrt(sum(torch.sum(g.detach().float() ** 2) for g in fusion_grads) + 1e-12)
    override = {}
    for (name, p), g in zip(named_params, fusion_grads):
        override[name] = p + (rho_f * g.detach() / (norm + eps)).to(p.dtype)
    # Include buffers for compatibility with both torch.func.functional_call and stateless.functional_call.
    for name, b in model.fusion.named_buffers():
        override[name] = b
    pred_plus = functional_call(model.fusion, override, reps)
    return loss_fn(pred_plus, labels)


def strict_cofla_backward(
    model: TriModalLateFusionModel,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    loss_fn: nn.Module,
    args,
    variant: str,
) -> Tuple[float, Dict[str, float]]:
    """Paper-aligned COFLA-E/F backward pass for tri-modal late fusion.

    COFLA-E:
      - finite-step probes for text/audio/vision branch reps and fusion params;
      - L_pert^E = mean(L_text^+, L_audio^+, L_vision^+, L_f^+);
      - R_BF^E = log((S_f + eps) / (S_branch + eps));
      - branch and fusion projection safeguards.

    COFLA-F:
      - first-order detached branch reference;
      - finite-step fusion probe L_f^+;
      - L_pert^F = L_f^+;
      - no branch-side projection; fusion projection is retained.

    No manual auxiliary loss weight is used.
    """
    variant = variant.lower()
    text = batch["text"].to(device)
    audio = batch["audio"].to(device)
    vision = batch["vision"].to(device)
    labels = batch["labels"].to(device)

    pred, reps = model(text, audio, vision, return_reps=True)
    l0 = loss_fn(pred, labels)

    # Gradients used only to form detached perturbation directions/references.
    rep_grads_raw = torch.autograd.grad(l0, reps, retain_graph=True, allow_unused=True)
    deltas = [
        rep_delta(rep, grad, rho, args.fcf_eps)
        for rep, grad, rho in zip(reps, rep_grads_raw, [args.rho_text, args.rho_audio, args.rho_vision])
    ]

    fusion_params = [p for p in model.fusion_parameters() if p.requires_grad]
    fusion_grads_raw = torch.autograd.grad(l0, fusion_params, retain_graph=True, allow_unused=True)
    fusion_grads = safe_grad_list(fusion_grads_raw, fusion_params)
    lf_plus = fusion_loss_with_finite_step_probe(
        model, reps, labels, loss_fn, fusion_grads, args.rho_f, args.fcf_eps
    )
    s_f = F.relu(lf_plus - l0)

    branch_losses: List[torch.Tensor] = []
    branch_sharps: List[torch.Tensor] = []
    if variant == "cofla_e":
        for i in range(3):
            perturbed = list(reps)
            perturbed[i] = perturbed[i] + deltas[i]
            lm_plus = loss_fn(model.fuse_from_reps(tuple(perturbed)), labels)
            branch_losses.append(lm_plus)
            branch_sharps.append(F.relu(lm_plus - l0))
        s_branch = torch.stack(branch_sharps).mean()
        r_bf = torch.log((s_f + args.fcf_eps) / (s_branch + args.fcf_eps))
        l_pert = torch.stack(branch_losses + [lf_plus]).mean()
    elif variant == "cofla_f":
        branch_fast_vals = []
        for grad, rho in zip(rep_grads_raw, [args.rho_text, args.rho_audio, args.rho_vision]):
            branch_fast_vals.append(rho * tensor_global_norm([grad], normalize_numel=False).to(l0.device))
        s_branch_fast = torch.stack(branch_fast_vals).mean().detach()
        r_bf = torch.log((s_f + args.fcf_eps) / (s_branch_fast + args.fcf_eps))
        l_pert = lf_plus
    else:
        raise ValueError(f"Unknown COFLA variant: {variant}")

    j_bf = F.relu(r_bf)
    fusegate = (j_bf / (1.0 + j_bf)).detach()
    l_rob = l0 + fusegate * (l_pert - l0)

    groups = branch_param_groups(model)
    text_params = groups["text"]
    audio_params = groups["audio"]
    vision_params = groups["vision"]
    branch_groups = [text_params, audio_params, vision_params]
    branch_params = text_params + audio_params + vision_params

    all_rob_params = branch_params + fusion_params
    g_rob_raw = torch.autograd.grad(l_rob, all_rob_params, retain_graph=True, allow_unused=True)
    g_rob_all = safe_grad_list(g_rob_raw, all_rob_params)
    g_rob_branch = g_rob_all[: len(branch_params)]
    g_rob_fusion = g_rob_all[len(branch_params) :]

    final_branch_grads: List[torch.Tensor] = []
    offset = 0
    beta_values = []
    if variant == "cofla_e":
        for i, params_i in enumerate(branch_groups):
            n_i = len(params_i)
            robust_i = g_rob_branch[offset : offset + n_i]
            g_stab_raw = torch.autograd.grad(branch_sharps[i], params_i, retain_graph=True, allow_unused=True)
            g_stab_i = safe_grad_list(g_stab_raw, params_i)
            projected_i, beta_i = project_gradient(robust_i, g_stab_i, args.projection_eps)
            final_branch_grads.extend(projected_i)
            beta_values.append(float(beta_i.detach().cpu()))
            offset += n_i
    else:
        final_branch_grads = g_rob_branch
        beta_values = [0.0, 0.0, 0.0]

    g_bf_raw = torch.autograd.grad(j_bf, fusion_params, retain_graph=False, allow_unused=True)
    g_bf = safe_grad_list(g_bf_raw, fusion_params)
    final_fusion_grads, alpha = project_gradient(g_rob_fusion, g_bf, args.projection_eps)

    set_gradients(branch_params, final_branch_grads)
    set_gradients(fusion_params, final_fusion_grads)

    stats = {
        "l0": float(l0.detach().cpu()),
        "l_pert": float(l_pert.detach().cpu()),
        "l_rob": float(l_rob.detach().cpu()),
        "s_f": float(s_f.detach().cpu()),
        "r_bf": float(r_bf.detach().cpu()),
        "j_bf": float(j_bf.detach().cpu()),
        "fusegate": float(fusegate.detach().cpu()),
        "alpha": float(alpha.detach().cpu()),
        "beta_text": beta_values[0],
        "beta_audio": beta_values[1],
        "beta_vision": beta_values[2],
    }
    return float(l_rob.detach().cpu()), stats


# -------------------------
# Training losses/steps
# -------------------------


def compute_base_loss(model, batch, device, loss_fn, dgl_strength: float = 0.0):
    text = batch["text"].to(device)
    audio = batch["audio"].to(device)
    vision = batch["vision"].to(device)
    labels = batch["labels"].to(device)
    pred, reps = model(text, audio, vision, return_reps=True)
    loss = loss_fn(pred, labels)
    if dgl_strength > 0:
        # A small representation-level regularizer used for the DGL baseline.
        # It encourages modality representations to avoid extreme conflict without dominating task loss.
        h1, h2, h3 = [F.normalize(h, dim=-1) for h in reps]
        conflict = (1 - (h1 * h2).sum(-1)).mean() + (1 - (h1 * h3).sum(-1)).mean() + (1 - (h2 * h3).sum(-1)).mean()
        loss = loss + dgl_strength * 0.01 * conflict / 3.0
    return loss


def train_step(model, batch, optimizer, device, loss_fn, args, state: Dict[str, float]):
    method = args.method.lower()
    optimizer.zero_grad(set_to_none=True)

    if method == "vanilla":
        loss = compute_base_loss(model, batch, device, loss_fn)
        loss.backward()
    elif method == "dgl":
        loss = compute_base_loss(model, batch, device, loss_fn, dgl_strength=args.dgl_correction_strength)
        loss.backward()
    elif method in {"cofla_e", "cofla_f"}:
        loss_value, cofla_stats = strict_cofla_backward(model, batch, device, loss_fn, args, method)
        state["last_cofla_stats"] = cofla_stats
        loss = torch.tensor(loss_value)
    elif method in {"sam", "esam", "msam", "masam"}:
        # First gradient pass.
        base_loss = compute_base_loss(model, batch, device, loss_fn)
        base_loss.backward()
        loss = base_loss.detach()
        if method == "esam":
            params = get_trainable_params(model)
            backup = perturb_parameters(params, rho=args.sam_rho, mode="topk", keep_ratio=args.esam_keep_ratio)
        elif method == "msam":
            branch_params, fusion_params = split_params(model)
            backup_b = perturb_parameters(branch_params, rho=args.sam_rho)
            backup_f = perturb_parameters(fusion_params, rho=args.sam_rho * 0.5)
            backup = (backup_b, backup_f)
        elif method == "masam":
            params = get_trainable_params(model)
            grad_norm = float(tensor_global_norm([p.grad for p in params]).detach().cpu())
            prev = state.get("masam_ema_grad_norm", grad_norm)
            ema = args.masam_ma_beta * prev + (1.0 - args.masam_ma_beta) * grad_norm
            state["masam_ema_grad_norm"] = ema
            scale = max(args.masam_rho_min_scale, min(args.masam_rho_max_scale, grad_norm / (ema + 1e-12)))
            backup = perturb_parameters(params, rho=args.sam_rho * scale)
        else:  # sam
            params = get_trainable_params(model)
            backup = perturb_parameters(params, rho=args.sam_rho)

        optimizer.zero_grad(set_to_none=True)
        pert_loss = compute_base_loss(model, batch, device, loss_fn)
        pert_loss.backward()
        if method == "msam":
            restore_parameters(backup[1])
            restore_parameters(backup[0])
        else:
            restore_parameters(backup)
        loss = pert_loss.detach()
    else:
        raise ValueError(f"Unsupported method={method}")

    if args.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    return float(loss.detach().cpu())


# -------------------------
# Evaluation and calibration
# -------------------------


@torch.no_grad()
def predict_epoch(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, labels = [], []
    for batch in loader:
        pred = model(batch["text"].to(device), batch["audio"].to(device), batch["vision"].to(device))
        preds.append(pred.detach().cpu().numpy())
        labels.append(batch["labels"].numpy())
    return np.concatenate(preds), np.concatenate(labels)


def evaluate(model, loader, device, loss_fn, args, compute_geometry: bool) -> Dict[str, float]:
    preds, labels = predict_epoch(model, loader, device)
    metrics = regression_metrics(preds, labels)
    if compute_geometry:
        model.eval()
        vals = []
        for batch in loader:
            vals.append(
                compute_fast_rfcf_batch(
                    model, batch, loss_fn, device,
                    args.eval_rho_text, args.eval_rho_audio, args.eval_rho_vision, args.eval_rho_f,
                    eps=args.fcf_eps,
                )
            )
        for key in ["branch_fast_sharpness", "fusion_fast_sharpness", "fast_fcf", "fast_rfcf", "positive_fast_rfcf"]:
            metrics[key] = float(np.mean([v[key] for v in vals])) if vals else 0.0
    else:
        metrics["branch_fast_sharpness"] = 0.0
        metrics["fusion_fast_sharpness"] = 0.0
        metrics["fast_fcf"] = 0.0
        metrics["fast_rfcf"] = 0.0
        metrics["positive_fast_rfcf"] = 0.0
    return metrics


def calibrate_rho_f(model, loader, device, loss_fn, args) -> Optional[float]:
    if not args.auto_rho_f:
        return None
    model.eval()
    ratios = []
    num_batches = 0
    for batch in loader:
        num_batches += 1
        if num_batches > args.rho_f_calib_batches:
            break
        text = batch["text"].to(device)
        audio = batch["audio"].to(device)
        vision = batch["vision"].to(device)
        labels = batch["labels"].to(device)
        model.zero_grad(set_to_none=True)
        pred, reps = model(text, audio, vision, return_reps=True)
        loss = loss_fn(pred, labels)
        rep_grads = torch.autograd.grad(loss, reps, retain_graph=True, allow_unused=True)
        branch_vals = []
        for grad, rho in zip(rep_grads, [args.rho_text, args.rho_audio, args.rho_vision]):
            branch_vals.append(float((rho * tensor_global_norm([grad], normalize_numel=False)).detach().cpu()))
        branch_mean = float(np.mean(branch_vals))
        fusion_params = [p for p in model.fusion_parameters() if p.requires_grad]
        fusion_grads = torch.autograd.grad(loss, fusion_params, retain_graph=False, allow_unused=True)
        raw_fusion = float(tensor_global_norm(fusion_grads, normalize_numel=False).detach().cpu())
        if raw_fusion > 1e-12 and branch_mean > 0:
            ratios.append(branch_mean / raw_fusion)
    if not ratios:
        return None
    # Paper-aligned branch-matched calibration: median/mean branch reference divided by raw fusion-gradient norm.
    # No manually chosen clipping bound is applied here.
    new_rho = float(np.median(ratios) if args.rho_f_calib_stat == "median" else np.mean(ratios))
    args.rho_f = new_rho
    args.eval_rho_f = new_rho
    return new_rho


# -------------------------
# Main
# -------------------------


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", type=str2bool, default=False)
    ap.add_argument("--dataset", choices=["mosi", "mosei"], required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--result_root", required=True)
    ap.add_argument("--exp_name", required=True)
    ap.add_argument("--method", choices=["vanilla", "sam", "esam", "msam", "masam", "dgl", "cofla_e", "cofla_f"], required=True)
    ap.add_argument("--fusion_type", choices=["concat_mlp", "gated_mlp"], required=True)
    ap.add_argument("--branch_hidden_dim", type=int, default=256)
    ap.add_argument("--fusion_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--temporal_encoder", choices=["bigru", "transformer_lite", "transformer"], default="bigru")
    ap.add_argument("--num_train_epochs", type=int, default=30)
    ap.add_argument("--per_device_train_batch_size", type=int, default=32)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--loss", choices=["smoothl1", "mse"], default="smoothl1")
    ap.add_argument("--sam_rho", type=float, default=0.02)
    ap.add_argument("--esam_keep_ratio", type=float, default=0.75)
    ap.add_argument("--masam_ma_beta", type=float, default=0.9)
    ap.add_argument("--masam_rho_min_scale", type=float, default=0.5)
    ap.add_argument("--masam_rho_max_scale", type=float, default=2.0)
    ap.add_argument("--dgl_correction_strength", type=float, default=0.5)
    ap.add_argument("--rho_text", type=float, default=0.001)
    ap.add_argument("--rho_audio", type=float, default=0.001)
    ap.add_argument("--rho_vision", type=float, default=0.001)
    ap.add_argument("--rho_f", type=float, default=0.005)
    ap.add_argument("--eval_rho_text", type=float, default=0.001)
    ap.add_argument("--eval_rho_audio", type=float, default=0.001)
    ap.add_argument("--eval_rho_vision", type=float, default=0.001)
    ap.add_argument("--eval_rho_f", type=float, default=0.005)
    ap.add_argument("--fcf_eps", type=float, default=1e-8)
    ap.add_argument("--projection_eps", type=float, default=1e-8)
    ap.add_argument("--auto_rho_f", type=str2bool, default=False)
    ap.add_argument("--rho_f_calib_batches", type=int, default=20)
    ap.add_argument("--rho_f_calib_stat", choices=["median", "mean"], default="median")
    ap.add_argument("--val_compute_geometry", type=str2bool, default=True)
    ap.add_argument("--test_compute_geometry", type=str2bool, default=True)
    ap.add_argument("--log_every_n_steps", type=int, default=50)
    ap.add_argument("--save_best_metric", choices=["val_acc2", "val_macro_f1", "val_mae"], default="val_acc2")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed, args.deterministic)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    exp_dir = ensure_dir(Path(args.result_root) / args.exp_name)
    log_path = exp_dir / "train_log.jsonl"
    with open(exp_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    train_ds = CMUAlignedFeatureDataset(args.data_root, "train")
    val_ds = CMUAlignedFeatureDataset(args.data_root, "valid")
    test_ds = CMUAlignedFeatureDataset(args.data_root, "test")
    dims = train_ds.dims

    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_cmu,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_cmu,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_cmu,
        drop_last=False,
    )

    model = TriModalLateFusionModel(
        text_dim=dims["text_dim"],
        audio_dim=dims["audio_dim"],
        vision_dim=dims["vision_dim"],
        branch_hidden_dim=args.branch_hidden_dim,
        fusion_dim=args.fusion_dim,
        dropout=args.dropout,
        fusion_type=args.fusion_type,
        temporal_encoder=args.temporal_encoder,
    ).to(device)

    loss_fn = nn.SmoothL1Loss() if args.loss == "smoothl1" else nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run_info = {
        "dataset": args.dataset,
        "train_summary": train_ds.summary(),
        "val_summary": val_ds.summary(),
        "test_summary": test_ds.summary(),
        "dims": dims,
        "param_count": int(param_count),
        "trainable_count": int(trainable_count),
        "device": str(device),
    }
    print(json.dumps(run_info, indent=2, ensure_ascii=False))
    with open(exp_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    before_rho_f = args.rho_f
    calibrated = calibrate_rho_f(model, train_loader, device, loss_fn, args)
    print(f"[Calibration] rho_f before={before_rho_f}, after={args.rho_f}, calibrated={calibrated}")
    with open(exp_dir / "args_after_calibration.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    best_score = -1e18
    best_epoch = -1
    best_val = None
    last_val = None
    state: Dict[str, float] = {}
    global_step = 0
    t0 = time.time()

    for epoch in range(args.num_train_epochs):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}")
        for batch in pbar:
            global_step += 1
            loss_value = train_step(model, batch, optimizer, device, loss_fn, args, state)
            losses.append(loss_value)
            if global_step % args.log_every_n_steps == 0:
                pbar.set_postfix(loss=f"{np.mean(losses[-args.log_every_n_steps:]):.4f}")

        val_metrics = evaluate(model, val_loader, device, loss_fn, args, compute_geometry=args.val_compute_geometry)
        last_val = val_metrics
        if args.save_best_metric == "val_mae":
            score = -val_metrics["mae"]
        elif args.save_best_metric == "val_macro_f1":
            score = val_metrics["macro_f1"]
        else:
            score = val_metrics["acc2"]
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "elapsed_sec": time.time() - t0,
        }
        print(json.dumps(row, ensure_ascii=False))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_val = val_metrics
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "val_metrics": val_metrics}, exp_dir / "best.pt")

    ckpt = torch.load(exp_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    val_metrics = evaluate(model, val_loader, device, loss_fn, args, compute_geometry=args.val_compute_geometry)
    test_metrics = evaluate(model, test_loader, device, loss_fn, args, compute_geometry=args.test_compute_geometry)
    final = {
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "best_val_at_training_time": best_val,
        "last_val_at_final_epoch": last_val,
        "best_val_recomputed_after_reload": val_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "rho_f_final": float(args.rho_f),
        "elapsed_sec": time.time() - t0,
    }
    print("[FINAL]")
    print(json.dumps(final, indent=2, ensure_ascii=False))
    with open(exp_dir / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
