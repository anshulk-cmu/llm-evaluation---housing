# -*- coding: utf-8 -*-
"""
Runtime sanity / validation checks (PLAN §14–§15).

These are model-agnostic where possible: faithfulness takes a `target_logit_fn`
callback so it can be unit-tested without a GPU.
"""

from __future__ import annotations
from typing import Callable, Dict, List, Sequence
import numpy as np

from config import CONSERVATION_REL_TOL, FEATURES
from stats import spearman

# np.trapz was removed in NumPy 2.x -> prefer np.trapezoid, fall back for 1.x.
_TRAPZ = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def conservation_summary(residuals: Sequence[float]) -> Dict[str, float]:
    """Median relative residual of (sum R0 vs injected target); pass vs tolerance."""
    r = np.asarray(list(residuals), dtype=float)
    med = float(np.median(r)) if r.size else float("nan")
    return {"median_rel_residual": med, "tolerance": CONSERVATION_REL_TOL,
            "pass": bool(med <= CONSERVATION_REL_TOL)}


def token_count_neutrality(per_feature_relevance: Dict[str, List[float]],
                           per_feature_token_counts: Dict[str, List[int]]) -> Dict[str, float]:
    """Correlate relevance with token count across all (sample,feature) points.

    Should be ~0 if MEAN aggregation neutralised token-count bias (PLAN §15.3).
    """
    rel, cnt = [], []
    for f in FEATURES:
        rel.extend(per_feature_relevance.get(f, []))
        cnt.extend(per_feature_token_counts.get(f, []))
    if len(rel) < 3:
        return {"spearman_rho": float("nan"), "p": float("nan"), "n": len(rel)}
    rho, p = spearman(rel, cnt)
    return {"spearman_rho": rho, "p": p, "n": len(rel)}


# ---------------------------------------------------------------------------
# Deletion / insertion faithfulness (PLAN §15.5) — diagnostic figure
# ---------------------------------------------------------------------------
def deletion_curve(order: Sequence[int],
                   target_logit_fn: Callable[[Sequence[int]], float]) -> np.ndarray:
    """Target logit as feature tokens are removed cumulatively in `order`.

    Returns an array of length len(order)+1: [full, after 1 removed, ...].
    A faithful ordering (most-relevant first) drops fastest.
    """
    masked: List[int] = []
    curve = [target_logit_fn(masked)]
    for tok in order:
        masked.append(tok)
        curve.append(target_logit_fn(masked))
    return np.asarray(curve, dtype=float)


def normalized_auc(curve: np.ndarray) -> float:
    """Area under the (normalised) deletion curve; lower = more faithful."""
    base = curve[0]
    span = abs(base) + 1e-9
    norm = (curve - curve.min()) / span
    return float(_TRAPZ(norm) / (len(curve) - 1))


def randomize_top_layers(model, k: int = 4, seed: int = 0):
    """Re-initialise the top-k decoder layers in place (Adebayo sanity, §15.4).

    Returns a restore() closure to put the original weights back.
    """
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    layers = model.model.layers
    saved = {}
    for layer in list(layers)[-k:]:
        for name, p in layer.named_parameters():
            saved[(id(layer), name)] = p.detach().clone()
            with torch.no_grad():
                if p.dim() >= 2:
                    std = float(p.float().std().clamp_min(1e-4))
                    p.copy_(torch.randn(p.shape, generator=g).to(p.dtype).to(p.device) * std)
                else:
                    p.zero_()

    def restore():
        for layer in list(layers)[-k:]:
            for name, p in layer.named_parameters():
                with torch.no_grad():
                    p.copy_(saved[(id(layer), name)].to(p.device))
    return restore
