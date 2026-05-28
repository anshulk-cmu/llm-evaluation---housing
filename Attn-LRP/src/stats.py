# -*- coding: utf-8 -*-
"""
Statistical-robustness layer (PLAN §12).

Inputs are per-sample normalised-relevance matrices X of shape [n_samples, n_features]
with columns ordered as config.FEATURES.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from config import FEATURES, BOOTSTRAP_B, CI_ALPHA, SEED, HOUSING_PUBLISHED_RANKS

try:
    from scipy import stats as _sps
    _HAVE_SCIPY = True
except Exception:                       # pragma: no cover
    _HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# Point summaries
# ---------------------------------------------------------------------------
def mean_std(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return X.mean(axis=0), X.std(axis=0, ddof=1)


def topk_frequency(X: np.ndarray, k: int) -> np.ndarray:
    """Fraction of samples where each feature is in the per-sample top-k."""
    n, F = X.shape
    freq = np.zeros(F)
    order = np.argsort(-X, axis=1)      # descending per row
    topk = order[:, :k]
    for row in topk:
        freq[row] += 1
    return freq / n


def bootstrap_ci(X: np.ndarray, B: int = BOOTSTRAP_B, alpha: float = CI_ALPHA,
                 seed: int = SEED) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Percentile bootstrap CI on each feature's mean (PLAN §12.2)."""
    rng = np.random.default_rng(seed)
    n, F = X.shape
    means = np.empty((B, F))
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        means[b] = X[idx].mean(axis=0)
    lo = np.percentile(means, 100 * (alpha / 2), axis=0)
    hi = np.percentile(means, 100 * (1 - alpha / 2), axis=0)
    return X.mean(axis=0), lo, hi


# ---------------------------------------------------------------------------
# Significance
# ---------------------------------------------------------------------------
def friedman(X: np.ndarray):
    """Omnibus Friedman test across feature columns (PLAN §12.3)."""
    if not _HAVE_SCIPY:
        return float("nan"), float("nan")
    cols = [X[:, j] for j in range(X.shape[1])]
    stat, p = _sps.friedmanchisquare(*cols)
    return float(stat), float(p)


def _bh_fdr(pvals: List[float]) -> List[float]:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    # enforce monotonicity from the largest down
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.clip(ranked, 0, 1)
    return adj.tolist()


def wilcoxon_posthoc(X: np.ndarray, pairs: List[Tuple[str, str]]):
    """Wilcoxon signed-rank per feature pair, with BH-FDR (PLAN §12.3)."""
    if not _HAVE_SCIPY:
        return [{"pair": f"{a} vs {b}", "stat": float("nan"),
                 "p": float("nan"), "p_fdr": float("nan")} for a, b in pairs]
    col = {f: i for i, f in enumerate(FEATURES)}
    rows, raw = [], []
    for a, b in pairs:
        try:
            stat, p = _sps.wilcoxon(X[:, col[a]], X[:, col[b]])
        except ValueError:              # zero differences etc.
            stat, p = float("nan"), 1.0
        rows.append({"pair": f"{a} vs {b}", "stat": float(stat), "p": float(p)})
        raw.append(p)
    for r, padj in zip(rows, _bh_fdr(raw)):
        r["p_fdr"] = float(padj)
    return rows


# ---------------------------------------------------------------------------
# Ranks + cross-method / cross-target correlation
# ---------------------------------------------------------------------------
def ranks_from_means(mean_vec: np.ndarray) -> Dict[str, int]:
    """Rank features 1..F by descending mean (1 = most relevant)."""
    order = np.argsort(-mean_vec)
    rank = {}
    for r, j in enumerate(order, start=1):
        rank[FEATURES[j]] = r
    return rank


def spearman(a: List[float], b: List[float]):
    if not _HAVE_SCIPY:
        return float("nan"), float("nan")
    rho, p = _sps.spearmanr(a, b)
    return float(rho), float(p)


def cross_method_table(attn_mean_vec: np.ndarray) -> List[dict]:
    """Spearman of AttnLRP ranks vs each published housing method (PLAN §12.4).

    NOTE: this is a CROSS-MODEL comparison (AttnLRP on Llama vs IG/SHAP/Occ on
    Qwen3) — interpret per PLAN §12.4.
    """
    attn_rank = ranks_from_means(attn_mean_vec)
    rows = []
    for method in ("IG", "KernelSHAP", "Occlusion", "avg"):
        a = [attn_rank[f] for f in FEATURES]
        b = [HOUSING_PUBLISHED_RANKS[f][method] for f in FEATURES]
        rho, p = spearman(a, b)
        rows.append({"comparison": f"AttnLRP(Llama) vs {method}(Qwen3)",
                     "spearman_rho": rho, "p": p, "cross_model": True})
    return rows


def cross_target_correlation(mean_single: np.ndarray, mean_logdiff: np.ndarray) -> dict:
    rs = [ranks_from_means(mean_single)[f] for f in FEATURES]
    rl = [ranks_from_means(mean_logdiff)[f] for f in FEATURES]
    rho, p = spearman(rs, rl)
    return {"comparison": "single_logit vs logit_diff (same model)",
            "spearman_rho": rho, "p": p, "cross_model": False}
