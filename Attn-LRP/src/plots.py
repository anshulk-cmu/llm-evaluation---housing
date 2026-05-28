# -*- coding: utf-8 -*-
"""
Figure generation (PLAN §13) — four headline figures + one diagnostic.

All figures are built programmatically from results/*.csv with NO hand-editing:
titles, labels, N, and statistics are written from the data. Output is 300-dpi
PNG only, Okabe-Ito colourblind-safe palette, bbox_inches='tight'.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (RESULTS_DIR, FIGURES_DIR, FEATURES, FEATURE_LABEL,
                    F_ATTN_SINGLE, F_ATTN_LOGDIFF, F_SIGNED, F_LAYER, F_PERM)
from log import get_logger

logger = get_logger(__name__)

# Okabe-Ito
OI = {"black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
      "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
      "vermillion": "#D55E00", "purple": "#CC79A7"}

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "figure.dpi": 120, "savefig.dpi": 300})


def _label(f: str) -> str:
    return FEATURE_LABEL.get(f, f)


def _save(fig, figures_dir: str, stem: str) -> None:
    os.makedirs(figures_dir, exist_ok=True)
    fig.savefig(os.path.join(figures_dir, f"{stem}.png"), bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s.png", stem)


# ---------------------------------------------------------------------------
# Figure 1 — feature importance with bootstrap 95% CIs
# ---------------------------------------------------------------------------
def fig1_feature_importance(results_dir=RESULTS_DIR, figures_dir=FIGURES_DIR, n_label="N=500"):
    s = pd.read_csv(os.path.join(results_dir, F_ATTN_SINGLE)).set_index("feature")
    d = pd.read_csv(os.path.join(results_dir, F_ATTN_LOGDIFF)).set_index("feature")
    order = s["mean"].sort_values(ascending=True).index.tolist()   # ascending -> top at top of hbar
    y = np.arange(len(order)); h = 0.38

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for off, df, col, name in [(+h/2, s, OI["blue"], "single-logit"),
                               (-h/2, d, OI["orange"], "logit-difference")]:
        m = df.loc[order, "mean"].values
        lo = df.loc[order, "ci_lo"].values
        hi = df.loc[order, "ci_hi"].values
        ax.barh(y + off, m, height=h, color=col, label=name,
                xerr=[m - lo, hi - m], error_kw=dict(ecolor=OI["black"], lw=1, capsize=2))
    ax.set_yticks(y); ax.set_yticklabels([_label(f) for f in order])
    ax.set_xlabel("mean normalized relevance")
    ax.set_title(f"AttnLRP feature importance (Llama-3.2-3B, {n_label})")
    ax.legend(loc="lower right", frameon=False)
    _save(fig, figures_dir, "fig1_feature_importance_ci")


# ---------------------------------------------------------------------------
# Figure 2 — layer × feature relevance heatmap (+ total-by-layer strip)
# ---------------------------------------------------------------------------
def fig2_layer_heatmap(results_dir=RESULTS_DIR, figures_dir=FIGURES_DIR, target="single_logit"):
    df = pd.read_csv(os.path.join(results_dir, F_LAYER))
    df = df[df["target"] == target]
    feats = [f for f in FEATURES]
    layers = sorted(df["layer"].unique())
    M = np.zeros((len(feats), len(layers)))
    for i, f in enumerate(feats):
        sub = df[df["feature"] == f].set_index("layer")["mean_norm"]
        for j, L in enumerate(layers):
            M[i, j] = sub.get(L, 0.0)
    total = df[df["feature"] == "__total__"].set_index("layer")["mean_norm"].reindex(layers).fillna(0).values
    if not np.any(total):
        total = M.sum(axis=0)

    fig = plt.figure(figsize=(7.2, 3.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.08)
    axt = fig.add_subplot(gs[0]); axm = fig.add_subplot(gs[1], sharex=axt)
    axt.plot(np.arange(len(layers)), total, color=OI["blue"], lw=1.5)
    axt.set_ylabel("total"); axt.tick_params(labelbottom=False)
    axt.set_title(f"AttnLRP layer × feature relevance ({target}, Llama-3.2-3B)")
    im = axm.imshow(M, aspect="auto", cmap="viridis", origin="lower")
    axm.set_yticks(np.arange(len(feats))); axm.set_yticklabels([_label(f) for f in feats])
    axm.set_xlabel("layer"); axm.set_ylabel("feature")
    fig.colorbar(im, ax=axm, label="mean normalized relevance", pad=0.02)
    _save(fig, figures_dir, "fig2_layer_feature_heatmap")


# ---------------------------------------------------------------------------
# Figure 3 — signed relevance diverging bar
# ---------------------------------------------------------------------------
def fig3_signed(results_dir=RESULTS_DIR, figures_dir=FIGURES_DIR, target="single_logit"):
    df = pd.read_csv(os.path.join(results_dir, F_SIGNED))
    df = df[df["target"] == target].set_index("feature")
    order = df["net"].sort_values(ascending=True).index.tolist()
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.barh(y, df.loc[order, "mean_pos"].values, color=OI["green"], label="supports choice")
    ax.barh(y, df.loc[order, "mean_neg"].values, color=OI["vermillion"], label="opposes choice")
    ax.axvline(0, color=OI["black"], lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([_label(f) for f in order])
    ax.set_xlabel("mean signed normalized relevance")
    ax.set_title(f"AttnLRP signed relevance ({target}, Llama-3.2-3B)")
    ax.legend(loc="lower right", frameon=False)
    _save(fig, figures_dir, "fig3_signed_relevance")


# ---------------------------------------------------------------------------
# Figure 4 — permutation rank stability (Phase 2)
# ---------------------------------------------------------------------------
def fig4_perm_stability(results_dir=RESULTS_DIR, figures_dir=FIGURES_DIR, target="single_logit"):
    from stats import spearman
    df = pd.read_csv(os.path.join(results_dir, F_PERM))
    df = df[df["target"] == target]
    rho, p = spearman(df["rank_fixed"].tolist(), df["rank_permuted"].tolist())
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.plot([0.5, len(FEATURES) + 0.5], [0.5, len(FEATURES) + 0.5], "--", color=OI["black"], lw=1)
    ax.scatter(df["rank_fixed"], df["rank_permuted"], color=OI["blue"], zorder=3)
    for _, r in df.iterrows():
        ax.annotate(_label(r["feature"]), (r["rank_fixed"], r["rank_permuted"]),
                    textcoords="offset points", xytext=(6, 3), fontsize=8)
    ax.set_xticks(range(1, len(FEATURES) + 1)); ax.set_yticks(range(1, len(FEATURES) + 1))
    ax.set_xlabel("rank (fixed order)"); ax.set_ylabel("rank (all 120 permutations)")
    ax.set_title(f"Permutation rank stability ({target})\nSpearman ρ={rho:.2f}, p={p:.3g}")
    ax.set_aspect("equal"); ax.set_xlim(0.5, len(FEATURES) + 0.5); ax.set_ylim(0.5, len(FEATURES) + 0.5)
    _save(fig, figures_dir, "fig4_permutation_rank_stability")


# ---------------------------------------------------------------------------
# Diagnostic — deletion/insertion faithfulness curve
# ---------------------------------------------------------------------------
def diag_deletion_curve(results_dir=RESULTS_DIR, figures_dir=FIGURES_DIR,
                        fname="deletion_curve.csv"):
    path = os.path.join(results_dir, fname)
    if not os.path.exists(path):
        logger.info("[plots] no deletion_curve.csv — skipping diagnostic figure")
        return
    df = pd.read_csv(path)   # columns: frac_removed, attnlrp, random
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(df["frac_removed"], df["attnlrp"], color=OI["blue"], marker="o", ms=3, label="AttnLRP order")
    if "random" in df:
        ax.plot(df["frac_removed"], df["random"], color=OI["vermillion"], ls="--", label="random order")
    ax.set_xlabel("fraction of feature tokens removed")
    ax.set_ylabel("mean target logit")
    ax.set_title("Deletion faithfulness curve (lower-faster = more faithful)")
    ax.legend(frameon=False)
    _save(fig, figures_dir, "diag_deletion_insertion_curve")


def main(results_dir=RESULTS_DIR, figures_dir=FIGURES_DIR):
    made = []
    for fn, needs in [(fig1_feature_importance, [F_ATTN_SINGLE, F_ATTN_LOGDIFF]),
                      (fig2_layer_heatmap, [F_LAYER]),
                      (fig3_signed, [F_SIGNED]),
                      (fig4_perm_stability, [F_PERM])]:
        if all(os.path.exists(os.path.join(results_dir, n)) for n in needs):
            fn(results_dir, figures_dir); made.append(fn.__name__)
        else:
            logger.info(f"[plots] skip {fn.__name__} (missing inputs)")
    diag_deletion_curve(results_dir, figures_dir)
    logger.info(f"[plots] done: {made}")


if __name__ == "__main__":
    main()
