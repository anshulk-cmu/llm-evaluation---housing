# -*- coding: utf-8 -*-
"""
Central configuration for the Attn-LRP housing study.

Implements the locked decisions from ../PLAN.md:
  - AttnLRP only (no CP-LRP), two targets (single-logit, logit-difference)
  - One model: Llama-3.2-3B-Instruct  (Qwen3 dropped — LXT first-token-skew defect)
  - Five features: lot, bathrooms, bedrooms, year built, property type
  - N = 500 valid pairs, bf16 backward, full 5! = 120 permutation control (Phase 2)

Nothing here is GPU-specific; importing this module is safe on any machine.
"""

from __future__ import annotations
import os

# ---------------------------------------------------------------------------
# Paths (computed from this file's location — no hard-coded absolute paths)
# ---------------------------------------------------------------------------
SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
PKG_DIR     = os.path.dirname(SRC_DIR)                 # .../Attn-LRP
REPO_ROOT   = os.path.dirname(PKG_DIR)                 # .../LLM_Product_Valuation
HOUSING_REF = os.path.join(REPO_ROOT, "llm-evaluation-housing-reference")

DATA_PATH    = os.path.join(HOUSING_REF, "data", "pairs_20pct_price_diff.csv")
RESULTS_DIR  = os.path.join(PKG_DIR, "results")
FIGURES_DIR  = os.path.join(PKG_DIR, "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Model (sole model — see PLAN.md §9.5). Qwen3 deliberately excluded.
# ---------------------------------------------------------------------------
MODEL_ID         = "meta-llama/Llama-3.2-3B-Instruct"
# Optional local snapshot dir; if it exists it is preferred over the HF hub id.
LOCAL_MODEL_DIR  = os.environ.get("ATTNLRP_MODEL_DIR",
                                  os.path.join(REPO_ROOT, "models", "Llama-3.2-3B-Instruct"))
DTYPE            = "bfloat16"          # mandatory on the 12 GB card (PLAN §6.4)
N_LAYERS_EXPECTED = 28                 # Llama-3.2-3B; asserted at runtime, not assumed

# ---------------------------------------------------------------------------
# Features (PLAN §9.2). Keys are internal; LABELS are what appears in the prompt.
# `zpid` is NOT a feature — it is the should-be-~0 control (PLAN §15.2).
# ---------------------------------------------------------------------------
FEATURES = ["lot", "bathrooms", "bedrooms", "year_built", "property_type"]

FEATURE_LABEL = {
    "lot":           "lot",
    "bathrooms":     "bathrooms",
    "bedrooms":      "bedrooms",
    "year_built":    "year built",
    "property_type": "property type",
}

# Source columns in pairs_20pct_price_diff.csv (suffix _1 / _2 added at build time).
# lot is special: value = f"{lot} {lotUnit}".
FEATURE_COLUMNS = {
    "lot":           ("lot", "lotUnit"),
    "bathrooms":     ("bathrooms",),
    "bedrooms":      ("bedrooms",),
    "year_built":    ("yearBuilt",),
    "property_type": ("type",),
}

# ---------------------------------------------------------------------------
# Sampling / statistics (PLAN §10, §12)
# ---------------------------------------------------------------------------
N_VALID_TARGET = 500          # locked
SEED           = 42
BOOTSTRAP_B    = 10_000       # bootstrap resamples for 95% CIs (PLAN §12.2)
CI_ALPHA       = 0.05
EPS            = 1e-6         # LRP stabiliser (PLAN §3.2)

# Conservation tolerance (PLAN §14 / §15.1): median relative residual under bf16.
CONSERVATION_REL_TOL = 0.02

# ---------------------------------------------------------------------------
# Targets (PLAN §5)
# ---------------------------------------------------------------------------
TARGETS = ["single_logit", "logit_diff"]

# ---------------------------------------------------------------------------
# Parallelism (local box: 24-core CPU, 64 GB RAM, 1x RTX 5070 Ti 12 GB)
# GPU is batched (BATCH_SIZE prompts per forward+backward); CPU ops use threads.
# Lower ATTNLRP_BATCH if VRAM is tight; raise if headroom remains.
# ---------------------------------------------------------------------------
CPU_THREADS = int(os.environ.get("ATTNLRP_THREADS", str(min(24, os.cpu_count() or 8))))
BATCH_SIZE  = int(os.environ.get("ATTNLRP_BATCH", "8"))

# ---------------------------------------------------------------------------
# Published housing attribution ranks on Qwen3 (PLAN §12.4, main.tex
# tab:housing-attribution-ranks). Used for the CROSS-MODEL Spearman check.
# ---------------------------------------------------------------------------
HOUSING_PUBLISHED_RANKS = {
    #              IG  SHAP Occ   (rank 1 = most important)
    "lot":           {"IG": 1, "KernelSHAP": 1, "Occlusion": 1, "avg": 1.00},
    "year_built":    {"IG": 5, "KernelSHAP": 2, "Occlusion": 2, "avg": 3.00},
    "bedrooms":      {"IG": 4, "KernelSHAP": 4, "Occlusion": 3, "avg": 3.67},
    "bathrooms":     {"IG": 2, "KernelSHAP": 5, "Occlusion": 4, "avg": 3.67},
    "property_type": {"IG": 3, "KernelSHAP": 3, "Occlusion": 5, "avg": 3.67},
}

# Identity-dilution reference rates (PLAN §12.5, main.tex) — informational.
IDENTICAL_RATE = {"bathrooms": 0.401, "bedrooms": 0.393, "parking_type": 0.976}

# ---------------------------------------------------------------------------
# Output file names (PLAN §18)
# ---------------------------------------------------------------------------
F_ATTN_SINGLE   = "attnlrp_single.csv"
F_ATTN_LOGDIFF  = "attnlrp_logitdiff.csv"
F_SIGNED        = "signed_relevance.csv"
F_LAYER         = "layer_feature_relevance.csv"
F_CROSS         = "cross_method_correlation.csv"
F_PERM          = "permutation_stability.csv"
F_SANITY        = "conservation_and_sanity.csv"
# Per-sample raw (for reproducibility / bootstrap) — one row per (sample,target,feature)
F_PERSAMPLE     = "per_sample_relevance.csv"


def banner(msg: str) -> None:
    from log import banner as _banner
    _banner(msg)
