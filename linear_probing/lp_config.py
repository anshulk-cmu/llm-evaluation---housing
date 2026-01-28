### RENAMED: lp_config.py
# -*- coding: utf-8 -*-
"""
Configuration for Linear Probing Pipeline
"""
import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_PATH = PROJECT_ROOT / "data" / "pairs_20pct_price_diff.csv"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "checkpoints"

# Activation storage
ACTIVATIONS_DIR = PROJECT_ROOT / "data" / "activations"
ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)

# Probe results
PROBE_RESULTS_DIR = PROJECT_ROOT / "data" / "probe_results"
PROBE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODEL SETTINGS
# ============================================================================

# Model configurations
# use_transformerlens: True = use TransformerLens, False = use manual HuggingFace hooks
MODELS = {
    "llama-3.2-3b": {
        "hf_name": "meta-llama/Llama-3.2-3B-Instruct",
        "local_path": PROJECT_ROOT / "models" / "Llama-3.2-3B-Instruct",
        "n_layers": 28,
        "d_model": 3072,
        "use_transformerlens": True,  # Supported by TransformerLens
    },
    "qwen3-4b": {
        "hf_name": "Qwen/Qwen3-4B-Instruct-2507",
        "local_path": PROJECT_ROOT / "models" / "Qwen3-4B-Instruct-2507",
        "n_layers": 36,
        "d_model": 2560,
        "use_transformerlens": False,  # NOT supported - use manual HF extraction
    },
}

# ============================================================================
# EXTRACTION SETTINGS
# ============================================================================

# Which layers to extract (None = all layers)
LAYERS_TO_EXTRACT = None  # Will extract all layers

# Hook pattern for residual stream post-attention
HOOK_PATTERN = "hook_resid_post"

# Token position to extract (-1 = last token)
TOKEN_POSITION = -1

# Batch size for extraction (adjust based on GPU memory)
EXTRACTION_BATCH_SIZE = 50  # Optimized for A6000 (48GB)

# Checkpoint frequency
EXTRACTION_CHECKPOINT_INTERVAL = 500

# ============================================================================
# PROBING SETTINGS
# ============================================================================

# Features to probe for (binary labels derived from data)
PROBE_FEATURES = [
    "bathrooms_p1_more",   # Property 1 has more bathrooms
    "bedrooms_p1_more",    # Property 1 has more bedrooms
    "sqft_p1_larger",      # Property 1 has larger square footage
    "lot_p1_larger",       # Property 1 has larger lot size
    "year_p1_newer",       # Property 1 is newer
    "price_p1_higher",     # Property 1 has higher price (ground truth)
]

# Data split ratios
TRAIN_RATIO = 0.70  # 70% for training (with CV for hyperparameter selection)
VAL_RATIO = 0.10    # 10% for validation
TEST_RATIO = 0.20   # 20% for final unbiased evaluation

# Cross-validation settings (used within training set only)
CV_FOLDS = 5
PROBE_MAX_ITER = 2000  # Base iterations (will be multiplied for final model)
PROBE_RANDOM_STATE = 42

# Regularization values to try
PROBE_CS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# Statistical settings
BOOTSTRAP_ITERATIONS = 1000
CONFIDENCE_LEVEL = 0.95
SIGNIFICANCE_THRESHOLD = 0.05

# ============================================================================
# PROMPT STRATEGIES (will be updated based on experiment results)
# ============================================================================

# This will be set based on winning prompt from experiments
WINNING_PROMPT_STRATEGY = None  # e.g., "2_fewshot_cot_temp0"

# Import prompt functions from parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))
