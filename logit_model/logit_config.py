# -*- coding: utf-8 -*-
"""
Configuration for Logistic Regression Price Predictor on LLM Activations
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ACTIVATIONS_DIR = DATA_DIR / "activations"
RESULTS_DIR = DATA_DIR / "logit_results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Create directories if they don't exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODEL SPECIFICATIONS
# ============================================================================

MODELS = {
    "llama-3.2-3b": {
        "n_layers": 28,
        "d_model": 3072,
    },
    "qwen3-4b": {
        "n_layers": 36,
        "d_model": 2560,
    },
}

# Default condition to use
DEFAULT_CONDITION = "2_fewshot_cot_temp0"

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

# Data splits
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20
RANDOM_STATE = 42

# Logistic regression hyperparameters
LOGIT_CS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
LOGIT_MAX_ITER = 2000

# ============================================================================
# STATISTICAL PARAMETERS
# ============================================================================

CONFIDENCE_LEVEL = 0.95
SIGNIFICANCE_THRESHOLD = 0.05

# ============================================================================
# BASELINE REFERENCES
# ============================================================================

BASELINE_CHANCE = 0.50
BASELINE_LLM_OUTPUT = 0.60
BASELINE_ZESTIMATE = 0.776
