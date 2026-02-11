# -*- coding: utf-8 -*-
"""
Configuration for Behavioral Macro Mapping Experiment

All constants, paths, perturbation ranges, and feature definitions live here.
Mirrors the structure of the project's existing config.py.
"""

import sys
import os
import logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(step_name: str, level=logging.INFO) -> logging.Logger:
    """
    Create a logger for a pipeline step.
    All output goes to stdout/stderr (captured by SLURM .out/.err files).

    Format: [TIMESTAMP] [STEP_NAME] LEVEL — message

    Usage:
        from mm_config import setup_logger
        logger = setup_logger("Step5_ReasoningStages")
        logger.info("Loaded 100 CoT responses")
        logger.warning("Missing feature column: lot")
        logger.error("File not found: ...")
    """
    logger = logging.getLogger(step_name)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Console handler (stdout → SLURM .out)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(name)s] %(levelname)s — %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Also log warnings/errors to stderr (→ SLURM .err)
    err_handler = logging.StreamHandler(sys.stderr)
    err_handler.setLevel(logging.WARNING)
    err_handler.setFormatter(formatter)
    logger.addHandler(err_handler)

    return logger

# ============================================================================
# PATHS
# ============================================================================

# Input: existing experiment results (few-shot CoT, temp 0)
RESULTS_DIR = "data/results"
DATA_PATH = "data/pairs_20pct_price_diff.csv"

# Output: macro mapping artifacts
MM_DATA_DIR = "macro_mapping/data"
MM_OUTPUT_DIR = "macro_mapping/outputs"
MM_GRAPHS_DIR = "macro_mapping/outputs/graphs"
MM_TABLES_DIR = "macro_mapping/outputs/tables"

# ============================================================================
# SAMPLE SELECTION (Step 1)
# ============================================================================

N_TOTAL = 100
N_CORRECT = 60
N_INCORRECT = 40
RANDOM_STATE = 42

# Stratification bins for price gap difficulty
# "easy" = large price gap, "hard" = small price gap (near 20% threshold)
DIFFICULTY_BINS = {
    "easy": (0.50, 1.00),    # price ratio >= 50% apart
    "medium": (0.30, 0.50),  # 30-50% apart
    "hard": (0.20, 0.30),    # 20-30% apart (near threshold)
}

# ============================================================================
# MODEL SETTINGS (Step 4)
# ============================================================================

# Which model(s) to run perturbation experiments on
MODELS_TO_TEST = ["llama-3.2-3b", "qwen3-4b"]

# Best condition from original experiments
CONDITION = "2_fewshot_cot_temp0"

# Generation settings (same as main experiment for consistency)
MAX_NEW_TOKENS = 2048
BATCH_SIZE = 1  # Sequential for perturbation (need to check flip after each)

# ============================================================================
# PERTURBATION SETTINGS (Step 3)
# ============================================================================

# Feature perturbation increments
# Each list defines the cumulative increase applied to the "losing" property.
# Fine-grained uniform steps for precise flip-point detection:
#   - Bedrooms:  +1 at a time (cap 10)
#   - Bathrooms: +1 at a time (cap 10)
#   - YearBuilt: +2yr at a time (cap 2025)
#   - Lot:       +100 sqft at a time (max additional 5000 sqft)
#
# Data context (within-pair diffs from 5,130 pairs):
#   - Bedrooms:  median diff = 1, 95th pct = 2  → most flips expected by step 2-3
#   - Bathrooms: median diff = 1, 95th pct = 2  → most flips expected by step 2-3
#   - YearBuilt: median diff = 17yr, 95th pct = 65yr → most flips by step 8-15
#   - Lot:       median diff = 1964 sqft, 95th pct = 9110 sqft → most flips by step 20
#
# Round-by-round early stopping in Step 4 ensures only stubborn pairs
# reach the higher steps — actual query count << theoretical max.
PERTURBATION_STEPS = {
    "bedrooms":   list(range(1, 10)),              # +1, +2, ..., +9
    "bathrooms":  list(range(1, 10)),              # +1, +2, ..., +9
    "yearBuilt":  list(range(2, 96, 2)),           # +2, +4, +6, ..., +94
    "lot":        list(range(100, 5001, 100)),      # +100, +200, ..., +5000
}

# Absolute caps — never push a feature beyond these values
FEATURE_CAPS = {
    "bedrooms":  10,     # cap at 10 bedrooms
    "bathrooms": 10,     # cap at 10 bathrooms
    "yearBuilt": 2025,   # cap at 2025
    "lot":       43560,  # 1 acre (max in dataset)
}

# ============================================================================
# FEATURES UNDER STUDY
# ============================================================================

# The four numeric features we perturb and analyze
TARGET_FEATURES = ["bedrooms", "bathrooms", "yearBuilt", "lot"]

# Column suffixes in the dataset for property 1 and property 2
PROP_SUFFIXES = ("_1", "_2")

# ============================================================================
# REASONING STAGE TAGS (Step 5)
# ============================================================================

# Expected reasoning stages to look for in CoT text
# These are refined during Step 5 based on actual model outputs
REASONING_STAGES = [
    "feature_reading",       # Model lists features
    "pairwise_comparison",   # Model compares each feature head-to-head
    "weighing_aggregation",  # Model weighs which advantages matter more
    "final_decision",        # Model commits to CHOICE
]

# ============================================================================
# VALIDATION (Step 9)
# ============================================================================

# Path to existing raw feature logit results for comparison
RAW_LOGIT_RESULTS_DIR = "data/raw_feature_logit_results"
