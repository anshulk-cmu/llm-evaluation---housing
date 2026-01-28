# -*- coding: utf-8 -*-
"""
Configuration for LLM Real Estate Valuation Experiment
Edit values here to change experiment settings
"""

# ============================================================================
# DATA SETTINGS
# ============================================================================

DATA_PATH = "data/pairs_20pct_price_diff.csv"
RESULTS_DIR = "data/results"
CHECKPOINT_DIR = "data/checkpoints"
LOG_DIR = "logs"

# Checkpoint settings
CHECKPOINT_INTERVAL = 100  # Save checkpoint every N samples

# Sampling
SAMPLE_SIZE = 5130  # Change this to run on different sample sizes
RANDOM_STATE = 42

# ============================================================================
# MODEL SETTINGS
# ============================================================================

# Generation settings
MAX_NEW_TOKENS = 1024
BATCH_SIZE = 50  # Adjust based on GPU memory

# Models to test (if using inference.py)
MODELS = {
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct"
}

# Local model cache directory
LOCAL_MODEL_DIR = "models"

# ============================================================================
# SYSTEM MESSAGE
# ============================================================================

SYSTEM_MSG = """You are a professional residential real-estate appraiser. You are given two home listings from the SAME ZIP CODE and the SAME MONTH.
Your task is to determine which home is more likely to have a HIGHER CURRENT MARKET VALUE.
You must rely only on the provided listing attributes. Do NOT use external knowledge, price indexes, or unstated assumptions. Be consistent, careful, and comparative."""

# ============================================================================
# EXPERIMENTAL CONDITIONS
# ============================================================================

# Each condition defines:
# - prompt_fn: which prompt builder to use (imported from prompts.py)
# - do_sample: whether to use sampling
# - temperature, top_p: sampling parameters (if do_sample=True)

CONDITIONS = {
    "0_baseline_temp0": {
        "prompt_fn": "build_prompt_baseline",
        "do_sample": False
    },
    "1_zeroshot_cot_temp0": {
        "prompt_fn": "build_prompt_zero_shot_cot",
        "do_sample": False
    },
    "2_fewshot_cot_temp0": {
        "prompt_fn": "build_prompt_few_shot_cot",
        "do_sample": False
    },
    "3_zeroshot_cot_topp01": {
        "prompt_fn": "build_prompt_zero_shot_cot",
        "temperature": 0.1,
        "top_p": 0.1,
        "do_sample": True
    },
    "4_fewshot_cot_topp01": {
        "prompt_fn": "build_prompt_few_shot_cot",
        "temperature": 0.1,
        "top_p": 0.1,
        "do_sample": True
    },
}
