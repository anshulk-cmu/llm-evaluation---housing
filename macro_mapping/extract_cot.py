# -*- coding: utf-8 -*-
"""
Step 2: Extract Chain-of-Thought Reasoning via Re-Inference (Dual-GPU)

Re-runs inference on the 100 selected pairs using the new REASON-augmented prompt.
This is required because existing Llama results contain no CoT text (just "CHOICE: X").
Even for Qwen (which had CoT), we re-run to establish a consistent baseline under
the new prompt format.

After re-inference, correct/incorrect labels are RE-DETERMINED from the new predictions.
The original labels from Step 1 are NOT reused — the REASON field may change behavior.

Uses 2x A6000 GPUs with the same dual-GPU threading pattern as the original experiment.

Usage:
    python macro_mapping/extract_cot.py --model llama-3.2-3b
    python macro_mapping/extract_cot.py --model qwen3-4b
"""

import os
import sys
import gc
import time
import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from threading import Thread
from queue import Queue
from tqdm import tqdm

# Add parent to path for inference.py imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inference
from inference import load_model, unload_model, process_batch_on_gpu
from mm_utils import build_perturbation_prompt, parse_response, SYSTEM_MSG
from mm_config import CONDITION, RANDOM_STATE, MAX_NEW_TOKENS, setup_logger

logger = setup_logger("Step2_ExtractCoT")

# Override inference module's token limit for macro mapping
# inference.py imports MAX_NEW_TOKENS=1024 from config.py; we need 2048
# SYSTEM_MSG text is identical but override to ensure mm_utils version is used
inference.MAX_NEW_TOKENS = MAX_NEW_TOKENS
inference.SYSTEM_MSG = SYSTEM_MSG


# ============================================================================
# PATHS
# ============================================================================

MM_DATA_DIR = Path(__file__).resolve().parent / "data"
MM_DATA_DIR.mkdir(exist_ok=True)

# Batch size per GPU — 50 per GPU = 100 total, same as original experiment
BATCH_SIZE = 50

# Model name mapping (short name -> HuggingFace name)
MODEL_HF_NAMES = {
    'llama-3.2-3b': 'meta-llama/Llama-3.2-3B-Instruct',
    'qwen3-4b': 'Qwen/Qwen3-4B-Instruct-2507',
}


# ============================================================================
# DUAL-GPU INFERENCE (mirrors experiment.py / inference.py pattern)
# ============================================================================

def run_dual_gpu_inference(model_name: str, prompts: list) -> list:
    """
    Run inference on both GPUs simultaneously using pre-loaded models.
    Mirrors the pattern from experiment.py run_dual_gpu_batch().

    Args:
        model_name: HuggingFace model name
        prompts: List of prompt strings

    Returns:
        List of response strings (same order as input prompts)
    """
    condition_config = {"do_sample": False}  # Greedy decoding, temp=0

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    logger.info(f"  GPUs available: {gpu_count}")

    if gpu_count < 2:
        logger.info("  Falling back to single-GPU inference")
        model, tokenizer = load_model(model_name, device=0)
        responses = _run_single_gpu(model, tokenizer, prompts, condition_config)
        unload_model(model, tokenizer)
        return responses

    # Load model on both GPUs
    logger.info(f"  Loading model on GPU 0...")
    model_0, tokenizer = load_model(model_name, device=0)
    logger.info(f"  Loading model on GPU 1...")
    model_1, _ = load_model(model_name, device=1)

    logger.info(f"  Both models loaded. Processing {len(prompts)} prompts...")

    # Split prompts between GPUs
    mid = len(prompts) // 2
    batch_0 = prompts[:mid]
    batch_1 = prompts[mid:]

    # Queue for results
    results_queue = Queue()

    # Create threads
    thread_0 = Thread(
        target=process_batch_on_gpu,
        args=(model_0, tokenizer, batch_0, condition_config, BATCH_SIZE, 0, results_queue)
    )
    thread_1 = Thread(
        target=process_batch_on_gpu,
        args=(model_1, tokenizer, batch_1, condition_config, BATCH_SIZE, 1, results_queue)
    )

    # Run both GPUs simultaneously
    thread_0.start()
    thread_1.start()
    thread_0.join()
    thread_1.join()

    # Collect results in correct order
    results = {}
    while not results_queue.empty():
        gpu_id, responses = results_queue.get()
        results[gpu_id] = responses

    all_responses = []
    if 0 in results and results[0] is not None:
        all_responses.extend(results[0])
    if 1 in results and results[1] is not None:
        all_responses.extend(results[1])

    # Cleanup
    unload_model(model_0, tokenizer)
    unload_model(model_1, None)

    return all_responses


def _run_single_gpu(model, tokenizer, prompts, condition_config):
    """Fallback single-GPU inference."""
    from inference import run_batch_inference
    return run_batch_inference(model, tokenizer, prompts, condition_config, BATCH_SIZE)


# ============================================================================
# MAIN
# ============================================================================

def main(model: str = "llama-3.2-3b"):
    logger.info(f"Step 2: Extract CoT Reasoning for {model}")
    logger.info(f"Using REASON-augmented prompt (full re-inference)")

    # Load selected samples from Step 1
    samples_file = MM_DATA_DIR / f"selected_samples_{model}.csv"
    if not samples_file.exists():
        logger.error(f"{samples_file} not found. Run Step 1 first.")
        sys.exit(1)

    samples_df = pd.read_csv(samples_file)
    logger.info(f"Loaded {len(samples_df)} selected samples")

    # Build prompts (unperturbed baseline — no feature changes)
    logger.info("Building prompts...")
    prompts = []
    for _, row in samples_df.iterrows():
        prompt = build_perturbation_prompt(row)  # No perturbation = baseline
        prompts.append(prompt)
    logger.info(f"Built {len(prompts)} prompts")

    # Run inference
    hf_name = MODEL_HF_NAMES[model]
    logger.info(f"\nRunning dual-GPU inference with {hf_name}...")
    start_time = time.time()

    responses = run_dual_gpu_inference(hf_name, prompts)

    elapsed = time.time() - start_time
    logger.info(f"Inference complete: {elapsed:.1f}s ({elapsed/len(prompts):.2f}s per sample)")

    # Parse responses
    logger.info("\nParsing responses...")
    parsed = [parse_response(r) for r in responses]

    # Build results DataFrame with re-determined correctness
    results = []
    for i, (parsed_r, (_, row)) in enumerate(zip(parsed, samples_df.iterrows())):
        # Ground truth from prices
        p1_price = row['price.y_1']
        p2_price = row['price.y_2']
        true_choice = 1 if p1_price > p2_price else 2

        new_choice = parsed_r['choice']
        is_valid = not np.isnan(new_choice) if isinstance(new_choice, float) else new_choice is not None
        new_correct = (int(new_choice) == true_choice) if is_valid else False

        results.append({
            'pair_index': i,
            'zpid_1': row['zpid_1'],
            'zpid_2': row['zpid_2'],
            'price_1': p1_price,
            'price_2': p2_price,
            'true_choice': true_choice,
            'new_pred_choice': int(new_choice) if is_valid else None,
            'new_is_correct': new_correct,
            'old_pred_choice': row.get('pred_choice', None),
            'old_is_correct': row.get('is_correct', None),
            'reason': parsed_r['reason'],
            'full_cot': parsed_r['full_text'],
            'difficulty': row.get('difficulty', None),
            'feature_profile': row.get('feature_profile', None),
        })

    results_df = pd.DataFrame(results)

    # Report baseline shift
    old_correct = results_df['old_is_correct'].sum()
    new_correct_count = results_df['new_is_correct'].sum()
    flipped = 0
    for _, r in results_df.iterrows():
        if pd.notna(r['new_pred_choice']) and pd.notna(r['old_pred_choice']):
            if int(r['new_pred_choice']) != int(r['old_pred_choice']):
                flipped += 1
    unparseable = results_df['new_pred_choice'].isna().sum()

    logger.info(f"\n{'='*60}")
    logger.info(f"  BASELINE RE-INFERENCE RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"  Old accuracy: {old_correct}/{len(results_df)} "
          f"({100*old_correct/len(results_df):.1f}%)")
    logger.info(f"  New accuracy: {new_correct_count}/{len(results_df)} "
          f"({100*new_correct_count/len(results_df):.1f}%)")
    logger.info(f"  Predictions flipped by REASON prompt: {flipped}")
    logger.info(f"  Unparseable responses: {unparseable}")
    logger.info(f"  New split: {new_correct_count} correct, "
          f"{len(results_df) - new_correct_count} incorrect")

    # Check reason extraction
    has_reason = results_df['reason'].notna().sum()
    has_cot = (results_df['full_cot'].str.len() > 20).sum()
    logger.info(f"\n  Responses with REASON field: {has_reason}/{len(results_df)}")
    logger.info(f"  Responses with CoT text (>20 chars): {has_cot}/{len(results_df)}")

    # Save main results
    output_file = MM_DATA_DIR / f"cot_responses_{model}.csv"
    results_df.to_csv(output_file, index=False)
    logger.info(f"\nSaved: {output_file}")

    # Save full text responses for manual inspection
    text_file = MM_DATA_DIR / f"cot_full_text_{model}.txt"
    with open(text_file, 'w') as f:
        for r in results:
            f.write(f"{'='*70}\n")
            f.write(f"PAIR {r['pair_index']} | correct={r['new_is_correct']} "
                    f"| choice={r['new_pred_choice']}\n")
            f.write(f"{'='*70}\n")
            f.write(r['full_cot'] or '(empty)')
            f.write('\n\n')
    logger.info(f"Saved full text: {text_file}")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: Extract CoT reasoning")
    parser.add_argument('--model', type=str, default='llama-3.2-3b',
                        choices=['llama-3.2-3b', 'qwen3-4b'],
                        help='Model to run inference on')
    args = parser.parse_args()
    main(model=args.model)
