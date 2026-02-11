# -*- coding: utf-8 -*-
"""
Step 4: Run Perturbation Queries and Record Flip Points (Dual-GPU)

Executes the perturbation plan from Step 3 using round-by-round batched
inference for efficiency:
  - Round 0: Send step_index=0 for ALL 400 pair×feature combos (batched on 2 GPUs)
  - Round 1: Send step_index=1 for only non-flipped combos
  - Round 2: ...
  - Until all combos have either flipped or exhausted their steps

This is dramatically faster than sequential per-pair processing — at most 6 rounds
of batched inference vs ~2000 individual calls.

Early stopping: once a (pair, feature) combo flips, it's excluded from all
subsequent rounds.

Inputs:
  - macro_mapping/data/perturbation_plan_{model}.csv   (from Step 3)
  - macro_mapping/data/selected_samples_{model}.csv    (feature data for prompts)
  - macro_mapping/data/cot_responses_{model}.csv       (baseline predictions)

Outputs:
  - macro_mapping/data/perturbation_responses_{model}.csv
      Every query's result with 13 columns:
        pair_index, is_correct, feature, step_index, increment_label,
        loser_property, original_value, perturbed_value,
        original_choice, new_choice, did_flip, reason, cot_text
  - macro_mapping/data/flip_results_{model}.csv
      One row per (pair, feature) with 9 columns:
        pair_index, is_correct, feature, flipped, flip_step_index,
        flip_increment, original_value, flipped_value, n_steps_tried

Usage:
    python macro_mapping/run_perturbations.py --model llama-3.2-3b
    python macro_mapping/run_perturbations.py --model qwen3-4b
"""

import os
import sys
import gc
import time
import json
import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from threading import Thread
from queue import Queue

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inference
from inference import load_model, unload_model, process_batch_on_gpu
from mm_utils import build_perturbation_prompt, parse_response, SYSTEM_MSG
from mm_config import MAX_NEW_TOKENS, TARGET_FEATURES, setup_logger

logger = setup_logger("Step4_Perturbations")

# Override inference module settings for macro mapping
inference.MAX_NEW_TOKENS = MAX_NEW_TOKENS
inference.SYSTEM_MSG = SYSTEM_MSG

# ============================================================================
# PATHS & SETTINGS
# ============================================================================

MM_DATA_DIR = Path(__file__).resolve().parent / "data"

# Batch size per GPU — same as original experiment
BATCH_SIZE = 50

# Model name mapping
MODEL_HF_NAMES = {
    'llama-3.2-3b': 'meta-llama/Llama-3.2-3B-Instruct',
    'qwen3-4b': 'Qwen/Qwen3-4B-Instruct-2507',
}


# ============================================================================
# DUAL-GPU INFERENCE (persistent models across rounds)
# ============================================================================

def run_inference_round(model_0, model_1, tokenizer, prompts: list) -> list:
    """
    Run one round of batched inference on pre-loaded dual-GPU models.

    Args:
        model_0: Model on GPU 0 (or None for single-GPU)
        model_1: Model on GPU 1 (or None for single-GPU)
        tokenizer: Shared tokenizer
        prompts: List of prompt strings

    Returns:
        List of response strings (same order as input)
    """
    condition_config = {"do_sample": False}  # Greedy decoding

    if model_1 is None:
        # Single-GPU fallback
        responses = []
        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i:i+BATCH_SIZE]
            q = Queue()
            process_batch_on_gpu(model_0, tokenizer, batch, condition_config,
                                 BATCH_SIZE, 0, q)
            _, batch_responses = q.get()
            if batch_responses:
                responses.extend(batch_responses)
        return responses

    # Split between GPUs
    mid = len(prompts) // 2
    batch_0 = prompts[:mid]
    batch_1 = prompts[mid:]

    results_queue = Queue()

    thread_0 = Thread(
        target=process_batch_on_gpu,
        args=(model_0, tokenizer, batch_0, condition_config, BATCH_SIZE, 0, results_queue)
    )
    thread_1 = Thread(
        target=process_batch_on_gpu,
        args=(model_1, tokenizer, batch_1, condition_config, BATCH_SIZE, 1, results_queue)
    )

    thread_0.start()
    thread_1.start()
    thread_0.join()
    thread_1.join()

    # Collect in order
    results = {}
    while not results_queue.empty():
        gpu_id, responses = results_queue.get()
        results[gpu_id] = responses

    all_responses = []
    if 0 in results and results[0] is not None:
        all_responses.extend(results[0])
    if 1 in results and results[1] is not None:
        all_responses.extend(results[1])

    return all_responses


# ============================================================================
# MAIN
# ============================================================================

def main(model: str = "llama-3.2-3b"):
    logger.info(f"Step 4: Run Perturbation Queries for {model}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    plan_file = MM_DATA_DIR / f"perturbation_plan_{model}.csv"
    samples_file = MM_DATA_DIR / f"selected_samples_{model}.csv"
    cot_file = MM_DATA_DIR / f"cot_responses_{model}.csv"
    checkpoint_file = MM_DATA_DIR / f"perturbation_checkpoint_{model}.json"

    for f, step in [(plan_file, 3), (samples_file, 1), (cot_file, 2)]:
        if not f.exists():
            logger.error(f"{f} not found. Run Step {step} first.")
            sys.exit(1)

    plan_df = pd.read_csv(plan_file)
    samples_df = pd.read_csv(samples_file)
    cot_df = pd.read_csv(cot_file)

    logger.info(f"Perturbation plan: {len(plan_df)} queries across {plan_df['pair_index'].nunique()} pairs")
    logger.info(f"Samples loaded: {len(samples_df)} rows, columns: {list(samples_df.columns[:8])}...")
    logger.info(f"CoT responses loaded: {len(cot_df)} rows")

    # Build lookup: (zpid_1, zpid_2) -> merged row (for prompt building)
    # Merge samples with cot to get new_pred_choice
    merged = samples_df.merge(
        cot_df[['zpid_1', 'zpid_2', 'new_pred_choice']],
        on=['zpid_1', 'zpid_2'],
        how='inner'
    )
    # Index by zpid pair — NOT positional index (which can silently mismatch)
    sample_rows = {}
    for _, row in merged.iterrows():
        key = (int(row['zpid_1']), int(row['zpid_2']))
        sample_rows[key] = row
    logger.info(f"Built sample lookup: {len(sample_rows)} zpid pairs")

    max_step = int(plan_df['step_index'].max())
    total_combos = plan_df.groupby(['pair_index', 'feature']).ngroups
    logger.info(f"Unique (pair, feature) combos: {total_combos}")
    logger.info(f"Max step index: {max_step}")
    logger.info(f"Features in plan: {plan_df['feature'].value_counts().to_dict()}")

    # ------------------------------------------------------------------
    # Check for checkpoint (resume support)
    # ------------------------------------------------------------------
    all_response_rows = []
    flipped_set = set()  # (pair_index, feature) that already flipped
    start_round = 0

    if checkpoint_file.exists():
        logger.info(f"Found checkpoint: {checkpoint_file}")
        with open(checkpoint_file, 'r') as f:
            ckpt = json.load(f)
        start_round = ckpt['completed_rounds']
        flipped_set = set(tuple(x) for x in ckpt['flipped'])

        # Load existing response rows
        resp_ckpt_file = MM_DATA_DIR / f"perturbation_responses_{model}_partial.csv"
        if resp_ckpt_file.exists():
            existing = pd.read_csv(resp_ckpt_file)
            all_response_rows = existing.to_dict('records')
            logger.info(f"Resuming from round {start_round} "
                        f"({len(all_response_rows)} responses, {len(flipped_set)} flips)")

    # ------------------------------------------------------------------
    # Load model(s)
    # ------------------------------------------------------------------
    hf_name = MODEL_HF_NAMES[model]
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    logger.info(f"GPUs available: {gpu_count}")
    if torch.cuda.is_available():
        for i in range(gpu_count):
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f} GB)")

    logger.info(f"Loading model '{hf_name}' on GPU 0...")
    model_0, tokenizer = load_model(hf_name, device=0)
    logger.info(f"GPU 0 model loaded successfully")

    model_1 = None
    if gpu_count >= 2:
        logger.info(f"Loading model '{hf_name}' on GPU 1...")
        model_1, _ = load_model(hf_name, device=1)
        logger.info(f"GPU 1 model loaded successfully")
    else:
        logger.warning(f"Single-GPU mode — no GPU 1 available, inference will be slower")

    logger.info(f"Models loaded. Starting perturbation rounds...")

    # ------------------------------------------------------------------
    # Round-by-round batched inference
    # ------------------------------------------------------------------
    total_queries_run = 0
    start_time = time.time()

    for step_idx in range(start_round, max_step + 1):
        round_start = time.time()

        # Get queries for this step_index
        round_df = plan_df[plan_df['step_index'] == step_idx].copy()

        # Filter out already-flipped combos
        mask = round_df.apply(
            lambda r: (int(r['pair_index']), r['feature']) not in flipped_set,
            axis=1
        )
        round_df = round_df[mask].reset_index(drop=True)

        if len(round_df) == 0:
            logger.debug(f"Round {step_idx}: 0 queries (all flipped or exhausted) — skipping")
            continue

        logger.info(f"Round {step_idx}/{max_step}: {len(round_df)} queries "
                    f"({total_combos - len(flipped_set)} combos remaining)")

        # Build prompts
        prompts = []
        query_meta = []  # Track which query each prompt corresponds to
        for _, query in round_df.iterrows():
            zpid_key = (int(query['zpid_1']), int(query['zpid_2']))
            if zpid_key not in sample_rows:
                logger.warning(f"  zpid pair {zpid_key} not found in sample lookup — skipping")
                continue
            row = sample_rows[zpid_key]
            prompt = build_perturbation_prompt(
                row,
                perturb_property=int(query['loser_property']),
                perturb_feature=query['feature'],
                perturb_value=query['perturbed_value']
            )
            prompts.append(prompt)
            query_meta.append(query)

        if not prompts:
            continue

        # Run inference
        responses = run_inference_round(model_0, model_1, tokenizer, prompts)

        # Parse results and check for flips
        round_flips = 0
        for i, query in enumerate(query_meta):
            if i >= len(responses):
                break

            parsed = parse_response(responses[i])
            new_choice = parsed['choice']
            original_choice = int(query['pred_choice'])

            is_valid = not (isinstance(new_choice, float) and np.isnan(new_choice))
            did_flip = (int(new_choice) != original_choice) if is_valid else False

            if did_flip:
                flipped_set.add((int(query['pair_index']), query['feature']))
                round_flips += 1

            all_response_rows.append({
                'pair_index': int(query['pair_index']),
                'is_correct': bool(query['is_correct']),
                'feature': query['feature'],
                'step_index': int(query['step_index']),
                'increment_label': query['increment_label'],
                'loser_property': int(query['loser_property']),
                'original_value': query['original_value'],
                'perturbed_value': query['perturbed_value'],
                'original_choice': original_choice,
                'new_choice': int(new_choice) if is_valid else None,
                'did_flip': did_flip,
                'reason': parsed['reason'],
                'cot_text': parsed['full_text'],
            })

        total_queries_run += len(prompts)
        round_elapsed = time.time() - round_start
        elapsed_total = time.time() - start_time

        logger.info(f"  Round {step_idx} done: {round_flips} new flips, "
                    f"{len(flipped_set)}/{total_combos} total flipped "
                    f"({100*len(flipped_set)/total_combos:.1f}%), "
                    f"round={round_elapsed:.1f}s, total={elapsed_total:.0f}s")

        # Log parse failures for debugging
        round_parse_failures = sum(
            1 for r in all_response_rows[-len(prompts):]
            if r.get('new_choice') is None
        )
        if round_parse_failures > 0:
            logger.warning(f"  Round {step_idx}: {round_parse_failures}/{len(prompts)} "
                          f"responses failed to parse")

        # Checkpoint after each round
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'completed_rounds': step_idx + 1,
                'flipped': [list(x) for x in flipped_set],
                'total_queries_run': total_queries_run,
            }, f)

        partial_df = pd.DataFrame(all_response_rows)
        partial_df.to_csv(
            MM_DATA_DIR / f"perturbation_responses_{model}_partial.csv",
            index=False
        )

        # Clear GPU cache between rounds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Cleanup models
    # ------------------------------------------------------------------
    unload_model(model_0, tokenizer)
    if model_1 is not None:
        unload_model(model_1, None)

    total_elapsed = time.time() - start_time
    logger.info(f"All rounds complete. Total inference time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    # ------------------------------------------------------------------
    # Build results DataFrames
    # ------------------------------------------------------------------
    responses_df = pd.DataFrame(all_response_rows)
    logger.info(f"Building response DataFrame: {len(responses_df)} rows, "
                f"{responses_df.columns.tolist()}")

    # Save full response log
    resp_file = MM_DATA_DIR / f"perturbation_responses_{model}.csv"
    responses_df.to_csv(resp_file, index=False)
    logger.info(f"Saved responses: {resp_file} ({len(responses_df)} rows)")

    # Build flip summary: one row per (pair, feature)
    flip_rows = []
    for (pi, feat), group in responses_df.groupby(['pair_index', 'feature']):
        group_sorted = group.sort_values('step_index')
        flip_row = group_sorted[group_sorted['did_flip'] == True]

        if len(flip_row) > 0:
            first_flip = flip_row.iloc[0]
            flip_rows.append({
                'pair_index': pi,
                'is_correct': first_flip['is_correct'],
                'feature': feat,
                'flipped': True,
                'flip_step_index': int(first_flip['step_index']),
                'flip_increment': first_flip['increment_label'],
                'original_value': first_flip['original_value'],
                'flipped_value': first_flip['perturbed_value'],
                'n_steps_tried': int(first_flip['step_index']) + 1,
            })
        else:
            last = group_sorted.iloc[-1]
            flip_rows.append({
                'pair_index': pi,
                'is_correct': last['is_correct'],
                'feature': feat,
                'flipped': False,
                'flip_step_index': None,
                'flip_increment': 'no_flip',
                'original_value': last['original_value'],
                'flipped_value': None,
                'n_steps_tried': len(group_sorted),
            })

    flip_df = pd.DataFrame(flip_rows)

    flip_file = MM_DATA_DIR / f"flip_results_{model}.csv"
    flip_df.to_csv(flip_file, index=False)
    logger.info(f"Saved flip results: {flip_file} ({len(flip_df)} rows)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_flipped = flip_df['flipped'].sum()
    n_no_flip = (~flip_df['flipped']).sum()

    logger.info("=" * 60)
    logger.info("PERTURBATION RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total queries executed: {total_queries_run}")
    logger.info(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    logger.info(f"Avg per query: {total_elapsed/max(total_queries_run,1):.2f}s")
    logger.info(f"Flip summary: Flipped={n_flipped}/{len(flip_df)} "
                f"({100*n_flipped/len(flip_df):.1f}%), "
                f"No flip={n_no_flip}/{len(flip_df)} "
                f"({100*n_no_flip/len(flip_df):.1f}%)")

    # Per-feature flip rates
    logger.info("Flip rate by feature:")
    for feat in TARGET_FEATURES:
        feat_df = flip_df[flip_df['feature'] == feat]
        feat_flipped = feat_df['flipped'].sum()
        logger.info(f"  {feat:12s}: {feat_flipped}/{len(feat_df)} "
                    f"({100*feat_flipped/len(feat_df):.1f}%)")

    # Correct vs incorrect flip rates
    logger.info("Flip rate by correctness:")
    for label, val in [("correct", True), ("incorrect", False)]:
        sub = flip_df[flip_df['is_correct'] == val]
        sub_flipped = sub['flipped'].sum()
        logger.info(f"  {label:12s}: {sub_flipped}/{len(sub)} "
                    f"({100*sub_flipped/len(sub):.1f}%)")

    # Average flip distance (step index) for flipped features
    flipped_only = flip_df[flip_df['flipped']]
    if len(flipped_only) > 0:
        logger.info("Average flip step (lower = easier to flip):")
        for feat in TARGET_FEATURES:
            feat_flipped = flipped_only[flipped_only['feature'] == feat]
            if len(feat_flipped) > 0:
                avg_step = feat_flipped['flip_step_index'].mean()
                logger.info(f"  {feat:12s}: {avg_step:.2f}")

    # Clean up checkpoint and partial files
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    partial_file = MM_DATA_DIR / f"perturbation_responses_{model}_partial.csv"
    if partial_file.exists():
        partial_file.unlink()

    logger.info("Cleaned up checkpoint files")
    logger.info(f"Step 4 complete for {model}")

    return flip_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4: Run perturbation queries")
    parser.add_argument('--model', type=str, default='llama-3.2-3b',
                        choices=['llama-3.2-3b', 'qwen3-4b'],
                        help='Model to run perturbations on')
    args = parser.parse_args()
    main(model=args.model)
