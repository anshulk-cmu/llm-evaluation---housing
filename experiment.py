# -*- coding: utf-8 -*-
"""
Experiment orchestration - runs all conditions and collects results
"""

import logging
import pandas as pd
import numpy as np
import time
import gc
import torch
from tqdm import tqdm

from config import CONDITIONS, CHECKPOINT_INTERVAL, BATCH_SIZE
from prompts import PROMPT_BUILDERS
from utils import parse_choice
from results import calculate_accuracy, save_results
from inference import load_model, unload_model, run_batch_inference, process_batch_on_gpu
from checkpoint import save_checkpoint, load_checkpoint, delete_checkpoint
from threading import Thread
from queue import Queue

logger = logging.getLogger(__name__)


def run_experiment(model_name: str, sample_df: pd.DataFrame, device: int = 0, use_dual_gpu: bool = False):
    """
    Run all experimental conditions for a given model with checkpoint support.
    
    Args:
        model_name: HuggingFace model name (e.g., "meta-llama/Llama-3.2-3B-Instruct")
        sample_df: DataFrame with property pairs
        device: GPU device ID to use for this model (ignored if use_dual_gpu=True)
        use_dual_gpu: If True, use both GPUs simultaneously for faster processing
        
    Returns:
        dict: condition_name -> {accuracy, n_valid, n_total, time_sec, results_df}
    """
    model_short = model_name.split("/")[-1]
    all_results = {}
    
    # Load models ONCE at the start (not per-batch)
    if use_dual_gpu and torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        logger.info(f"Loading {model_short} on both GPUs (once for all conditions)...")
        print(f"Loading {model_short} on both GPUs (once for all conditions)...")
        model_0, tokenizer = load_model(model_name, device=0)
        model_1, _ = load_model(model_name, device=1)
        models = (model_0, model_1, tokenizer)
    else:
        logger.info(f"Loading {model_short} on GPU {device} (once for all conditions)...")
        print(f"Loading {model_short} on GPU {device} (once for all conditions)...")
        model, tokenizer = load_model(model_name, device=device)
        models = (model, tokenizer)
    
    try:
        for cond_name, cond_config in CONDITIONS.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Running: {model_short} - {cond_name}")
            logger.info(f"{'='*60}")
            print(f"\n{'='*60}")
            print(f"Running: {model_short} - {cond_name}")
            print(f"{'='*60}")
            
            # Check for existing checkpoint
            checkpoint = load_checkpoint(model_name, cond_name)
            if checkpoint and checkpoint.get('completed', False):
                logger.info(f"  Found completed checkpoint, loading results...")
                print(f"  Found completed checkpoint, loading results...")
                all_results[cond_name] = checkpoint
                continue
            
            # Get prompt builder function
            prompt_fn_name = cond_config["prompt_fn"]
            prompt_fn = PROMPT_BUILDERS[prompt_fn_name]
            
            # Initialize or resume from checkpoint
            if checkpoint:
                logger.info(f"  Resuming from checkpoint at index {checkpoint['last_index']}")
                print(f"  Resuming from checkpoint at index {checkpoint['last_index']}")
                results = checkpoint['results']
                start_idx = checkpoint['last_index'] + 1
                start_time_offset = checkpoint.get('elapsed_time', 0)
            else:
                logger.info("  Starting from scratch")
                print("  Starting from scratch")
                results = []
                start_idx = 0
                start_time_offset = 0
            
            # Build all prompts for this condition
            logger.info("Building prompts...")
            print("Building prompts...")
            all_prompts = [prompt_fn(row) for _, row in sample_df.iterrows()]
            
            # Process remaining samples with checkpointing
            start_time = time.time()
            
            for batch_start in range(start_idx, len(sample_df), CHECKPOINT_INTERVAL):
                batch_end = min(batch_start + CHECKPOINT_INTERVAL, len(sample_df))
                batch_prompts = all_prompts[batch_start:batch_end]
                
                logger.info(f"Processing samples {batch_start} to {batch_end-1}...")
                print(f"Processing samples {batch_start} to {batch_end-1}...")
                
                # Run inference on batch (models already loaded!)
                if use_dual_gpu and len(models) == 3:
                    batch_responses = run_dual_gpu_batch(models, batch_prompts, cond_config)
                else:
                    model, tokenizer = models
                    batch_responses = run_batch_inference(model, tokenizer, batch_prompts, cond_config)
                
                # Parse batch results
                for idx, response in enumerate(batch_responses):
                    global_idx = batch_start + idx
                    row = sample_df.iloc[global_idx]
                    results.append({
                        "zpid_1": row["zpid_1"],
                        "zpid_2": row["zpid_2"],
                        "pred_choice": parse_choice(response),
                        "model_text": response,
                        "price.y_1": row["price.y_1"],
                        "price.y_2": row["price.y_2"],
                    })
                
                # Save checkpoint
                elapsed = time.time() - start_time + start_time_offset
                checkpoint_data = {
                    'results': results,
                    'last_index': batch_end - 1,
                    'elapsed_time': elapsed,
                    'completed': batch_end >= len(sample_df)
                }
                save_checkpoint(model_name, cond_name, checkpoint_data)
                print(f"  Checkpoint saved at index {batch_end - 1}")
                
                # Clear memory periodically (but keep models loaded)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            elapsed = time.time() - start_time + start_time_offset
            
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Calculate accuracy
            metrics = calculate_accuracy(results_df)
            
            # Add results_df to metrics
            metrics["results_df"] = results_df
            metrics["time_sec"] = elapsed
            
            all_results[cond_name] = metrics
            
            # Print summary
            logger.info(f"  Accuracy: {metrics['accuracy']:.3f} ({metrics['n_valid']}/{metrics['n_total']} valid)")
            logger.info(f"  Time: {elapsed:.1f}s ({elapsed/len(sample_df):.2f}s per sample)")
            print(f"  Accuracy: {metrics['accuracy']:.3f} "
                  f"({metrics['n_valid']}/{metrics['n_total']} valid)")
            print(f"  Time: {elapsed:.1f}s ({elapsed/len(sample_df):.2f}s per sample)")
            
            # Save final results
            save_results(results_df, model_short, cond_name)
            
            # Mark checkpoint as completed and clean up
            checkpoint_data['completed'] = True
            save_checkpoint(model_name, cond_name, checkpoint_data)
            delete_checkpoint(model_name, cond_name)
            
            logger.info(f"  Completed {cond_name}")
    
    finally:
        # Unload models when done with all conditions
        logger.info(f"Unloading {model_short} from GPU(s)...")
        print(f"Unloading {model_short} from GPU(s)...")
        if use_dual_gpu and len(models) == 3:
            model_0, model_1, tokenizer = models
            unload_model(model_0, tokenizer)
            unload_model(model_1, None)
        else:
            model, tokenizer = models
            unload_model(model, tokenizer)
    
    return all_results


def run_dual_gpu_batch(models: tuple, prompts: list, condition_config: dict) -> list:
    """
    Run inference on both GPUs simultaneously using pre-loaded models.
    
    Args:
        models: Tuple of (model_0, model_1, tokenizer)
        prompts: List of prompt strings
        condition_config: Dict with generation parameters
        
    Returns:
        List of response strings
    """
    model_0, model_1, tokenizer = models
    
    # Split prompts between GPUs
    mid = len(prompts) // 2
    batch_0 = prompts[:mid]
    batch_1 = prompts[mid:]
    
    # Queue for results
    results_queue = Queue()
    
    # Create threads for each GPU
    thread_0 = Thread(target=process_batch_on_gpu, 
                     args=(model_0, tokenizer, batch_0, condition_config, BATCH_SIZE, 0, results_queue))
    thread_1 = Thread(target=process_batch_on_gpu, 
                     args=(model_1, tokenizer, batch_1, condition_config, BATCH_SIZE, 1, results_queue))
    
    # Start both threads
    thread_0.start()
    thread_1.start()
    
    # Wait for both to finish
    thread_0.join()
    thread_1.join()
    
    # Collect results from queue
    results = {}
    while not results_queue.empty():
        item = results_queue.get()
        if len(item) == 2:
            gpu_id, responses = item
            results[gpu_id] = responses
    
    # Merge results in correct order (GPU 0 batch first, then GPU 1 batch)
    all_responses = []
    if 0 in results and results[0] is not None:
        all_responses.extend(results[0])
    if 1 in results and results[1] is not None:
        all_responses.extend(results[1])
    
    return all_responses
