# -*- coding: utf-8 -*-
"""
Main entry point for LLM Real Estate Valuation Experiment

Usage:
    python main.py                  # Single GPU mode (sequential)
    python main.py --dual-gpu       # Dual GPU mode (parallel batches)

Edit config.py to change settings (sample size, models, conditions, etc.)
"""

import logging
import os
import sys
from datetime import datetime

from config import MODELS, SAMPLE_SIZE, LOG_DIR
from data_loader import get_experiment_data, load_data
from experiment import run_experiment
from results import (
    compute_zestimate_baseline, 
    print_experiment_summary, 
    save_experiment_summary
)


def setup_logging():
    """Setup logging configuration."""
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"experiment_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def main(use_dual_gpu=False):
    """
    Run the complete experiment.
    
    Args:
        use_dual_gpu: If True, use both GPUs simultaneously for each model.
                     If False, use single GPU per model (sequential).
    """
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("="*70)
    logger.info("LLM REAL ESTATE VALUATION EXPERIMENT")
    logger.info("="*70)
    print("="*70)
    print("LLM REAL ESTATE VALUATION EXPERIMENT")
    print("="*70)
    
    # Load data
    logger.info("\n1. Loading data...")
    print("\n1. Loading data...")
    full_data = load_data()
    sample_df = get_experiment_data()
    
    # Compute Zestimate baseline
    logger.info("\n2. Computing Zestimate baseline...")
    print("\n2. Computing Zestimate baseline...")
    zestimate_acc = compute_zestimate_baseline(full_data)
    logger.info(f"   Zestimate Accuracy: {zestimate_acc:.3f}")
    print(f"   Zestimate Accuracy: {zestimate_acc:.3f}")
    
    # Run experiments for each model
    all_model_results = {}
    
    if use_dual_gpu:
        logger.info(f"\n{'='*70}")
        logger.info(f"3. RUNNING EXPERIMENTS WITH DUAL-GPU MODE")
        logger.info(f"   Each model will use both GPUs simultaneously")
        logger.info(f"   Effective throughput: 100 samples per iteration")
        logger.info(f"{'='*70}")
        print(f"\n{'='*70}")
        print(f"3. RUNNING EXPERIMENTS WITH DUAL-GPU MODE")
        print(f"   Each model will use both GPUs simultaneously")
        print(f"   Effective throughput: 100 samples per iteration")
        print(f"{'='*70}")
    
    for gpu_id, (model_key, model_name) in enumerate(MODELS.items()):
        if not use_dual_gpu:
            logger.info(f"\n{'='*70}")
            logger.info(f"3. RUNNING EXPERIMENTS: {model_name} on GPU {gpu_id}")
            logger.info(f"{'='*70}")
            print(f"\n{'='*70}")
            print(f"3. RUNNING EXPERIMENTS: {model_name} on GPU {gpu_id}")
            print(f"{'='*70}")
        else:
            logger.info(f"\n{'='*70}")
            logger.info(f"3. RUNNING EXPERIMENTS: {model_name}")
            logger.info(f"{'='*70}")
            print(f"\n{'='*70}")
            print(f"3. RUNNING EXPERIMENTS: {model_name}")
            print(f"{'='*70}")
        
        try:
            model_results = run_experiment(model_name, sample_df, device=gpu_id, use_dual_gpu=use_dual_gpu)
            all_model_results[model_key] = model_results
            
            # Print summary for this model
            print_experiment_summary(
                model_results, 
                model_key, 
                SAMPLE_SIZE, 
                zestimate_acc
            )
            
            # Save summary
            summary_rows = []
            for cond_name, result in model_results.items():
                summary_rows.append({
                    "condition": cond_name,
                    "accuracy": result["accuracy"],
                    "n_valid": result["n_valid"],
                    "n_total": result["n_total"],
                    "time_sec": result["time_sec"]
                })
            save_experiment_summary(summary_rows, model_key)
            logger.info(f"Completed all conditions for {model_key}")
            
        except Exception as e:
            logger.error(f"Error running experiments for {model_key}: {e}", exc_info=True)
            print(f"\nError running experiments for {model_key}: {e}")
            print("Checkpoint saved. You can resume by running the script again.")
            raise
    
    logger.info("\n" + "="*70)
    logger.info("ALL EXPERIMENTS COMPLETE!")
    logger.info("="*70)
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    # Check command line arguments
    use_dual_gpu = "--dual-gpu" in sys.argv
    main(use_dual_gpu=use_dual_gpu)
