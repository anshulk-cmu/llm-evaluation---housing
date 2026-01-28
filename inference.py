# -*- coding: utf-8 -*-
"""
Model inference module - GPU/HuggingFace specific code
This module can be swapped out for different backends (API, local models, etc.)
"""

import os
import gc
import time
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from threading import Thread
from queue import Queue

from config import MAX_NEW_TOKENS, BATCH_SIZE, LOCAL_MODEL_DIR, SYSTEM_MSG

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_name: str, device: int = 0):
    """
    Load model from local directory with Flash Attention 2 for speed.
    Models must be pre-downloaded using download_model.py
    
    Args:
        model_name: HuggingFace model name (e.g., "meta-llama/Llama-3.2-3B-Instruct")
        device: GPU device ID (0, 1, etc.)
        
    Returns:
        tuple: (model, tokenizer)
    """
    model_short_name = model_name.split("/")[-1]
    local_path = os.path.join(LOCAL_MODEL_DIR, model_short_name)
    
    # Check if model exists locally
    if not os.path.exists(local_path) or not os.listdir(local_path):
        raise FileNotFoundError(
            f"Model '{model_short_name}' not found in {LOCAL_MODEL_DIR}.\n"
            f"Please download it first using download_model.py"
        )
    
    print(f"Loading {model_short_name} from {local_path} on GPU {device}...")
    load_path = local_path
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(load_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Important for batch generation
    
    # Dual-GPU setup: load model on specified GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        device_str = f"cuda:{device}"
        print(f"  Using GPU {device}")
        
        # Load model on specific GPU
        model = AutoModelForCausalLM.from_pretrained(
            load_path,
            torch_dtype=torch.float16,
            device_map={"": device}
        )
        print(f"  Loaded on GPU {device}")
    else:
        device_str = "cpu"
        print(f"  No GPU available, using CPU")
        model = AutoModelForCausalLM.from_pretrained(
            load_path,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
    
    model.eval()
    
    if torch.cuda.is_available():
        print(f"  GPU {device} Memory: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")
    else:
        print(f"  Running on CPU")
    
    return model, tokenizer


def unload_model(model, tokenizer):
    """Clean up model from memory."""
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ============================================================================
# INFERENCE
# ============================================================================

def run_batch_inference(model, tokenizer, prompts: list, condition_config: dict, 
                       batch_size: int = None) -> list:
    """
    Run batched inference for speed.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        prompts: List of prompt strings
        condition_config: Dict with generation parameters
        batch_size: Batch size (uses config if None)
        
    Returns:
        List of response strings
    """
    if batch_size is None:
        batch_size = BATCH_SIZE
    
    all_responses = []
    batch_count = 0
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Inference"):
        batch_prompts = prompts[i:i+batch_size]
        
        # Build messages for each prompt
        batch_texts = []
        for prompt in batch_prompts:
            messages = [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            batch_texts.append(text)
        
        # Tokenize batch
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(model.device)
        
        # Generation kwargs
        gen_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": condition_config.get("do_sample", False),
        }
        
        if condition_config.get("do_sample", False):
            gen_kwargs["temperature"] = condition_config.get("temperature", 0.1)
            gen_kwargs["top_p"] = condition_config.get("top_p", 0.1)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        
        # Decode each response
        for j, output in enumerate(outputs):
            new_tokens = output[inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            all_responses.append(response)
        
        # Clear memory every 10 batches
        batch_count += 1
        if batch_count % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return all_responses


def process_batch_on_gpu(model, tokenizer, prompts, condition_config, batch_size, gpu_id, results_queue):
    """
    Process prompts on a specific GPU.
    Thread-safe worker function for dual-GPU inference.
    
    Args:
        model: Loaded model on specific GPU
        tokenizer: Loaded tokenizer
        prompts: List of prompt strings to process
        condition_config: Dict with generation parameters
        batch_size: Number of prompts per batch
        gpu_id: GPU device ID
        results_queue: Thread-safe queue for results
    """
    try:
        responses = []
        
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            # Build messages for each prompt
            batch_texts = []
            for prompt in batch_prompts:
                messages = [
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": prompt}
                ]
                text = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                batch_texts.append(text)
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(f"cuda:{gpu_id}")
            
            # Generation kwargs
            gen_kwargs = {
                "max_new_tokens": MAX_NEW_TOKENS,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "do_sample": condition_config.get("do_sample", False),
            }
            
            if condition_config.get("do_sample", False):
                gen_kwargs["temperature"] = condition_config.get("temperature", 0.1)
                gen_kwargs["top_p"] = condition_config.get("top_p", 0.1)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
            
            # Decode each response
            for j, output in enumerate(outputs):
                new_tokens = output[inputs["input_ids"].shape[1]:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True)
                responses.append(response)
            
            # Clear memory periodically
            if (i // batch_size + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        # Put results in queue
        results_queue.put((gpu_id, responses))
        
    except Exception as e:
        print(f"Error on GPU {gpu_id}: {e}")
        results_queue.put((gpu_id, None))


def run_batch_inference_dual_gpu(model_path: str, prompts: list, condition_config: dict, 
                                 batch_size: int = None) -> list:
    """
    Run batch inference using both GPUs simultaneously.
    Each GPU processes separate batches of 'batch_size' samples in parallel.
    
    This doubles throughput by utilizing both GPUs at once:
    - GPU 0 processes batch of 50 samples
    - GPU 1 processes batch of 50 samples
    - Both run simultaneously → 100 samples processed at once
    
    Args:
        model_path: Path to model in models/ directory
        prompts: List of prompt strings
        condition_config: Dict with generation parameters
        batch_size: Number of prompts per batch per GPU (default from config)
        
    Returns:
        List of response strings
    """
    if batch_size is None:
        batch_size = BATCH_SIZE
    
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("Warning: Dual GPU inference requires 2 GPUs. Falling back to single GPU.")
        # Load single model and use standard inference
        model, tokenizer = load_model(model_path, device=0)
        responses = run_batch_inference(model, tokenizer, prompts, condition_config, batch_size)
        unload_model(model, tokenizer)
        return responses
    
    print(f"  Using dual-GPU batch processing: {batch_size} samples per GPU")
    print(f"  Effective throughput: {2 * batch_size} samples per iteration")
    
    # Load model on GPU 0
    print(f"  Loading model on GPU 0...")
    model_0, tokenizer = load_model(model_path, device=0)
    
    # Load model on GPU 1
    print(f"  Loading model on GPU 1...")
    model_1, _ = load_model(model_path, device=1)
    
    print(f"  Models loaded. Processing {len(prompts)} prompts...")
    
    all_responses = []
    
    # Process prompts in chunks of 2*batch_size
    chunk_size = 2 * batch_size
    total_chunks = (len(prompts) + chunk_size - 1) // chunk_size
    
    for chunk_idx in tqdm(range(0, len(prompts), chunk_size), total=total_chunks, 
                          desc="Dual-GPU inference"):
        chunk_prompts = prompts[chunk_idx:chunk_idx + chunk_size]
        
        # Split chunk into two batches
        mid = len(chunk_prompts) // 2
        batch_0 = chunk_prompts[:mid]
        batch_1 = chunk_prompts[mid:]
        
        # Queue for results
        results_queue = Queue()
        
        # Create threads for each GPU
        thread_0 = Thread(target=process_batch_on_gpu, 
                         args=(model_0, tokenizer, batch_0, condition_config, batch_size, 0, results_queue))
        thread_1 = Thread(target=process_batch_on_gpu, 
                         args=(model_1, tokenizer, batch_1, condition_config, batch_size, 1, results_queue))
        
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
        if 0 in results and results[0] is not None:
            all_responses.extend(results[0])
        if 1 in results and results[1] is not None:
            all_responses.extend(results[1])
        
        # Clear memory periodically
        if (chunk_idx // chunk_size + 1) % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    # Clean up models
    unload_model(model_0, tokenizer)
    unload_model(model_1, None)
    
    return all_responses


def run_inference(model_name: str, prompts: list, condition_config: dict, device: int = 0) -> list:
    """
    High-level inference function.
    Load model, run inference, cleanup.
    
    Args:
        model_name: HuggingFace model name
        prompts: List of prompt strings
        condition_config: Dict with generation parameters
        device: GPU device ID
        
    Returns:
        List of response strings
    """
    model, tokenizer = load_model(model_name, device=device)
    
    start_time = time.time()
    responses = run_batch_inference(model, tokenizer, prompts, condition_config)
    elapsed = time.time() - start_time
    
    unload_model(model, tokenizer)
    
    return responses, elapsed


def run_inference_dual_gpu(model_name: str, prompts: list, condition_config: dict) -> tuple:
    """
    High-level dual-GPU inference function.
    Uses both GPUs simultaneously for faster processing.
    
    Args:
        model_name: HuggingFace model name
        prompts: List of prompt strings
        condition_config: Dict with generation parameters
        
    Returns:
        Tuple of (responses list, elapsed time)
    """
    start_time = time.time()
    responses = run_batch_inference_dual_gpu(model_name, prompts, condition_config)
    elapsed = time.time() - start_time
    
    return responses, elapsed
