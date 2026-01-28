# -*- coding: utf-8 -*-
"""
Checkpoint management for experiment resumption
"""

import os
import json
import pickle
from config import CHECKPOINT_DIR


def ensure_checkpoint_dir():
    """Create checkpoint directory if it doesn't exist."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def get_checkpoint_path(model_name: str, condition_name: str):
    """Get checkpoint file path for a specific model-condition."""
    ensure_checkpoint_dir()
    model_short = model_name.split("/")[-1]
    filename = f"checkpoint_{model_short}_{condition_name}.pkl"
    return os.path.join(CHECKPOINT_DIR, filename)


def save_checkpoint(model_name: str, condition_name: str, checkpoint_data: dict):
    """
    Save checkpoint data.
    
    Args:
        model_name: Model name
        condition_name: Condition name
        checkpoint_data: Dict with 'results', 'last_index', 'completed' keys
    """
    filepath = get_checkpoint_path(model_name, condition_name)
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"  Checkpoint saved at index {checkpoint_data['last_index']}")


def load_checkpoint(model_name: str, condition_name: str):
    """
    Load checkpoint data if it exists.
    
    Args:
        model_name: Model name
        condition_name: Condition name
        
    Returns:
        dict or None: Checkpoint data if exists, None otherwise
    """
    filepath = get_checkpoint_path(model_name, condition_name)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None


def delete_checkpoint(model_name: str, condition_name: str):
    """Delete checkpoint file after successful completion."""
    filepath = get_checkpoint_path(model_name, condition_name)
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"  Checkpoint cleaned up")
