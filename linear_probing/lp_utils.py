# -*- coding: utf-8 -*-
"""
Utility functions for Linear Probing Pipeline
"""
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime


def load_experiment_results(results_dir: Path, model_name: str) -> Dict[str, Dict]:
    """
    Load experiment results for a model to find the winning prompt strategy.
    
    Args:
        results_dir: Path to results directory
        model_name: Name of the model (e.g., "llama-3.2-3b")
        
    Returns:
        Dictionary mapping condition names to their results
    """
    results = {}
    
    for result_file in results_dir.glob(f"*{model_name}*.pkl"):
        with open(result_file, "rb") as f:
            data = pickle.load(f)
        condition = result_file.stem
        results[condition] = data
        
    return results


def find_best_prompt_strategy(results_dir: Path, model_name: str) -> Tuple[str, float]:
    """
    Find the prompt strategy with highest accuracy.
    
    Args:
        results_dir: Path to results directory
        model_name: Name of the model
        
    Returns:
        Tuple of (best_condition_name, accuracy)
    """
    results = load_experiment_results(results_dir, model_name)
    
    if not results:
        raise ValueError(f"No results found for model {model_name} in {results_dir}")
    
    best_condition = None
    best_accuracy = 0.0
    
    for condition, data in results.items():
        if "accuracy" in data:
            acc = data["accuracy"]
        elif "predictions" in data and "labels" in data:
            preds = np.array(data["predictions"])
            labels = np.array(data["labels"])
            acc = (preds == labels).mean()
        else:
            continue
            
        if acc > best_accuracy:
            best_accuracy = acc
            best_condition = condition
            
    return best_condition, best_accuracy


def create_feature_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary feature labels for probing.
    
    These labels represent intermediate features the model might use
    to make its decision.
    
    Args:
        df: DataFrame with property pair data
        
    Returns:
        DataFrame with binary feature labels
    """
    labels = pd.DataFrame(index=df.index)
    
    # Parse numeric values safely
    def safe_float(val):
        try:
            if pd.isna(val):
                return np.nan
            return float(str(val).replace(",", "").strip())
        except:
            return np.nan
    
    # Bathrooms comparison
    bath_1 = df["bathrooms_1"].apply(safe_float)
    bath_2 = df["bathrooms_2"].apply(safe_float)
    labels["bathrooms_p1_more"] = (bath_1 > bath_2).astype(int)
    
    # Bedrooms comparison
    bed_1 = df["bedrooms_1"].apply(safe_float)
    bed_2 = df["bedrooms_2"].apply(safe_float)
    labels["bedrooms_p1_more"] = (bed_1 > bed_2).astype(int)
    
    # Year built comparison (newer = higher year)
    year_1 = df["yearBuilt_1"].apply(safe_float)
    year_2 = df["yearBuilt_2"].apply(safe_float)
    labels["year_p1_newer"] = (year_1 > year_2).astype(int)
    
    # Lot size comparison (need to handle units)
    def parse_lot(row, suffix):
        try:
            lot_val = safe_float(row[f"lot{suffix}"])
            lot_unit = str(row.get(f"lotUnit{suffix}", "sqft")).lower()
            if "acre" in lot_unit:
                lot_val = lot_val * 43560  # Convert to sqft
            return lot_val
        except:
            return np.nan

    lot_1 = df.apply(lambda r: parse_lot(r, "_1"), axis=1)
    lot_2 = df.apply(lambda r: parse_lot(r, "_2"), axis=1)
    labels["lot_p1_larger"] = (lot_1 > lot_2).astype(int)

    # Square footage comparison (lot size is used as proxy for sqft)
    # Note: In this dataset, "lot" represents the land area, which we use as sqft comparison
    labels["sqft_p1_larger"] = labels["lot_p1_larger"].copy()  # Using lot as proxy for sqft

    # Ground truth: which property has higher price
    # This is what we're ultimately trying to predict
    # Calculate from price columns if label doesn't exist
    if "label" in df.columns:
        labels["price_p1_higher"] = df["label"].astype(int)
    else:
        price_1 = df["price.y_1"].apply(safe_float)
        price_2 = df["price.y_2"].apply(safe_float)
        labels["price_p1_higher"] = (price_1 > price_2).astype(int)
    
    return labels


def save_checkpoint(data: Dict, checkpoint_path: Path, prefix: str = "extraction"):
    """Save a checkpoint during processing."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.pkl"
    filepath = checkpoint_path / filename
    
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    
    print(f"Checkpoint saved: {filepath}")
    return filepath


def load_latest_checkpoint(checkpoint_path: Path, prefix: str = "extraction") -> Optional[Dict]:
    """Load the most recent checkpoint."""
    checkpoints = sorted(checkpoint_path.glob(f"{prefix}_*.pkl"))
    
    if not checkpoints:
        return None
        
    latest = checkpoints[-1]
    print(f"Loading checkpoint: {latest}")
    
    with open(latest, "rb") as f:
        return pickle.load(f)


def get_activation_path(activations_dir: Path, model_name: str, condition: str) -> Path:
    """Get the path for storing activations."""
    return activations_dir / f"{model_name}_{condition}_activations.npz"


def save_activations(
    activations: np.ndarray,
    sample_indices: np.ndarray,
    labels: np.ndarray,
    feature_labels: pd.DataFrame,
    path: Path,
    metadata: Optional[Dict] = None
):
    """
    Save extracted activations to disk.
    
    Args:
        activations: Array of shape (n_samples, n_layers, d_model)
        sample_indices: Original indices in the dataset
        labels: Ground truth labels
        feature_labels: DataFrame with intermediate feature labels
        path: Output path
        metadata: Optional metadata dict
    """
    np.savez_compressed(
        path,
        activations=activations,
        sample_indices=sample_indices,
        labels=labels,
        feature_labels=feature_labels.values,
        feature_label_columns=feature_labels.columns.tolist(),
        metadata=metadata or {}
    )
    
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Saved activations to {path} ({size_mb:.1f} MB)")


def load_activations(path: Path) -> Dict[str, Any]:
    """
    Load activations from disk.
    
    Returns:
        Dictionary with keys: activations, sample_indices, labels, 
        feature_labels (as DataFrame), metadata
    """
    data = np.load(path, allow_pickle=True)
    
    feature_labels = pd.DataFrame(
        data["feature_labels"],
        columns=data["feature_label_columns"].tolist()
    )
    
    return {
        "activations": data["activations"],
        "sample_indices": data["sample_indices"],
        "labels": data["labels"],
        "feature_labels": feature_labels,
        "metadata": data["metadata"].item() if data["metadata"].ndim == 0 else data["metadata"]
    }


def print_banner(text: str, char: str = "=", width: int = 60):
    """Print a formatted banner."""
    print(char * width)
    print(f" {text}")
    print(char * width)
