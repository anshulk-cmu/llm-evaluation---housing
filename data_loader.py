# -*- coding: utf-8 -*-
"""
Data loading and sampling for LLM Real Estate Valuation Experiment
"""

import pandas as pd
from config import DATA_PATH, SAMPLE_SIZE, RANDOM_STATE

def load_data():
    """
    Load the pairs dataset from CSV.
    
    Returns:
        DataFrame with property pair features
    """
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} pairs from {DATA_PATH}")
    return df


def sample_pairs(df, sample_size=None, random_state=None):
    """
    Sample a subset of property pairs for the experiment.
    
    Args:
        df: Full DataFrame of property pairs
        sample_size: Number of pairs to sample (uses config if None)
        random_state: Random seed (uses config if None)
        
    Returns:
        Sampled DataFrame
    """
    if sample_size is None:
        sample_size = SAMPLE_SIZE
    if random_state is None:
        random_state = RANDOM_STATE
    
    if sample_size >= len(df):
        print(f"Using all {len(df)} pairs (requested {sample_size})")
        return df.reset_index(drop=True)
    
    sampled = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    print(f"Sampled {len(sampled)} pairs")
    return sampled


def get_experiment_data():
    """
    Convenience function: load and sample data in one step.
    
    Returns:
        Sampled DataFrame ready for experiment
    """
    df = load_data()
    return sample_pairs(df)
