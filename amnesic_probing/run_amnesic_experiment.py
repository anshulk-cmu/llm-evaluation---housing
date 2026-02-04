"""
Amnesic Probing Experiment - Main Runner

This script runs the full amnesic probing experiment to test whether
the model USES the features it encodes for price prediction.

Experiment flow:
1. Load activations and selected samples
2. For each feature, run INLP to create erasure projection matrix
3. Apply projections and measure price prediction accuracy
4. Compare with baseline and random control
5. Generate analysis and plots

Usage:
    python run_amnesic_experiment.py --model llama-3.2-3b --gpu 0
    
For parallel execution on 2 GPUs:
    GPU 0: python run_amnesic_experiment.py --model llama-3.2-3b --gpu 0 --features bedrooms,bathrooms,year_built
    GPU 1: python run_amnesic_experiment.py --model llama-3.2-3b --gpu 1 --features lot_size,sqft
"""

import argparse
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

# Set GPU before importing cuML
def setup_gpu(gpu_id: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

# Import INLP (handles cuML/sklearn fallback internally)
from inlp_cuml import INLPProjector, run_inlp_for_feature, CUML_AVAILABLE

# For price prediction
try:
    if CUML_AVAILABLE:
        from cuml.linear_model import LogisticRegression as GPU_LogisticRegression
        import cupy as cp
except:
    pass
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


# =============================================================================
# Configuration
# =============================================================================

class Config:
    # Paths
    DATA_DIR = Path("/home/anshulk/Housing/data")
    ACTIVATIONS_DIR = DATA_DIR / "activations"
    SAMPLES_DIR = DATA_DIR / "amnesic_samples"
    OUTPUT_DIR = DATA_DIR / "amnesic_results"
    
    # Model settings
    EXPERIMENT = "2_fewshot_cot_temp0"
    
    # INLP settings
    N_ITERATIONS = 300  # Max INLP iterations per feature
    MIN_ACCURACY = 0.52  # Stop when probe accuracy drops below this
    TRAIN_RATIO = 0.7   # 70% training for INLP
    VAL_RATIO = 0.1     # 10% validation for early stopping
    # Remaining 20% is held-out test set
    
    # Features to erase
    FEATURES = {
        'bedrooms': 'bedrooms_p1_more',
        'bathrooms': 'bathrooms_p1_more', 
        'year_built': 'year_built_p1_newer',
        'lot_size': 'lot_p1_larger',
        'sqft': 'sqft_p1_larger'  # If available
    }
    
    # Price prediction settings
    RANDOM_SEED = 42
    N_PRICE_SPLITS = 5  # Cross-validation folds for price prediction


# =============================================================================
# Data Loading
# =============================================================================

def load_activations(model: str, config: Config) -> Dict:
    """Load activation data from .npz file."""
    
    # Map model name to file
    model_map = {
        'llama-3.2-3b': 'llama-3.2-3b',
        'qwen3-4b': 'qwen3-4b'
    }
    
    model_key = model_map.get(model, model)
    filepath = config.ACTIVATIONS_DIR / f"{model_key}_{config.EXPERIMENT}_activations.npz"
    
    print(f"Loading activations from: {filepath}")
    data = np.load(filepath, allow_pickle=True)
    
    print(f"Keys in activation file: {list(data.keys())}")
    
    # Structure: activations shape = (n_samples, n_layers, hidden_dim)
    # Feature labels already included!
    result = {
        'activations': data['activations'],  # (5130, 28, 3072)
        'labels': data['labels'],  # Price prediction labels
        'feature_labels': data['feature_labels'],  # (5130, 6)
        'feature_columns': data['feature_label_columns'],  # Column names
        'sample_indices': data['sample_indices'],
        'metadata': data['metadata'].item() if data['metadata'].ndim == 0 else data['metadata']
    }
    
    print(f"Activations shape: {result['activations'].shape}")
    print(f"Feature columns: {result['feature_columns']}")
    
    return result


def load_samples(model: str, config: Config) -> pd.DataFrame:
    """Load selected samples with predictions and features."""
    
    filepath = config.SAMPLES_DIR / f"selected_samples_{model}_{config.EXPERIMENT}.csv"
    print(f"Loading samples from: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} samples")
    
    return df


def prepare_feature_labels(activation_data: Dict) -> Dict[str, np.ndarray]:
    """Extract feature labels from activation data (already pre-computed)."""
    
    feature_labels = activation_data['feature_labels']  # (n_samples, 6)
    feature_columns = activation_data['feature_columns']
    
    # Map column names to our feature names
    # Columns: ['bathrooms_p1_more', 'bedrooms_p1_more', 'year_p1_newer', 
    #           'lot_p1_larger', 'sqft_p1_larger', 'price_p1_higher']
    column_map = {
        'bedrooms': 'bedrooms_p1_more',
        'bathrooms': 'bathrooms_p1_more',
        'year_built': 'year_p1_newer',
        'lot_size': 'lot_p1_larger',
        'sqft': 'sqft_p1_larger'
    }
    
    labels = {}
    for feat_name, col_name in column_map.items():
        col_idx = np.where(feature_columns == col_name)[0]
        if len(col_idx) > 0:
            labels[feat_name] = feature_labels[:, col_idx[0]].astype(int)
    
    # Print class balance
    print("\nFeature label class balance:")
    for feat, y in labels.items():
        print(f"  {feat}: {y.mean():.2%} positive (p1 has more)")
    
    return labels


def prepare_price_labels(activation_data: Dict) -> np.ndarray:
    """Extract price labels from activation data."""
    
    # Either use pre-computed or from feature_labels
    if 'labels' in activation_data:
        labels = activation_data['labels'].astype(int)
    else:
        feature_columns = activation_data['feature_columns']
        col_idx = np.where(feature_columns == 'price_p1_higher')[0][0]
        labels = activation_data['feature_labels'][:, col_idx].astype(int)
    
    print(f"\nPrice prediction labels: {labels.mean():.2%} have property 1 more expensive")
    
    return labels


# =============================================================================
# INLP Training
# =============================================================================

def run_inlp_all_features(activations: np.ndarray,
                          feature_labels: Dict[str, np.ndarray],
                          features_to_run: List[str],
                          config: Config,
                          use_gpu: bool = True) -> Dict[str, Tuple[np.ndarray, dict]]:
    """
    Run INLP for each specified feature.
    
    Returns dict mapping feature name to (projection_matrix, scaler, info_dict)
    """
    
    results = {}
    
    for feature in features_to_run:
        if feature not in feature_labels:
            print(f"Warning: Feature {feature} not in labels, skipping")
            continue
        
        labels = feature_labels[feature]
        
        print(f"\n{'#'*60}")
        print(f"# Running INLP for: {feature}")
        print(f"{'#'*60}")
        
        P, scaler, info = run_inlp_for_feature(
            activations=activations,
            labels=labels,
            feature_name=feature,
            n_iterations=config.N_ITERATIONS,
            train_ratio=config.TRAIN_RATIO,
            val_ratio=config.VAL_RATIO,
            use_gpu=use_gpu,
            random_state=config.RANDOM_SEED
        )
        
        results[feature] = (P, scaler, info)
        
        # Save projection matrix
        output_path = config.OUTPUT_DIR / f"P_{feature}_{config.EXPERIMENT}.npy"
        np.save(output_path, P)
        print(f"Saved projection to: {output_path}")
        
        # Save scaler parameters for reproducibility
        scaler_path = config.OUTPUT_DIR / f"scaler_{feature}_{config.EXPERIMENT}.npz"
        np.savez(scaler_path, mean=scaler.mean_, scale=scaler.scale_)
        print(f"Saved scaler to: {scaler_path}")
    
    return results


def create_random_projection(input_dim: int, 
                            n_directions: int,
                            random_state: int = 42) -> np.ndarray:
    """
    Create random projection for control experiment.
    Removes same number of dimensions as INLP but in random directions.
    """
    np.random.seed(random_state)
    
    # Random orthonormal directions
    random_matrix = np.random.randn(n_directions, input_dim).astype(np.float32)
    Q, _ = np.linalg.qr(random_matrix.T)
    
    # Projection to nullspace of these random directions
    P = np.eye(input_dim) - Q @ Q.T
    
    return P.astype(np.float32)


# =============================================================================
# Price Prediction
# =============================================================================

def train_price_predictor(X_train: np.ndarray, 
                          y_train: np.ndarray,
                          X_test: np.ndarray,
                          y_test: np.ndarray,
                          use_gpu: bool = True) -> Tuple[float, float]:
    """
    Train logistic regression to predict price and return accuracy + AUC.
    """
    
    if use_gpu and CUML_AVAILABLE:
        try:
            clf = GPU_LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(cp.asarray(X_train), y_train)
            preds = cp.asnumpy(clf.predict(cp.asarray(X_test)))
            probs = cp.asnumpy(clf.predict_proba(cp.asarray(X_test)))[:, 1]
        except Exception as e:
            print(f"GPU prediction failed, falling back to CPU: {e}")
            use_gpu = False
    
    if not use_gpu or not CUML_AVAILABLE:
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    
    return acc, auc


def evaluate_price_prediction(activations: np.ndarray,
                              price_labels: np.ndarray,
                              projection_results: Dict[str, Tuple],
                              config: Config,
                              use_gpu: bool = True) -> pd.DataFrame:
    """
    Evaluate price prediction accuracy under different erasure conditions.
    
    Args:
        activations: Raw activations, shape (n_samples, hidden_dim)
        price_labels: Binary labels for price prediction
        projection_results: Dict mapping feature -> (P, scaler, info)
        config: Configuration object
        use_gpu: Whether to use GPU
    
    Conditions:
    - baseline: Original activations (scaled for fair comparison)
    - For each feature: Activations with that feature erased
    - all_features: All features erased
    - random_control: Random directions erased (same dimensionality)
    
    IMPORTANT: P was learned on scaled data, so we must scale before applying P
    """
    from sklearn.preprocessing import StandardScaler
    
    results = []
    
    # Create train/test split (same split as used in INLP)
    np.random.seed(config.RANDOM_SEED)
    n = len(activations)
    indices = np.random.permutation(n)
    n_train = int(n * config.TRAIN_RATIO)
    n_val = int(n * config.VAL_RATIO)
    
    train_idx = indices[:n_train]
    # Use remaining as test (val + test from INLP split)
    test_idx = indices[n_train:]
    
    y_train = price_labels[train_idx]
    y_test = price_labels[test_idx]
    
    print(f"\nPrice prediction evaluation:")
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
    
    # Fit a scaler on train activations for baseline (for fair comparison)
    baseline_scaler = StandardScaler()
    X_train_scaled = baseline_scaler.fit_transform(activations[train_idx]).astype(np.float32)
    X_test_scaled = baseline_scaler.transform(activations[test_idx]).astype(np.float32)
    
    # 1. Baseline (scaled activations, no projection)
    print("\nCondition: BASELINE")
    acc, auc = train_price_predictor(X_train_scaled, y_train, X_test_scaled, y_test, use_gpu)
    results.append({
        'condition': 'baseline',
        'accuracy': acc,
        'auc': auc,
        'accuracy_drop': 0.0,
        'dimensions_removed': 0
    })
    print(f"  Accuracy: {acc:.4f}, AUC: {auc:.4f}")
    baseline_acc = acc
    
    # 2. Single feature erasures
    for feature, (P, scaler, info) in projection_results.items():
        print(f"\nCondition: ERASE {feature}")
        
        # CRITICAL: Scale with the SAME scaler used during INLP training
        X_train_feat_scaled = scaler.transform(activations[train_idx]).astype(np.float32)
        X_test_feat_scaled = scaler.transform(activations[test_idx]).astype(np.float32)
        
        # Apply projection to scaled data
        X_train_proj = (P @ X_train_feat_scaled.T).T
        X_test_proj = (P @ X_test_feat_scaled.T).T
        
        acc, auc = train_price_predictor(X_train_proj, y_train, X_test_proj, y_test, use_gpu)
        
        results.append({
            'condition': f'erase_{feature}',
            'accuracy': acc,
            'auc': auc,
            'accuracy_drop': baseline_acc - acc,
            'dimensions_removed': info.get('n_iterations_used', 0)
        })
        print(f"  Accuracy: {acc:.4f}, AUC: {auc:.4f}, Drop: {baseline_acc - acc:.4f}")
    
    # 3. All features erased (cumulative)
    # For this, we need to chain the projections - use first feature's scaler
    if len(projection_results) > 1:
        print("\nCondition: ERASE ALL FEATURES")
        
        # Get first scaler and multiply projection matrices
        first_feature = list(projection_results.keys())[0]
        first_scaler = projection_results[first_feature][1]
        
        P_all = np.eye(activations.shape[1], dtype=np.float32)
        for feature, (P, scaler, info) in projection_results.items():
            P_all = P_all @ P
        
        # Scale with first feature's scaler
        X_train_scaled_all = first_scaler.transform(activations[train_idx]).astype(np.float32)
        X_test_scaled_all = first_scaler.transform(activations[test_idx]).astype(np.float32)
        
        X_train_proj = (P_all @ X_train_scaled_all.T).T
        X_test_proj = (P_all @ X_test_scaled_all.T).T
        
        acc, auc = train_price_predictor(X_train_proj, y_train, X_test_proj, y_test, use_gpu)
        
        total_dims = sum(info.get('n_iterations_used', 0) for _, _, info in projection_results.values())
        results.append({
            'condition': 'erase_all_features',
            'accuracy': acc,
            'auc': auc,
            'accuracy_drop': baseline_acc - acc,
            'dimensions_removed': total_dims
        })
        print(f"  Accuracy: {acc:.4f}, AUC: {auc:.4f}, Drop: {baseline_acc - acc:.4f}")
    
    # 4. Random control (on scaled data for fair comparison)
    print("\nCondition: RANDOM CONTROL")
    
    # Use average dimensions removed by feature erasures
    avg_dims = int(np.mean([r['dimensions_removed'] for r in results if r['dimensions_removed'] > 0]))
    if avg_dims < 10:
        avg_dims = 50  # Default if not enough info
    
    P_random = create_random_projection(activations.shape[1], avg_dims, config.RANDOM_SEED)
    
    # Apply to baseline-scaled data
    X_train_proj = (P_random @ X_train_scaled.T).T
    X_test_proj = (P_random @ X_test_scaled.T).T
    
    acc, auc = train_price_predictor(X_train_proj, y_train, X_test_proj, y_test, use_gpu)
    
    results.append({
        'condition': 'random_control',
        'accuracy': acc,
        'auc': auc,
        'accuracy_drop': baseline_acc - acc,
        'dimensions_removed': avg_dims
    })
    print(f"  Accuracy: {acc:.4f}, AUC: {auc:.4f}, Drop: {baseline_acc - acc:.4f}")
    
    return pd.DataFrame(results)


# =============================================================================
# Analysis
# =============================================================================

def analyze_by_correctness(activations: np.ndarray,
                           price_labels: np.ndarray,
                           samples_df: pd.DataFrame,
                           projection_results: Dict[str, Tuple],
                           config: Config,
                           use_gpu: bool = True) -> pd.DataFrame:
    """
    Analyze how erasure affects originally correct vs incorrect predictions.
    
    Args:
        projection_results: Dict mapping feature -> (P, scaler, info)
    """
    from sklearn.preprocessing import StandardScaler
    
    results = []
    
    correct_mask = samples_df['correct'].values
    incorrect_mask = ~correct_mask
    
    for group_name, mask in [('correct', correct_mask), ('incorrect', incorrect_mask)]:
        group_activations = activations[mask]
        group_prices = price_labels[mask]
        
        print(f"\n{'='*60}")
        print(f"Analyzing: {group_name.upper()} predictions ({mask.sum()} samples)")
        print(f"{'='*60}")
        
        # Train/test split within group
        n = len(group_activations)
        indices = np.random.permutation(n)
        n_train = int(n * config.TRAIN_RATIO)
        train_idx, test_idx = indices[:n_train], indices[n_train:]
        
        y_train = group_prices[train_idx]
        y_test = group_prices[test_idx]
        
        # Baseline with scaling (for fair comparison)
        baseline_scaler = StandardScaler()
        X_train = baseline_scaler.fit_transform(group_activations[train_idx]).astype(np.float32)
        X_test = baseline_scaler.transform(group_activations[test_idx]).astype(np.float32)
        acc, auc = train_price_predictor(X_train, y_train, X_test, y_test, use_gpu)
        baseline_acc = acc
        
        results.append({
            'group': group_name,
            'condition': 'baseline',
            'accuracy': acc,
            'auc': auc,
            'accuracy_drop': 0.0
        })
        print(f"  Baseline: {acc:.4f}")
        
        # Each feature erasure (using feature's scaler)
        for feature, (P, scaler, info) in projection_results.items():
            X_train_scaled = scaler.transform(group_activations[train_idx]).astype(np.float32)
            X_test_scaled = scaler.transform(group_activations[test_idx]).astype(np.float32)
            
            X_train_proj = (P @ X_train_scaled.T).T
            X_test_proj = (P @ X_test_scaled.T).T
            
            acc, auc = train_price_predictor(X_train_proj, y_train, X_test_proj, y_test, use_gpu)
            
            results.append({
                'group': group_name,
                'condition': f'erase_{feature}',
                'accuracy': acc,
                'auc': auc,
                'accuracy_drop': baseline_acc - acc
            })
            print(f"  After erasing {feature}: {acc:.4f} (drop: {baseline_acc - acc:.4f})")
    
    return pd.DataFrame(results)


def generate_summary(results_df: pd.DataFrame,
                     inlp_info: Dict[str, dict],
                     config: Config) -> Dict:
    """Generate summary statistics and interpretations."""
    
    summary = {
        'experiment': config.EXPERIMENT,
        'total_samples': len(results_df),
        'baseline_accuracy': results_df[results_df['condition'] == 'baseline']['accuracy'].values[0],
        'feature_importance': {},
        'causal_evidence': {}
    }
    
    baseline = summary['baseline_accuracy']
    
    # Feature importance (ranked by accuracy drop)
    feature_drops = []
    for _, row in results_df.iterrows():
        if row['condition'].startswith('erase_') and row['condition'] != 'erase_all_features':
            feature = row['condition'].replace('erase_', '')
            feature_drops.append({
                'feature': feature,
                'accuracy_drop': row['accuracy_drop'],
                'post_erasure_accuracy': row['accuracy']
            })
    
    feature_drops.sort(key=lambda x: x['accuracy_drop'], reverse=True)
    summary['feature_importance'] = feature_drops
    
    # Causal evidence interpretation
    for item in feature_drops:
        feature = item['feature']
        drop = item['accuracy_drop']
        
        if drop < 0.02:
            verdict = "NOT_USED"
            description = f"Model encodes {feature} but does NOT use it for pricing"
        elif drop < 0.05:
            verdict = "MINOR_CONTRIBUTOR"
            description = f"Model uses {feature} slightly for pricing"
        elif drop < 0.10:
            verdict = "SIGNIFICANT_CONTRIBUTOR"  
            description = f"Model uses {feature} significantly for pricing"
        else:
            verdict = "CRITICAL"
            description = f"Model heavily relies on {feature} for pricing"
        
        # Get probe accuracy for this feature
        probe_acc = inlp_info.get(feature, {}).get('initial_accuracy', 'N/A')
        
        summary['causal_evidence'][feature] = {
            'probe_accuracy': probe_acc,
            'accuracy_drop': drop,
            'verdict': verdict,
            'description': description
        }
    
    # Random control comparison
    random_row = results_df[results_df['condition'] == 'random_control']
    if len(random_row) > 0:
        random_drop = random_row['accuracy_drop'].values[0]
        summary['random_control_drop'] = random_drop
        
        # Check if feature drops are significantly larger than random
        avg_feature_drop = np.mean([f['accuracy_drop'] for f in feature_drops])
        if avg_feature_drop > random_drop + 0.02:
            summary['control_validation'] = "PASSED: Feature erasure drops > random erasure"
        else:
            summary['control_validation'] = "WARNING: Feature drops similar to random - may be dimensionality effect"
    
    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run Amnesic Probing Experiment')
    parser.add_argument('--model', type=str, default='llama-3.2-3b',
                       choices=['llama-3.2-3b', 'qwen3-4b'])
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--features', type=str, default='all',
                       help='Comma-separated features to run, or "all"')
    parser.add_argument('--layer', type=int, default=15,
                       help='Layer to use for activations')
    parser.add_argument('--skip-inlp', action='store_true',
                       help='Skip INLP training, load existing projections')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Force CPU mode')
    args = parser.parse_args()
    
    # Setup
    setup_gpu(args.gpu)
    config = Config()
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    
    use_gpu = not args.no_gpu and CUML_AVAILABLE
    print(f"Using GPU: {use_gpu}")
    
    # Determine features to run
    if args.features == 'all':
        features_to_run = list(config.FEATURES.keys())
    else:
        features_to_run = args.features.split(',')
    print(f"Features to process: {features_to_run}")
    
    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    activation_data = load_activations(args.model, config)
    samples_df = load_samples(args.model, config)
    
    # Get activations for specified layer
    # Shape: (n_samples, n_layers, hidden_dim) -> extract layer
    all_activations = activation_data['activations']
    n_layers = all_activations.shape[1]
    
    if args.layer >= n_layers:
        print(f"Warning: Layer {args.layer} >= {n_layers}, using layer {n_layers-1}")
        args.layer = n_layers - 1
    
    activations = all_activations[:, args.layer, :].astype(np.float32)
    print(f"Using layer {args.layer}, activations shape: {activations.shape}")
    
    # Prepare labels (from activation data, already aligned)
    feature_labels = prepare_feature_labels(activation_data)
    price_labels = prepare_price_labels(activation_data)
    
    # Run INLP for each feature
    print("\n" + "="*60)
    print("INLP TRAINING")
    print("="*60)
    
    if args.skip_inlp:
        print("Loading existing projection matrices and scalers...")
        projection_results = {}
        inlp_info = {}
        for feature in features_to_run:
            p_path = config.OUTPUT_DIR / f"P_{feature}_{config.EXPERIMENT}.npy"
            scaler_path = config.OUTPUT_DIR / f"scaler_{feature}_{config.EXPERIMENT}.npz"
            if p_path.exists() and scaler_path.exists():
                P = np.load(p_path)
                scaler_data = np.load(scaler_path)
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaler.mean_ = scaler_data['mean']
                scaler.scale_ = scaler_data['scale']
                scaler.n_features_in_ = len(scaler.mean_)
                projection_results[feature] = (P, scaler, {})
                print(f"Loaded: {p_path}")
            else:
                print(f"Warning: {p_path} or {scaler_path} not found, skipping {feature}")
    else:
        inlp_results = run_inlp_all_features(
            activations=activations,
            feature_labels=feature_labels,
            features_to_run=features_to_run,
            config=config,
            use_gpu=use_gpu
        )
        
        # inlp_results: {feature: (P, scaler, info)}
        projection_results = inlp_results
        inlp_info = {f: r[2] for f, r in inlp_results.items()}
        
        # Save INLP info
        info_path = config.OUTPUT_DIR / f"inlp_info_{args.model}_{config.EXPERIMENT}.json"
        with open(info_path, 'w') as f:
            # Convert numpy types for JSON
            info_serializable = {}
            for feat, info in inlp_info.items():
                info_serializable[feat] = {
                    k: (float(v) if isinstance(v, (np.floating, float)) else 
                        [float(x) for x in v] if isinstance(v, list) else v)
                    for k, v in info.items()
                }
            json.dump(info_serializable, f, indent=2)
        print(f"\nSaved INLP info to: {info_path}")
    
    # Evaluate price prediction
    print("\n" + "="*60)
    print("PRICE PREDICTION EVALUATION")
    print("="*60)
    
    results_df = evaluate_price_prediction(
        activations=activations,
        price_labels=price_labels,
        projection_results=projection_results,
        config=config,
        use_gpu=use_gpu
    )
    
    # Save results
    results_path = config.OUTPUT_DIR / f"amnesic_results_{args.model}_{config.EXPERIMENT}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved results to: {results_path}")
    
    # Analyze by correctness
    print("\n" + "="*60)
    print("ANALYSIS BY PREDICTION CORRECTNESS")
    print("="*60)
    
    correctness_df = analyze_by_correctness(
        activations=activations,
        price_labels=price_labels,
        samples_df=samples_df,
        projection_results=projection_results,
        config=config,
        use_gpu=use_gpu
    )
    
    correctness_path = config.OUTPUT_DIR / f"correctness_analysis_{args.model}_{config.EXPERIMENT}.csv"
    correctness_df.to_csv(correctness_path, index=False)
    print(f"\nSaved correctness analysis to: {correctness_path}")
    
    # Generate summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    summary = generate_summary(results_df, inlp_info if not args.skip_inlp else {}, config)
    
    print(f"\nBaseline Accuracy: {summary['baseline_accuracy']:.4f}")
    print(f"\nFeature Importance (by accuracy drop):")
    for i, item in enumerate(summary['feature_importance']):
        print(f"  {i+1}. {item['feature']}: {item['accuracy_drop']:.4f} drop")
    
    print(f"\nCausal Evidence:")
    for feature, evidence in summary['causal_evidence'].items():
        print(f"  {feature}: {evidence['verdict']}")
        print(f"    - Probe acc: {evidence['probe_accuracy']}")
        print(f"    - Drop: {evidence['accuracy_drop']:.4f}")
        print(f"    - {evidence['description']}")
    
    if 'control_validation' in summary:
        print(f"\nControl: {summary['control_validation']}")
    
    # Save summary
    summary_path = config.OUTPUT_DIR / f"amnesic_summary_{args.model}_{config.EXPERIMENT}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved summary to: {summary_path}")
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
