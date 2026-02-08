"""
Amnesic Probing with MP (Mean Projection) — 3-Step Causal Framework

Replaces the INLP-based experiment with the Dobrzeniecka et al. (2025) framework:
  Step 1: Erasure + Verification  (does the method erase the feature?)
  Step 2: Information Control     (does targeted erasure hurt more than random?)
  Step 3: Selectivity Control     (does injecting ground truth restore performance?)

Usage:
    python run_mp_experiment.py --model llama-3.2-3b --layer 15
    python run_mp_experiment.py --model qwen3-4b --layer 19
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from mp_erasure import (
    run_erasure_for_feature,
    fit_random_erasure,
    train_probe,
    compute_cosine_similarity,
    ErasureResult,
    CUML_AVAILABLE,
)

# cuML for GPU-accelerated price prediction
try:
    if CUML_AVAILABLE:
        from cuml.linear_model import LogisticRegression as GPU_LogisticRegression
        import cupy as cp
except Exception:
    pass


# =============================================================================
# Configuration
# =============================================================================

class Config:
    DATA_DIR = Path("/home/anshulk/Housing/data")
    ACTIVATIONS_DIR = DATA_DIR / "activations"
    SAMPLES_DIR = DATA_DIR / "amnesic_samples"
    OUTPUT_DIR = DATA_DIR / "mp_results"

    EXPERIMENT = "2_fewshot_cot_temp0"

    FEATURES = {
        'bedrooms': 'bedrooms_p1_more',
        'bathrooms': 'bathrooms_p1_more',
        'year_built': 'year_p1_newer',
        'lot_size': 'lot_p1_larger',
        'sqft': 'sqft_p1_larger',
    }

    RANDOM_SEED = 42
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.1
    N_RANDOM_CONTROLS = 10  # Number of random baselines for statistical confidence


# =============================================================================
# Data Loading (reused from original experiment)
# =============================================================================

def load_activations(model: str, config: Config) -> Dict:
    """Load activation data from .npz file."""
    filepath = config.ACTIVATIONS_DIR / f"{model}_{config.EXPERIMENT}_activations.npz"
    print(f"Loading activations from: {filepath}")
    data = np.load(filepath, allow_pickle=True)

    result = {
        'activations': data['activations'],
        'labels': data['labels'],
        'feature_labels': data['feature_labels'],
        'feature_columns': data['feature_label_columns'],
        'sample_indices': data['sample_indices'],
        'metadata': data['metadata'].item() if data['metadata'].ndim == 0 else data['metadata'],
    }
    print(f"Activations shape: {result['activations'].shape}")
    print(f"Feature columns: {result['feature_columns']}")
    return result


def load_samples(model: str, config: Config) -> pd.DataFrame:
    """Load selected samples with predictions."""
    filepath = config.SAMPLES_DIR / f"selected_samples_{model}_{config.EXPERIMENT}.csv"
    print(f"Loading samples from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} samples")
    return df


def prepare_feature_labels(activation_data: Dict) -> Dict[str, np.ndarray]:
    """Extract binary feature labels from activation data."""
    feature_labels = activation_data['feature_labels']
    feature_columns = activation_data['feature_columns']

    column_map = {
        'bedrooms': 'bedrooms_p1_more',
        'bathrooms': 'bathrooms_p1_more',
        'year_built': 'year_p1_newer',
        'lot_size': 'lot_p1_larger',
        'sqft': 'sqft_p1_larger',
    }

    labels = {}
    for feat_name, col_name in column_map.items():
        col_idx = np.where(feature_columns == col_name)[0]
        if len(col_idx) > 0:
            labels[feat_name] = feature_labels[:, col_idx[0]].astype(int)

    print("\nFeature label class balance:")
    for feat, y in labels.items():
        print(f"  {feat}: {y.mean():.2%} positive")

    return labels


def prepare_price_labels(activation_data: Dict) -> np.ndarray:
    """Extract price labels."""
    if 'labels' in activation_data:
        labels = activation_data['labels'].astype(int)
    else:
        feature_columns = activation_data['feature_columns']
        col_idx = np.where(feature_columns == 'price_p1_higher')[0][0]
        labels = activation_data['feature_labels'][:, col_idx].astype(int)

    print(f"Price labels: {labels.mean():.2%} have property 1 more expensive")
    return labels


# =============================================================================
# Price Prediction
# =============================================================================

def train_price_predictor(X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          use_gpu: bool = False) -> Tuple[float, float]:
    """Train logistic regression for price prediction, return (accuracy, auc)."""
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    if use_gpu and CUML_AVAILABLE:
        try:
            clf = GPU_LogisticRegression(max_iter=5000, C=1.0)
            clf.fit(cp.asarray(X_train), y_train)
            preds = cp.asnumpy(clf.predict(cp.asarray(X_test)))
            probs = cp.asnumpy(clf.predict_proba(cp.asarray(X_test)))[:, 1]
            acc = accuracy_score(y_test, preds)
            auc = roc_auc_score(y_test, probs)
            return acc, auc
        except Exception as e:
            print(f"  GPU failed, falling back to CPU: {e}")

    clf = LogisticRegression(max_iter=5000, C=1.0, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    return accuracy_score(y_test, preds), roc_auc_score(y_test, probs)


def get_train_test_split(n: int, config: Config):
    """Get train/test indices matching the INLP code splits."""
    np.random.seed(config.RANDOM_SEED)
    indices = np.random.permutation(n)
    n_train = int(n * config.TRAIN_RATIO)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    return train_idx, test_idx


# =============================================================================
# Step 1: Erasure + Verification
# =============================================================================

def step1_erasure(activations: np.ndarray,
                  feature_labels: Dict[str, np.ndarray],
                  features_to_run: List[str],
                  methods: List[str],
                  config: Config,
                  use_gpu: bool = False) -> Dict[str, Dict[str, Tuple[ErasureResult, dict]]]:
    """
    Run erasure for each feature × method combination.

    Returns:
        erasure_results[method][feature] = (eraser, info)
    """
    results = {}
    for method in methods:
        results[method] = {}
        for feature in features_to_run:
            if feature not in feature_labels:
                print(f"Warning: {feature} not in labels, skipping")
                continue

            eraser, info = run_erasure_for_feature(
                activations=activations,
                labels=feature_labels[feature],
                feature_name=feature,
                method=method,
                train_ratio=config.TRAIN_RATIO,
                val_ratio=config.VAL_RATIO,
                use_gpu=use_gpu,
                random_state=config.RANDOM_SEED,
            )
            results[method][feature] = (eraser, info)

    return results


# =============================================================================
# Step 2: Information Control
# =============================================================================

def step2_information_control(activations: np.ndarray,
                              price_labels: np.ndarray,
                              erasure_results: Dict[str, Dict[str, Tuple]],
                              config: Config,
                              use_gpu: bool = False) -> pd.DataFrame:
    """
    Test whether targeted erasure hurts price prediction more than random erasure.

    For each feature × method:
      - Apply targeted erasure, measure price prediction drop
      - Apply N random rank-1 erasures, measure mean drop
      - Causal test: targeted drop > random drop

    Also tests baseline (no erasure) and all-features-erased.
    """
    train_idx, test_idx = get_train_test_split(len(activations), config)
    y_train = price_labels[train_idx]
    y_test = price_labels[test_idx]

    X_train_raw = activations[train_idx].astype(np.float32)
    X_test_raw = activations[test_idx].astype(np.float32)

    # Scale for the price predictor (logistic regression). Erasure is applied to
    # raw activations first, then scaling is done independently per condition.
    # This is correct because MP is fit on raw data and assumes raw input;
    # StandardScaler is only for numeric stability of the downstream classifier.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_raw).astype(np.float32)

    rows = []

    # --- Baseline ---
    print("\n--- BASELINE ---")
    baseline_acc, baseline_auc = train_price_predictor(
        X_train_scaled, y_train, X_test_scaled, y_test, use_gpu
    )
    print(f"  Baseline: acc={baseline_acc:.4f}, auc={baseline_auc:.4f}")

    # --- Random controls (N random rank-1 erasures) ---
    print(f"\n--- RANDOM CONTROLS ({config.N_RANDOM_CONTROLS} runs) ---")
    random_drops = []
    for i in range(config.N_RANDOM_CONTROLS):
        random_eraser = fit_random_erasure(activations.shape[1], random_state=config.RANDOM_SEED + i)
        X_train_rand = random_eraser(X_train_raw)
        X_test_rand = random_eraser(X_test_raw)
        # Scale after erasure
        s = StandardScaler()
        X_tr = s.fit_transform(X_train_rand).astype(np.float32)
        X_te = s.transform(X_test_rand).astype(np.float32)
        acc, _ = train_price_predictor(X_tr, y_train, X_te, y_test, use_gpu)
        random_drops.append(baseline_acc - acc)

    random_mean_drop = np.mean(random_drops)
    random_std_drop = np.std(random_drops)
    random_mean_acc = baseline_acc - random_mean_drop
    print(f"  Random control: mean drop={random_mean_drop:.4f} +/- {random_std_drop:.4f}")

    # --- Per method ---
    for method, feat_results in erasure_results.items():
        print(f"\n--- {method.upper()} ---")

        rows.append({
            'method': method,
            'condition': 'baseline',
            'accuracy': baseline_acc,
            'auc': baseline_auc,
            'accuracy_drop': 0.0,
            'rank_removed': 0,
            'random_mean_drop': random_mean_drop,
            'random_std_drop': random_std_drop,
            'causal_test': '',
        })

        # Per-feature erasure
        for feature, (eraser, info) in feat_results.items():
            X_train_erased = eraser(X_train_raw)
            X_test_erased = eraser(X_test_raw)

            # Scale after erasure
            s = StandardScaler()
            X_tr = s.fit_transform(X_train_erased).astype(np.float32)
            X_te = s.transform(X_test_erased).astype(np.float32)

            acc, auc = train_price_predictor(X_tr, y_train, X_te, y_test, use_gpu)
            drop = baseline_acc - acc

            # Causal test: targeted drop > random mean drop
            passes = drop > random_mean_drop + random_std_drop
            causal = 'PASS' if passes else 'FAIL'

            print(f"  {feature}: acc={acc:.4f}, drop={drop:.4f}, random={random_mean_drop:.4f} -> {causal}")

            rows.append({
                'method': method,
                'condition': f'erase_{feature}',
                'accuracy': acc,
                'auc': auc,
                'accuracy_drop': drop,
                'rank_removed': info['rank_removed'],
                'random_mean_drop': random_mean_drop,
                'random_std_drop': random_std_drop,
                'causal_test': causal,
            })

        # All features erased (chain projections)
        # NOTE: With 5 rank-1 projections in 3072-d space the interaction effect
        # is negligible. This combined result is informational only — causal
        # verdicts are per-feature.
        if len(feat_results) > 1:
            print(f"\n  ALL FEATURES:")
            X_train_all = X_train_raw.copy()
            X_test_all = X_test_raw.copy()
            total_rank = 0
            for feature, (eraser, info) in feat_results.items():
                X_train_all = eraser(X_train_all)
                X_test_all = eraser(X_test_all)
                total_rank += info['rank_removed']

            s = StandardScaler()
            X_tr = s.fit_transform(X_train_all).astype(np.float32)
            X_te = s.transform(X_test_all).astype(np.float32)
            acc, auc = train_price_predictor(X_tr, y_train, X_te, y_test, use_gpu)
            drop = baseline_acc - acc
            print(f"  all_features: acc={acc:.4f}, drop={drop:.4f}")

            rows.append({
                'method': method,
                'condition': 'erase_all_features',
                'accuracy': acc,
                'auc': auc,
                'accuracy_drop': drop,
                'rank_removed': total_rank,
                'random_mean_drop': random_mean_drop,
                'random_std_drop': random_std_drop,
                'causal_test': 'PASS' if drop > random_mean_drop + random_std_drop else 'FAIL',
            })

        # Random control row
        rows.append({
            'method': method,
            'condition': 'random_control',
            'accuracy': random_mean_acc,
            'auc': 0.0,
            'accuracy_drop': random_mean_drop,
            'rank_removed': 1,
            'random_mean_drop': random_mean_drop,
            'random_std_drop': random_std_drop,
            'causal_test': 'BASELINE',
        })

    return pd.DataFrame(rows)


# =============================================================================
# Step 3: Selectivity Control (Gold Label Recovery)
# =============================================================================

def step3_selectivity_control(activations: np.ndarray,
                              price_labels: np.ndarray,
                              feature_labels: Dict[str, np.ndarray],
                              erasure_results: Dict[str, Dict[str, Tuple]],
                              config: Config,
                              use_gpu: bool = False) -> pd.DataFrame:
    """
    After erasing feature X, inject ground-truth label back and check
    if price prediction recovers. Recovery = clean causal evidence.

    Concatenates the binary label as an extra column:
        X_recovered = [X_erased (d dims), ground_truth_label (1 dim)]
    """
    train_idx, test_idx = get_train_test_split(len(activations), config)
    y_train = price_labels[train_idx]
    y_test = price_labels[test_idx]

    X_train_raw = activations[train_idx].astype(np.float32)
    X_test_raw = activations[test_idx].astype(np.float32)

    # Baseline with scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_raw).astype(np.float32)

    baseline_acc, _ = train_price_predictor(X_train_scaled, y_train, X_test_scaled, y_test, use_gpu)

    rows = []

    for method, feat_results in erasure_results.items():
        print(f"\n--- SELECTIVITY CONTROL ({method.upper()}) ---")

        for feature, (eraser, info) in feat_results.items():
            if feature not in feature_labels:
                continue

            feat_labels = feature_labels[feature]
            feat_train = feat_labels[train_idx].astype(np.float32).reshape(-1, 1)
            feat_test = feat_labels[test_idx].astype(np.float32).reshape(-1, 1)

            # Erased activations
            X_train_erased = eraser(X_train_raw)
            X_test_erased = eraser(X_test_raw)

            # Price prediction on erased (for reference)
            s1 = StandardScaler()
            X_tr_e = s1.fit_transform(X_train_erased).astype(np.float32)
            X_te_e = s1.transform(X_test_erased).astype(np.float32)
            erased_acc, _ = train_price_predictor(X_tr_e, y_train, X_te_e, y_test, use_gpu)

            # Inject gold label: concatenate ground truth feature label
            X_train_recovered = np.hstack([X_train_erased, feat_train])
            X_test_recovered = np.hstack([X_test_erased, feat_test])

            s2 = StandardScaler()
            X_tr_r = s2.fit_transform(X_train_recovered).astype(np.float32)
            X_te_r = s2.transform(X_test_recovered).astype(np.float32)
            recovered_acc, _ = train_price_predictor(X_tr_r, y_train, X_te_r, y_test, use_gpu)

            denom = baseline_acc - erased_acc
            if abs(denom) < 0.005:
                # Erasure barely changed price prediction — recovery is trivially 100%
                recovery_pct = 100.0
            else:
                recovery_pct = np.clip(
                    (recovered_acc - erased_acc) / denom * 100,
                    0.0, 200.0
                )

            selectivity_pass = recovered_acc >= baseline_acc - 0.02

            print(f"  {feature}: baseline={baseline_acc:.4f}, erased={erased_acc:.4f}, "
                  f"recovered={recovered_acc:.4f} ({recovery_pct:.1f}% recovery) "
                  f"-> {'PASS' if selectivity_pass else 'FAIL'}")

            rows.append({
                'method': method,
                'feature': feature,
                'baseline_accuracy': baseline_acc,
                'erased_accuracy': erased_acc,
                'recovered_accuracy': recovered_acc,
                'recovery_pct': recovery_pct,
                'selectivity_test': 'PASS' if selectivity_pass else 'FAIL',
            })

    return pd.DataFrame(rows)


# =============================================================================
# Correctness Analysis
# =============================================================================

def analyze_by_correctness(activations: np.ndarray,
                           price_labels: np.ndarray,
                           samples_df: pd.DataFrame,
                           erasure_results: Dict[str, Dict[str, Tuple]],
                           config: Config,
                           use_gpu: bool = False) -> pd.DataFrame:
    """Analyze erasure impact on originally correct vs incorrect predictions."""
    rows = []
    correct_mask = samples_df['correct'].values
    incorrect_mask = ~correct_mask

    for group_name, mask in [('correct', correct_mask), ('incorrect', incorrect_mask)]:
        group_act = activations[mask]
        group_prices = price_labels[mask]

        n = len(group_act)
        np.random.seed(config.RANDOM_SEED)
        indices = np.random.permutation(n)
        n_train = int(n * config.TRAIN_RATIO)
        train_idx, test_idx = indices[:n_train], indices[n_train:]

        y_train = group_prices[train_idx]
        y_test = group_prices[test_idx]

        X_train_raw = group_act[train_idx].astype(np.float32)
        X_test_raw = group_act[test_idx].astype(np.float32)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_raw).astype(np.float32)
        X_test_s = scaler.transform(X_test_raw).astype(np.float32)

        baseline_acc, baseline_auc = train_price_predictor(X_train_s, y_train, X_test_s, y_test, use_gpu)

        for method, feat_results in erasure_results.items():
            rows.append({
                'group': group_name,
                'method': method,
                'condition': 'baseline',
                'accuracy': baseline_acc,
                'auc': baseline_auc,
                'accuracy_drop': 0.0,
            })

            for feature, (eraser, info) in feat_results.items():
                X_tr_e = eraser(X_train_raw)
                X_te_e = eraser(X_test_raw)
                s = StandardScaler()
                X_tr = s.fit_transform(X_tr_e).astype(np.float32)
                X_te = s.transform(X_te_e).astype(np.float32)
                acc, auc = train_price_predictor(X_tr, y_train, X_te, y_test, use_gpu)

                rows.append({
                    'group': group_name,
                    'method': method,
                    'condition': f'erase_{feature}',
                    'accuracy': acc,
                    'auc': auc,
                    'accuracy_drop': baseline_acc - acc,
                })

    return pd.DataFrame(rows)


# =============================================================================
# Summary Generation
# =============================================================================

def generate_summary(results_df: pd.DataFrame,
                     erasure_info: Dict[str, Dict[str, dict]],
                     selectivity_df: pd.DataFrame,
                     config: Config) -> Dict:
    """Generate summary with causal verdicts."""
    summary = {'experiment': config.EXPERIMENT, 'methods': {}}

    for method in results_df['method'].unique():
        method_df = results_df[results_df['method'] == method]
        baseline_row = method_df[method_df['condition'] == 'baseline']
        if len(baseline_row) == 0:
            continue

        baseline_acc = baseline_row['accuracy'].values[0]
        random_row = method_df[method_df['condition'] == 'random_control']
        random_drop = random_row['accuracy_drop'].values[0] if len(random_row) > 0 else 0.0

        feature_results = {}
        for _, row in method_df.iterrows():
            cond = row['condition']
            if cond.startswith('erase_') and cond != 'erase_all_features':
                feature = cond.replace('erase_', '')

                # Get erasure info
                einfo = erasure_info.get(method, {}).get(feature, {})

                # Get selectivity info
                sel_rows = selectivity_df[
                    (selectivity_df['method'] == method) & (selectivity_df['feature'] == feature)
                ]
                sel_pass = sel_rows['selectivity_test'].values[0] if len(sel_rows) > 0 else 'N/A'
                recovery = sel_rows['recovery_pct'].values[0] if len(sel_rows) > 0 else 0.0

                # Verdict
                drop = row['accuracy_drop']
                info_control = row['causal_test']

                if info_control == 'FAIL':
                    verdict = 'NOT_CAUSAL'
                    desc = f"Erasure of {feature} does not hurt more than random"
                elif sel_pass == 'FAIL':
                    verdict = 'COLLATERAL_DAMAGE'
                    desc = f"Erasure of {feature} hurts but gold label doesn't recover — collateral damage"
                elif drop < 0.01:
                    verdict = 'NOT_USED'
                    desc = f"{feature} encoded but not used for pricing"
                elif drop < 0.03:
                    verdict = 'MINOR'
                    desc = f"{feature} has minor causal contribution to pricing"
                elif drop < 0.06:
                    verdict = 'SIGNIFICANT'
                    desc = f"{feature} has significant causal contribution to pricing"
                else:
                    verdict = 'CRITICAL'
                    desc = f"{feature} is critical for pricing"

                feature_results[feature] = {
                    'initial_probe_accuracy': einfo.get('initial_probe_accuracy', 'N/A'),
                    'verification_accuracy': einfo.get('verification_accuracy', 'N/A'),
                    'cosine_similarity': einfo.get('cosine_similarity', 'N/A'),
                    'price_drop': drop,
                    'information_control': info_control,
                    'selectivity_test': sel_pass,
                    'recovery_pct': recovery,
                    'verdict': verdict,
                    'description': desc,
                }

        summary['methods'][method] = {
            'baseline_accuracy': baseline_acc,
            'random_control_drop': random_drop,
            'features': feature_results,
        }

    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Amnesic Probing with MP')
    parser.add_argument('--model', type=str, default='llama-3.2-3b',
                        choices=['llama-3.2-3b', 'qwen3-4b'])
    parser.add_argument('--layer', type=int, default=15)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--method', type=str, default='mp',
                        choices=['mp'])
    parser.add_argument('--features', type=str, default='all')
    parser.add_argument('--n-random-controls', type=int, default=10)
    parser.add_argument('--no-gpu', action='store_true')
    args = parser.parse_args()

    # Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    config = Config()
    config.N_RANDOM_CONTROLS = args.n_random_controls
    use_gpu = not args.no_gpu and CUML_AVAILABLE

    methods = [args.method]
    features_to_run = list(config.FEATURES.keys()) if args.features == 'all' \
        else args.features.split(',')

    print("=" * 60)
    print("AMNESIC PROBING WITH MP")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Layer: {args.layer}")
    print(f"Methods: {methods}")
    print(f"Features: {features_to_run}")
    print(f"Random controls: {config.N_RANDOM_CONTROLS}")
    print(f"GPU: {use_gpu}")
    print("=" * 60)

    t0 = time.time()

    # Load data
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    activation_data = load_activations(args.model, config)
    samples_df = load_samples(args.model, config)

    all_activations = activation_data['activations']
    n_layers = all_activations.shape[1]
    layer = min(args.layer, n_layers - 1)
    activations = all_activations[:, layer, :].astype(np.float32)
    print(f"Using layer {layer}, shape: {activations.shape}")

    # Output directory: data/mp_results/{model}/layer_{layer}/
    run_dir = config.OUTPUT_DIR / args.model / f"layer_{layer}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {run_dir}")

    feature_labels = prepare_feature_labels(activation_data)
    price_labels = prepare_price_labels(activation_data)

    # =========================================================================
    # STEP 1: Erasure + Verification
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 1: ERASURE + VERIFICATION")
    print("=" * 60)

    erasure_results = step1_erasure(
        activations, feature_labels, features_to_run, methods, config, use_gpu
    )

    # Save erasure info
    erasure_info = {}
    for method in methods:
        erasure_info[method] = {}
        for feature, (eraser, info) in erasure_results[method].items():
            erasure_info[method][feature] = info
            # Save projection matrix
            np.save(
                run_dir / f"P_{method}_{feature}.npy",
                eraser.projection_matrix
            )

    info_path = run_dir / "erasure_info.json"
    with open(info_path, 'w') as f:
        json.dump(erasure_info, f, indent=2, default=str)
    print(f"\nSaved erasure info to: {info_path}")

    # =========================================================================
    # STEP 2: Information Control
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: INFORMATION CONTROL")
    print("=" * 60)

    results_df = step2_information_control(
        activations, price_labels, erasure_results, config, use_gpu
    )

    results_path = run_dir / "results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved results to: {results_path}")

    # =========================================================================
    # STEP 3: Selectivity Control
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: SELECTIVITY CONTROL")
    print("=" * 60)

    selectivity_df = step3_selectivity_control(
        activations, price_labels, feature_labels, erasure_results, config, use_gpu
    )

    sel_path = run_dir / "selectivity.csv"
    selectivity_df.to_csv(sel_path, index=False)
    print(f"\nSaved selectivity to: {sel_path}")

    # =========================================================================
    # Correctness Analysis
    # =========================================================================
    print("\n" + "=" * 60)
    print("CORRECTNESS ANALYSIS")
    print("=" * 60)

    correctness_df = analyze_by_correctness(
        activations, price_labels, samples_df, erasure_results, config, use_gpu
    )

    corr_path = run_dir / "correctness.csv"
    correctness_df.to_csv(corr_path, index=False)
    print(f"\nSaved correctness analysis to: {corr_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    summary = generate_summary(results_df, erasure_info, selectivity_df, config)

    for method, mdata in summary['methods'].items():
        print(f"\n--- {method.upper()} ---")
        print(f"Baseline accuracy: {mdata['baseline_accuracy']:.4f}")
        print(f"Random control drop: {mdata['random_control_drop']:.4f}")
        print(f"\nFeature verdicts:")
        for feature, fdata in mdata['features'].items():
            print(f"  {feature}:")
            print(f"    Probe: {fdata['initial_probe_accuracy']}")
            print(f"    Cosine sim: {fdata['cosine_similarity']}")
            print(f"    Price drop: {fdata['price_drop']:.4f}")
            print(f"    Info control: {fdata['information_control']}")
            print(f"    Selectivity: {fdata['selectivity_test']} ({fdata['recovery_pct']:.1f}%)")
            print(f"    Verdict: {fdata['verdict']} — {fdata['description']}")

    summary_path = run_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved summary to: {summary_path}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
