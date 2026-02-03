# -*- coding: utf-8 -*-
"""
Train Logistic Regression Price Predictor on LLM Activations

Trains logistic regression on activations from EACH layer to predict
which property costs more. Tests if a simple classifier can outperform
the LLM's own output (~60%) using the model's internal representations.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import argparse
import json
import logging
from tqdm import tqdm
import warnings

# Sklearn imports
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy import stats

# Try to import cuML for GPU acceleration
try:
    import cuml
    from cuml.linear_model import LogisticRegression as cuLogisticRegression
    import cupy as cp
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

# Import config
import importlib.util
_THIS_DIR = Path(__file__).parent.resolve()
config_spec = importlib.util.spec_from_file_location("logit_config", str(_THIS_DIR / "logit_config.py"))
config = importlib.util.module_from_spec(config_spec)
config_spec.loader.exec_module(config)

# Import lp_utils for load_activations
lp_utils_spec = importlib.util.spec_from_file_location("lp_utils", str(_THIS_DIR.parent / "linear_probing" / "lp_utils.py"))
lp_utils = importlib.util.module_from_spec(lp_utils_spec)
lp_utils_spec.loader.exec_module(lp_utils)
load_activations = lp_utils.load_activations


@dataclass
class LayerResult:
    """Result for a single layer's probe."""
    layer: int
    test_accuracy: float
    test_auc: float
    val_accuracy: float
    train_accuracy: float
    best_C: float
    ci_lower: float
    ci_upper: float
    p_value: float
    is_significant: bool
    n_train: int
    n_val: int
    n_test: int

    def to_dict(self) -> Dict:
        return self.__dict__


def setup_logging(output_dir: Path, model_name: str, condition: str) -> logging.Logger:
    """Setup logging to both file and console."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"logit_{model_name}_{condition}_{timestamp}.log"

    logger = logging.getLogger(f"logit_{model_name}_{condition}")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def compute_wilson_ci(n_correct: int, n_total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute Wilson score confidence interval."""
    if n_total == 0:
        return 0.0, 1.0
    p_hat = n_correct / n_total
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denom
    offset = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_total)) / n_total) / denom
    return max(0, center - offset), min(1, center + offset)


def create_stratified_splits(n_samples: int, labels: np.ndarray, random_state: int = 42) -> Dict[str, np.ndarray]:
    """Create stratified 70/10/20 train/val/test splits."""
    indices = np.arange(n_samples)

    # 80% train+val, 20% test
    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.20, stratify=labels, random_state=random_state
    )

    # Split train+val into 70% train, 10% val
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.125, stratify=labels[train_val_idx], random_state=random_state
    )

    return {'train': train_idx, 'val': val_idx, 'test': test_idx}


def train_layer_probe(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    layer: int, use_gpu: bool = True
) -> LayerResult:
    """Train and evaluate probe for a single layer."""

    Cs = config.LOGIT_CS
    max_iter = config.LOGIT_MAX_ITER
    use_cuml = use_gpu and CUML_AVAILABLE

    # Standardize
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    best_C, best_val_score = None, 0

    if use_cuml:
        X_train_gpu = cp.asarray(X_train_s.astype(np.float32))
        y_train_gpu = cp.asarray(y_train.astype(np.float32))
        X_val_gpu = cp.asarray(X_val_s.astype(np.float32))
        X_test_gpu = cp.asarray(X_test_s.astype(np.float32))

        for C in Cs:
            probe = cuLogisticRegression(C=C, max_iter=max_iter, tol=1e-4, solver="qn")
            probe.fit(X_train_gpu, y_train_gpu)
            val_acc = accuracy_score(y_val, cp.asnumpy(probe.predict(X_val_gpu)))
            if val_acc > best_val_score:
                best_val_score, best_C = val_acc, C

        final = cuLogisticRegression(C=best_C, max_iter=max_iter*2, tol=1e-5, solver="qn")
        final.fit(X_train_gpu, y_train_gpu)

        y_train_pred = cp.asnumpy(final.predict(X_train_gpu))
        y_val_pred = cp.asnumpy(final.predict(X_val_gpu))
        y_test_pred = cp.asnumpy(final.predict(X_test_gpu))
        y_test_prob = cp.asnumpy(final.predict_proba(X_test_gpu))[:, 1]
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            for C in Cs:
                probe = LogisticRegression(C=C, max_iter=max_iter, random_state=42,
                                          class_weight="balanced", solver="saga", n_jobs=-1)
                probe.fit(X_train_s, y_train)
                val_acc = accuracy_score(y_val, probe.predict(X_val_s))
                if val_acc > best_val_score:
                    best_val_score, best_C = val_acc, C

            final = LogisticRegression(C=best_C, max_iter=max_iter*2, random_state=42,
                                       class_weight="balanced", solver="saga", n_jobs=-1)
            final.fit(X_train_s, y_train)

        y_train_pred = final.predict(X_train_s)
        y_val_pred = final.predict(X_val_s)
        y_test_pred = final.predict(X_test_s)
        y_test_prob = final.predict_proba(X_test_s)[:, 1]

    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_prob) if len(np.unique(y_test)) > 1 else 0.5

    n_correct = (y_test_pred == y_test).sum()
    ci_lower, ci_upper = compute_wilson_ci(n_correct, len(y_test))
    p_value = stats.binomtest(n_correct, len(y_test), 0.5, alternative='greater').pvalue

    return LayerResult(
        layer=layer, test_accuracy=test_acc, test_auc=test_auc,
        val_accuracy=val_acc, train_accuracy=train_acc, best_C=best_C,
        ci_lower=ci_lower, ci_upper=ci_upper, p_value=p_value,
        is_significant=p_value < 0.05,
        n_train=len(y_train), n_val=len(y_val), n_test=len(y_test)
    )


def run_all_layers(activations: np.ndarray, labels: np.ndarray,
                   splits: Dict, use_gpu: bool, logger: logging.Logger) -> List[LayerResult]:
    """Run probing on all layers."""
    results = []
    n_layers = activations.shape[1]
    y = labels.astype(int)

    for layer in tqdm(range(n_layers), desc="Probing layers"):
        X = activations[:, layer, :]
        result = train_layer_probe(
            X[splits['train']], y[splits['train']],
            X[splits['val']], y[splits['val']],
            X[splits['test']], y[splits['test']],
            layer=layer, use_gpu=use_gpu
        )
        results.append(result)
        logger.info(f"Layer {layer:2d}: acc={result.test_accuracy:.4f}, auc={result.test_auc:.4f}, C={result.best_C}")

    return results


def save_results(results: List[LayerResult], output_dir: Path, model: str, condition: str):
    """Save results to CSV and JSON."""
    df = pd.DataFrame([r.to_dict() for r in results])
    df.to_csv(output_dir / f"logit_results_{model}_{condition}.csv", index=False)

    best = df.loc[df['test_accuracy'].idxmax()]
    summary = {
        "model": model, "condition": condition,
        "best_layer": int(best['layer']),
        "best_accuracy": float(best['test_accuracy']),
        "best_auc": float(best['test_auc']),
        "ci_lower": float(best['ci_lower']),
        "ci_upper": float(best['ci_upper']),
        "mean_accuracy": float(df['test_accuracy'].mean()),
        "n_layers": len(results),
        "timestamp": datetime.now().isoformat()
    }

    with open(output_dir / f"logit_summary_{model}_{condition}.json", 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama-3.2-3b", choices=list(config.MODELS.keys()))
    parser.add_argument("--condition", default=config.DEFAULT_CONDITION)
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    output_dir = config.RESULTS_DIR
    logger = setup_logging(output_dir, args.model, args.condition)

    logger.info("=" * 60)
    logger.info("LOGISTIC REGRESSION PRICE PREDICTOR")
    logger.info(f"Model: {args.model}, Condition: {args.condition}")
    logger.info(f"cuML available: {CUML_AVAILABLE}, Using GPU: {not args.no_gpu and CUML_AVAILABLE}")
    logger.info("=" * 60)

    # Load activations
    act_file = config.ACTIVATIONS_DIR / f"{args.model}_{args.condition}_activations.npz"
    if not act_file.exists():
        logger.error(f"Not found: {act_file}")
        sys.exit(1)

    data = load_activations(act_file)
    activations, labels = data["activations"], data["labels"]
    logger.info(f"Loaded: {activations.shape}, labels: {labels.shape}")

    # Splits
    splits = create_stratified_splits(len(labels), labels)
    logger.info(f"Splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # Run
    results = run_all_layers(activations, labels, splits, not args.no_gpu, logger)

    # Save
    summary = save_results(results, output_dir, args.model, args.condition)

    logger.info("=" * 60)
    logger.info(f"BEST: Layer {summary['best_layer']}, Accuracy {summary['best_accuracy']:.4f}")
    logger.info(f"95% CI: [{summary['ci_lower']:.4f}, {summary['ci_upper']:.4f}]")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
