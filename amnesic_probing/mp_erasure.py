"""
Mean Projection (MP) erasure for Amnesic Probing.

Implements rank-1 concept erasure via Mean Projection:
  - Orthogonal projection removing the class-centroid difference direction
  - Removes exactly 1 direction from the representation for binary features
    (vs INLP which removed 50-300 directions and destroyed the price signal)

Reference:
  - Dobrzeniecka et al. (2025) "Improving Causal Interventions in Amnesic Probing
    with Mean Projection or LEACE"
"""

import numpy as np
from typing import Tuple, Dict, Optional, Callable
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

# Try cuML for GPU-accelerated probing
try:
    from cuml.linear_model import LogisticRegression as cuLogisticRegression
    import cupy as cp
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False


# =============================================================================
# Eraser Objects
# =============================================================================

@dataclass
class ErasureResult:
    """Container for an erasure result that can transform activations."""
    method: str
    feature_name: str
    projection_matrix: np.ndarray  # (d, d) — P such that x' = P @ (x - bias) + bias
    bias: Optional[np.ndarray]     # (d,) — mean vector for centering
    rank_removed: int

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Apply erasure to activations. X shape: (n, d)."""
        if self.bias is not None:
            X_centered = X - self.bias
            return (self.projection_matrix @ X_centered.T).T + self.bias
        return (self.projection_matrix @ X.T).T


# =============================================================================
# Mean Projection (MP)
# =============================================================================

def fit_mean_projection(X_train: np.ndarray,
                        y_train: np.ndarray) -> ErasureResult:
    """
    Fit Mean Projection erasure for binary labels.

    Computes the direction between class centroids and projects it out.
    For binary labels, this removes exactly 1 direction (rank-1).

    P_MP = I - ŵŵᵀ  where ŵ = (μ₁ - μ₀) / ‖μ₁ - μ₀‖

    Args:
        X_train: Training activations, shape (n, d)
        y_train: Binary labels, shape (n,)

    Returns:
        ErasureResult with orthogonal projection matrix
    """
    d = X_train.shape[1]

    mu_0 = X_train[y_train == 0].mean(axis=0)
    mu_1 = X_train[y_train == 1].mean(axis=0)

    w = mu_1 - mu_0
    w_norm = np.linalg.norm(w)

    if w_norm < 1e-10:
        # Classes are indistinguishable — no projection needed
        return ErasureResult(
            method='mp',
            feature_name='',
            projection_matrix=np.eye(d, dtype=np.float32),
            bias=None,
            rank_removed=0
        )

    w_hat = w / w_norm
    P = np.eye(d, dtype=np.float32) - np.outer(w_hat, w_hat).astype(np.float32)

    return ErasureResult(
        method='mp',
        feature_name='',
        projection_matrix=P,
        bias=None,  # MP is origin-invariant for orthogonal projection
        rank_removed=1
    )


# =============================================================================
# Random Erasure (for information control baseline)
# =============================================================================

def fit_random_erasure(d: int, random_state: int = 42) -> ErasureResult:
    """
    Remove 1 random direction from the representation.

    Used as a control: if target erasure causes more damage than this,
    the damage is specific to the erased feature, not just dimensionality loss.

    Args:
        d: Dimensionality of activations
        random_state: Random seed

    Returns:
        ErasureResult with random rank-1 orthogonal projection
    """
    rng = np.random.RandomState(random_state)
    w = rng.randn(d).astype(np.float32)
    w_hat = w / np.linalg.norm(w)
    P = np.eye(d, dtype=np.float32) - np.outer(w_hat, w_hat)

    return ErasureResult(
        method='random',
        feature_name='random',
        projection_matrix=P,
        bias=None,
        rank_removed=1
    )


# =============================================================================
# Projection Matrix Validation
# =============================================================================

def validate_projection(eraser: ErasureResult, tol: float = 1e-4) -> None:
    """Sanity-check projection matrix properties. Warns on failure."""
    P = eraser.projection_matrix.astype(np.float64)
    d = P.shape[0]

    # Check idempotent: P² ≈ P
    P2 = P @ P
    idem_err = np.max(np.abs(P2 - P))
    if idem_err > tol:
        warnings.warn(f"{eraser.method}/{eraser.feature_name}: P is not idempotent "
                       f"(max |P²-P| = {idem_err:.2e})")

    # Check rank removed
    rank = np.linalg.matrix_rank(P, tol=1e-3)
    expected_rank = d - eraser.rank_removed
    if rank != expected_rank:
        warnings.warn(f"{eraser.method}/{eraser.feature_name}: rank(P) = {rank}, "
                       f"expected {expected_rank}")

    # For MP (orthogonal projection): P should be symmetric
    if eraser.method == 'mp':
        sym_err = np.max(np.abs(P - P.T))
        if sym_err > tol:
            warnings.warn(f"mp/{eraser.feature_name}: P is not symmetric "
                           f"(max |P-Pᵀ| = {sym_err:.2e})")


# =============================================================================
# Probing Utilities
# =============================================================================

def train_probe(X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray, y_test: np.ndarray,
                use_gpu: bool = False) -> float:
    """Train logistic regression probe and return test accuracy."""
    if use_gpu and CUML_AVAILABLE:
        clf = cuLogisticRegression(max_iter=2000, C=1.0, solver='qn')
        clf.fit(cp.asarray(X_train.astype(np.float32)),
                cp.asarray(y_train.astype(np.float32)))
        preds = cp.asnumpy(clf.predict(cp.asarray(X_test.astype(np.float32))))
        return float((preds == y_test).mean())
    else:
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
        clf.fit(X_train, y_train)
        return clf.score(X_test, y_test)


def compute_cosine_similarity(X_original: np.ndarray,
                              X_erased: np.ndarray) -> float:
    """
    Compute mean cosine similarity between original and erased activations.
    Higher = less distortion. MP rank-1 erasure should be >0.90.
    """
    # Normalize rows
    norms_orig = np.linalg.norm(X_original, axis=1, keepdims=True)
    norms_erased = np.linalg.norm(X_erased, axis=1, keepdims=True)

    # Avoid division by zero
    norms_orig = np.maximum(norms_orig, 1e-10)
    norms_erased = np.maximum(norms_erased, 1e-10)

    X_orig_normed = X_original / norms_orig
    X_erased_normed = X_erased / norms_erased

    cosines = np.sum(X_orig_normed * X_erased_normed, axis=1)
    return float(np.mean(cosines))


# =============================================================================
# Main Erasure Pipeline
# =============================================================================

def run_erasure_for_feature(activations: np.ndarray,
                            labels: np.ndarray,
                            feature_name: str,
                            method: str = 'mp',
                            train_ratio: float = 0.7,
                            val_ratio: float = 0.1,
                            use_gpu: bool = False,
                            random_state: int = 42) -> Tuple[ErasureResult, dict]:
    """
    Run concept erasure for a single binary feature.

    Steps:
      1. Split data (same splits as INLP code for comparability)
      2. Verify feature is encoded (pre-erasure probe)
      3. Fit MP eraser on training data
      4. Verify erasure worked (post-erasure probe on test set)
      5. Compute cosine similarity

    Args:
        activations: Model activations, shape (n, d)
        labels: Binary feature labels, shape (n,)
        feature_name: Name of feature being erased
        method: 'mp'
        train_ratio: Fraction for training (default 0.7)
        val_ratio: Fraction for validation (default 0.1)
        use_gpu: Whether to use cuML for probes
        random_state: Random seed (42 matches INLP code)

    Returns:
        eraser: ErasureResult object (callable)
        info: Dict with erasure statistics
    """
    # Same split as INLP code (seed=42, 70/10/20)
    np.random.seed(random_state)
    n = len(activations)
    indices = np.random.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    X_train = activations[train_idx].astype(np.float32)
    y_train = labels[train_idx].astype(int)
    X_test = activations[test_idx].astype(np.float32)
    y_test = labels[test_idx].astype(int)

    print(f"\n{'='*60}")
    print(f"{method.upper()} erasure for feature: {feature_name}")
    print(f"{'='*60}")
    print(f"Samples: {n} (train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)})")
    print(f"Class balance: {y_train.mean():.2%} positive")

    # Step 1: Pre-erasure probe (verify feature is encoded)
    initial_probe_acc = train_probe(X_train, y_train, X_test, y_test, use_gpu)
    print(f"Pre-erasure probe accuracy: {initial_probe_acc:.3f}")

    if initial_probe_acc < 0.55:
        print(f"WARNING: Low encoding ({initial_probe_acc:.3f}). Feature may not be linearly encoded.")

    # Step 2: Fit eraser
    if method == 'mp':
        eraser = fit_mean_projection(X_train, y_train)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'mp'.")

    eraser.feature_name = feature_name
    print(f"Rank removed: {eraser.rank_removed}")
    validate_projection(eraser)

    # Step 3: Verify erasure on test set
    X_train_erased = eraser(X_train)
    X_test_erased = eraser(X_test)

    verification_acc = train_probe(X_train_erased, y_train, X_test_erased, y_test, use_gpu)
    print(f"Post-erasure probe accuracy: {verification_acc:.3f} (target: ~0.50)")

    # Threshold accounts for class imbalance: majority class baseline is chance-level
    majority_baseline = max(y_train.mean(), 1 - y_train.mean())
    warn_threshold = majority_baseline + 0.03
    if verification_acc > warn_threshold:
        print(f"WARNING: Incomplete erasure! Post-erasure accuracy {verification_acc:.3f} "
              f"> {warn_threshold:.3f} (majority baseline {majority_baseline:.3f} + 0.03)")

    # Step 4: Cosine similarity
    cos_sim = compute_cosine_similarity(X_test, X_test_erased)
    print(f"Cosine similarity (original vs erased): {cos_sim:.4f}")

    info = {
        'feature': feature_name,
        'method': method,
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'n_test': len(test_idx),
        'initial_probe_accuracy': initial_probe_acc,
        'verification_accuracy': verification_acc,
        'rank_removed': eraser.rank_removed,
        'cosine_similarity': cos_sim,
    }

    return eraser, info
