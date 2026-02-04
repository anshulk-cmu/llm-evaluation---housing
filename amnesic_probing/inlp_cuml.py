"""
GPU-Accelerated INLP (Iterative Nullspace Projection) using cuML

This implementation uses RAPIDS cuML for 10-20x speedup over sklearn.
Based on: Ravfogel et al. (2020) "Null It Out: Guarding Protected Attributes 
by Iterative Nullspace Projection" (ACL 2020)

Key advantage: With 2x A6000 GPUs, we can process all 5,130 samples
with 300 iterations per feature in ~1-1.5 hours total.
"""

import numpy as np
import cupy as cp
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings

# Scaling for activations (critical for logistic regression)
from sklearn.preprocessing import StandardScaler

# Try to import cuML, fall back to sklearn if not available
try:
    from cuml.linear_model import LogisticRegression as cuLogisticRegression
    from cuml.svm import LinearSVC as cuLinearSVC
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False
    warnings.warn("cuML not available, falling back to sklearn (slower)")
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC


def get_rowspace_projection_gpu(W: cp.ndarray) -> cp.ndarray:
    """
    Compute the projection matrix onto the rowspace of W (GPU version).
    
    Args:
        W: Weight matrix from classifier (on GPU)
        
    Returns:
        P_W: Projection matrix onto W's rowspace
    """
    if cp.allclose(W, 0):
        return cp.zeros((W.shape[1], W.shape[1]), dtype=W.dtype)
    
    # Orthogonal basis for rowspace via QR decomposition
    # W.T has shape (d, n_classes) - we want basis for column space of W.T = rowspace of W
    Q, R = cp.linalg.qr(W.T)
    
    # Handle rank deficiency
    rank = cp.sum(cp.abs(cp.diag(R)) > 1e-10).item()
    Q = Q[:, :rank]
    
    # Projection matrix: P = Q @ Q.T
    P_W = Q @ Q.T
    
    return P_W


def get_nullspace_projection_gpu(rowspace_projections: List[cp.ndarray], 
                                  input_dim: int) -> cp.ndarray:
    """
    Compute projection to intersection of nullspaces.
    
    Uses Ben-Israel (2013) formula:
    N(w1) ∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))
    
    Args:
        rowspace_projections: List of rowspace projection matrices
        input_dim: Dimensionality of input vectors
        
    Returns:
        P: Projection matrix to intersection of nullspaces
    """
    I = cp.eye(input_dim, dtype=cp.float32)
    
    if len(rowspace_projections) == 0:
        return I
    
    # Sum of rowspace projections
    Q = cp.sum(cp.stack(rowspace_projections), axis=0)
    
    # Projection to nullspace of Q
    P_Q = get_rowspace_projection_gpu(Q)
    P = I - P_Q
    
    return P


class INLPProjector:
    """
    GPU-accelerated INLP for removing linear information from representations.
    
    Example usage:
        projector = INLPProjector(n_iterations=200, use_gpu=True)
        P = projector.fit(X_train, y_train, X_dev, y_dev)
        X_projected = projector.transform(X)
    """
    
    def __init__(self, 
                 n_iterations: int = 200,
                 classifier: str = 'logistic',  # 'logistic' or 'svm'
                 min_accuracy: float = 0.51,
                 min_iterations: int = 10,  # Run at least this many iterations
                 use_gpu: bool = True,
                 random_state: int = 42,
                 verbose: bool = True):
        """
        Args:
            n_iterations: Maximum number of INLP iterations
            classifier: Type of classifier ('logistic' or 'svm')
            min_accuracy: Stop when classifier accuracy drops below this
            min_iterations: Minimum iterations before allowing early stopping
            use_gpu: Whether to use cuML (GPU) or sklearn (CPU)
            random_state: Random seed for reproducibility
            verbose: Whether to show progress bar
        """
        self.n_iterations = n_iterations
        self.classifier_type = classifier
        self.min_accuracy = min_accuracy
        self.min_iterations = min_iterations
        self.use_gpu = use_gpu and CUML_AVAILABLE
        self.random_state = random_state
        self.verbose = verbose
        
        # Results storage
        self.projection_matrix_ = None
        self.rowspace_projections_ = []
        self.classifiers_ = []
        self.accuracies_ = []
        self.n_iterations_used_ = 0
        
    def _get_classifier(self):
        """Get a fresh classifier instance."""
        if self.use_gpu:
            if self.classifier_type == 'logistic':
                # cuML LogisticRegression - use QN solver which is most robust
                return cuLogisticRegression(
                    max_iter=2000,
                    C=1.0,
                    solver='qn',  # Quasi-Newton solver, more stable
                    verbose=0
                )
            else:
                return cuLinearSVC(
                    max_iter=2000,
                    C=1.0,
                    verbose=0
                )
        else:
            if self.classifier_type == 'logistic':
                return LogisticRegression(
                    max_iter=2000,
                    C=1.0,
                    class_weight='balanced',
                    random_state=self.random_state
                )
            else:
                return LinearSVC(
                    max_iter=2000,
                    C=1.0,
                    class_weight='balanced',
                    dual=False,
                    random_state=self.random_state
                )
    
    def _to_gpu(self, X: np.ndarray) -> cp.ndarray:
        """Move numpy array to GPU."""
        if self.use_gpu:
            return cp.asarray(X, dtype=cp.float32)
        return X
    
    def _to_cpu(self, X) -> np.ndarray:
        """Move cupy array to CPU."""
        if self.use_gpu and isinstance(X, cp.ndarray):
            return cp.asnumpy(X)
        return np.asarray(X)
    
    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            X_dev: Optional[np.ndarray] = None,
            y_dev: Optional[np.ndarray] = None) -> 'INLPProjector':
        """
        Fit INLP to remove the ability to predict y from X.
        
        Args:
            X_train: Training representations, shape (n_samples, n_features)
            y_train: Training labels (binary), shape (n_samples,)
            X_dev: Dev representations for early stopping (optional)
            y_dev: Dev labels (optional)
            
        Returns:
            self
        """
        if X_dev is None:
            X_dev = X_train
            y_dev = y_train
            
        input_dim = X_train.shape[1]
        
        # Move to GPU if available
        if self.use_gpu:
            X_train_gpu = self._to_gpu(X_train)
            X_dev_gpu = self._to_gpu(X_dev)
            X_train_current = X_train_gpu.copy()
            X_dev_current = X_dev_gpu.copy()
            # cuML requires cupy arrays for labels too
            y_train_gpu = cp.asarray(y_train.astype(np.float32))
            y_dev_np = y_dev.astype(np.float32)  # Keep numpy copy for accuracy calc
        else:
            X_train_current = X_train.copy()
            X_dev_current = X_dev.copy()
        
        self.rowspace_projections_ = []
        self.accuracies_ = []
        
        pbar = tqdm(range(self.n_iterations), disable=not self.verbose,
                   desc="INLP iterations")
        
        for i in pbar:
            # Train classifier
            clf = self._get_classifier()
            
            # cuML needs cupy arrays, sklearn needs numpy
            if self.use_gpu:
                clf.fit(X_train_current, y_train_gpu)
                preds = clf.predict(X_dev_current)
                acc = float((self._to_cpu(preds) == y_dev_np).mean())
            else:
                clf.fit(X_train_current, y_train)
                acc = clf.score(X_dev_current, y_dev)
            
            self.accuracies_.append(acc)
            pbar.set_description(f"INLP iter {i}, acc: {acc:.3f}")
            
            # Check stopping criterion (only after min_iterations)
            if i >= self.min_iterations and acc < self.min_accuracy:
                if self.verbose:
                    print(f"\nStopped at iteration {i}: accuracy {acc:.3f} < {self.min_accuracy}")
                break
            
            # Get classifier weights
            W = clf.coef_
            if self.use_gpu:
                W = self._to_gpu(W) if not isinstance(W, cp.ndarray) else W
            else:
                W = np.asarray(W)
            
            # Compute rowspace projection
            if self.use_gpu:
                P_rowspace = get_rowspace_projection_gpu(W)
            else:
                # CPU version using scipy
                import scipy.linalg
                if np.allclose(W, 0):
                    w_basis = np.zeros_like(W.T)
                else:
                    w_basis = scipy.linalg.orth(W.T)
                P_rowspace = w_basis @ w_basis.T
            
            self.rowspace_projections_.append(P_rowspace)
            
            # Project data for next iteration (autoregressive)
            if self.use_gpu:
                P = get_nullspace_projection_gpu(self.rowspace_projections_, input_dim)
                X_train_current = (P @ X_train_gpu.T).T
                X_dev_current = (P @ X_dev_gpu.T).T
            else:
                I = np.eye(input_dim)
                Q = np.sum(self.rowspace_projections_, axis=0)
                import scipy.linalg
                w_basis = scipy.linalg.orth(Q.T) if not np.allclose(Q, 0) else np.zeros_like(Q.T)
                P = I - w_basis @ w_basis.T
                X_train_current = (P @ X_train.T).T
                X_dev_current = (P @ X_dev.T).T
        
        self.n_iterations_used_ = len(self.rowspace_projections_)
        
        # Compute final projection matrix
        if self.use_gpu:
            self.projection_matrix_ = get_nullspace_projection_gpu(
                self.rowspace_projections_, input_dim
            )
            # Keep on GPU for fast transforms
        else:
            I = np.eye(input_dim)
            if len(self.rowspace_projections_) > 0:
                Q = np.sum(self.rowspace_projections_, axis=0)
                import scipy.linalg
                w_basis = scipy.linalg.orth(Q.T) if not np.allclose(Q, 0) else np.zeros_like(Q.T)
                self.projection_matrix_ = I - w_basis @ w_basis.T
            else:
                self.projection_matrix_ = I
        
        if self.verbose:
            print(f"\nINLP complete: {self.n_iterations_used_} iterations used")
            print(f"Initial accuracy: {self.accuracies_[0]:.3f}")
            print(f"Final accuracy: {self.accuracies_[-1]:.3f}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the learned projection to remove information.
        
        Args:
            X: Input representations, shape (n_samples, n_features)
            
        Returns:
            X_projected: Projected representations
        """
        if self.projection_matrix_ is None:
            raise ValueError("Must call fit() before transform()")
        
        if self.use_gpu:
            X_gpu = self._to_gpu(X)
            X_proj = (self.projection_matrix_ @ X_gpu.T).T
            return self._to_cpu(X_proj)
        else:
            return (self.projection_matrix_ @ X.T).T
    
    def fit_transform(self,
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_dev: Optional[np.ndarray] = None,
                      y_dev: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit INLP and transform training data."""
        self.fit(X_train, y_train, X_dev, y_dev)
        return self.transform(X_train)
    
    def get_projection_matrix(self) -> np.ndarray:
        """Get the projection matrix as numpy array."""
        if self.projection_matrix_ is None:
            raise ValueError("Must call fit() first")
        return self._to_cpu(self.projection_matrix_)
    
    def save(self, path: str):
        """Save projection matrix to file."""
        P = self.get_projection_matrix()
        np.save(path, P)
        print(f"Saved projection matrix to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'INLPProjector':
        """Load projection matrix from file."""
        obj = cls()
        obj.projection_matrix_ = np.load(path)
        return obj


def run_inlp_for_feature(activations: np.ndarray,
                         labels: np.ndarray,
                         feature_name: str,
                         n_iterations: int = 200,
                         train_ratio: float = 0.7,
                         val_ratio: float = 0.1,
                         use_gpu: bool = True,
                         random_state: int = 42) -> Tuple[np.ndarray, StandardScaler, dict]:
    """
    Run INLP to remove a single feature from activations.
    
    Args:
        activations: Model activations, shape (n_samples, hidden_dim)
        labels: Binary labels for the feature, shape (n_samples,)
        feature_name: Name of the feature being removed
        n_iterations: Maximum INLP iterations
        train_ratio: Fraction of data for training (default 0.7)
        val_ratio: Fraction of data for validation/early stopping (default 0.1)
        use_gpu: Whether to use GPU acceleration
        random_state: Random seed
        
    Returns:
        P: Projection matrix (learned on scaled data)
        scaler: StandardScaler fitted on training data - MUST apply before using P
        info: Dictionary with training info
        
    Note: 
        - Remaining 20% is held out as test set (not used in INLP, only for verification)
        - P expects SCALED input: X_proj = P @ scaler.transform(X).T
    """
    # 70/10/20 train/val/test split
    np.random.seed(random_state)
    n = len(activations)
    indices = np.random.permutation(n)
    
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    X_train_raw = activations[train_idx]
    y_train = labels[train_idx]
    X_val_raw = activations[val_idx]
    y_val = labels[val_idx]
    X_test_raw = activations[test_idx]
    y_test = labels[test_idx]
    
    print(f"\n{'='*60}")
    print(f"INLP for feature: {feature_name}")
    print(f"{'='*60}")
    print(f"Samples: {n} (train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)})")
    print(f"Class balance: {y_train.mean():.2%} positive")
    print(f"Using GPU: {use_gpu and CUML_AVAILABLE}")
    
    # CRITICAL: Scale activations (like linear probing does)
    # This is essential for logistic regression to work properly
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)
    print(f"Applied StandardScaler to activations")
    
    # First, verify we can actually predict this feature (on SCALED data)
    # Use held-out TEST set for this check (not val which is used in INLP)
    if use_gpu and CUML_AVAILABLE:
        test_clf = cuLogisticRegression(max_iter=2000, C=1.0, solver='qn')
        test_clf.fit(cp.asarray(X_train), cp.asarray(y_train.astype(np.float32)))
        test_preds = test_clf.predict(cp.asarray(X_test))
        initial_probe_acc = float((cp.asnumpy(test_preds) == y_test).mean())
    else:
        test_clf = LogisticRegression(max_iter=2000, C=1.0, random_state=random_state)
        test_clf.fit(X_train, y_train)
        initial_probe_acc = test_clf.score(X_test, y_test)
    
    print(f"Initial probe accuracy (before INLP, on test set): {initial_probe_acc:.3f}")
    
    if initial_probe_acc < 0.55:
        print(f"WARNING: Initial probe accuracy is low ({initial_probe_acc:.3f}). Feature may not be well-encoded at this layer.")
    
    # Run INLP - train on train set, early stopping on val set
    projector = INLPProjector(
        n_iterations=n_iterations,
        classifier='logistic',
        min_accuracy=0.52,  # Just above chance
        min_iterations=50,  # Run at least 50 iterations
        use_gpu=use_gpu,
        random_state=random_state,
        verbose=True
    )
    
    projector.fit(X_train, y_train, X_val, y_val)
    
    # Verify erasure worked on HELD-OUT TEST SET (not used in training)
    X_train_proj = projector.transform(X_train)
    X_test_proj = projector.transform(X_test)
    
    # Train new classifier on projected data to verify erasure on test set
    if use_gpu and CUML_AVAILABLE:
        verify_clf = cuLogisticRegression(max_iter=1000, C=1.0, solver='qn')
        verify_clf.fit(cp.asarray(X_train_proj), cp.asarray(y_train.astype(np.float32)))
        verify_preds = verify_clf.predict(cp.asarray(X_test_proj))
        verify_acc = float((cp.asnumpy(verify_preds) == y_test).mean())
    else:
        verify_clf = LogisticRegression(max_iter=1000, C=1.0, random_state=random_state)
        verify_clf.fit(X_train_proj, y_train)
        verify_acc = verify_clf.score(X_test_proj, y_test)
    
    print(f"\nVerification (on held-out test set): Post-INLP probe accuracy = {verify_acc:.3f} (should be ~0.50)")
    
    # Check if erasure was successful
    if verify_acc > 0.60:
        print(f"WARNING: Erasure incomplete! Post-INLP accuracy {verify_acc:.3f} > 0.60")
        print(f"  Initial probe accuracy was: {initial_probe_acc:.3f}")
        print(f"  Iterations used: {projector.n_iterations_used_}")
    
    info = {
        'feature': feature_name,
        'n_iterations_used': projector.n_iterations_used_,
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'n_test': len(test_idx),
        'initial_probe_accuracy': initial_probe_acc,  # Probe accuracy BEFORE INLP (on test set)
        'initial_accuracy': projector.accuracies_[0] if projector.accuracies_ else initial_probe_acc,
        'final_accuracy': projector.accuracies_[-1] if projector.accuracies_ else initial_probe_acc,
        'verification_accuracy': verify_acc,  # Post-INLP accuracy on test set
        'accuracies': projector.accuracies_
    }
    
    # Return P, scaler, and info - scaler needed to apply P correctly
    return projector.get_projection_matrix(), scaler, info


if __name__ == "__main__":
    # Quick test
    print("Testing INLP implementation...")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    hidden_dim = 100
    
    # Create data where first 5 dimensions encode the label
    X = np.random.randn(n_samples, hidden_dim).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)
    
    # Add signal to first few dimensions
    X[y == 1, :5] += 0.5
    X[y == 0, :5] -= 0.5
    
    # Test INLP
    P, scaler, info = run_inlp_for_feature(
        X, y, "test_feature",
        n_iterations=50,
        use_gpu=CUML_AVAILABLE
    )
    
    print(f"\nTest complete!")
    print(f"Projection matrix shape: {P.shape}")
    print(f"Scaler mean shape: {scaler.mean_.shape}")
    print(f"Iterations used: {info['n_iterations_used']}")
    print(f"Initial accuracy: {info['initial_probe_accuracy']:.3f}")
    print(f"Final accuracy: {info['final_accuracy']:.3f}")
