"""
Amnesic Probing Module

Tests whether the model USES the features it encodes for price prediction,
establishing causal (not just correlational) links between encoded information
and behavior.

Method:
  - Mean Projection (MP): Dobrzeniecka et al. (2025) rank-1 orthogonal erasure

Three-step causal framework from Dobrzeniecka et al. (2025):
  Step 1: Erasure + Verification
  Step 2: Information Control (target vs random erasure)
  Step 3: Selectivity Control (gold label recovery)
"""

from .mp_erasure import (
    run_erasure_for_feature,
    fit_mean_projection,
    fit_random_erasure,
    ErasureResult,
)

__all__ = [
    'run_erasure_for_feature',
    'fit_mean_projection',
    'fit_random_erasure',
    'ErasureResult',
]
