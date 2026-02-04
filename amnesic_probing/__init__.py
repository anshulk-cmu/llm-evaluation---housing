"""
Amnesic Probing Module

Implements the amnesic probing methodology from:
- Elazar et al. (2021) "Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals" (TACL)
- Ravfogel et al. (2020) "Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection" (ACL)

This module tests whether the model USES the features it encodes for price prediction,
establishing causal (not just correlational) links between encoded information and behavior.
"""

from .inlp_cuml import INLPProjector, run_inlp_for_feature, CUML_AVAILABLE

__all__ = ['INLPProjector', 'run_inlp_for_feature', 'CUML_AVAILABLE']
