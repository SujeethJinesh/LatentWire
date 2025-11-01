"""
Procrustes alignment experiment.

This module provides zero-training baseline alignment using SVD-based
orthogonal/affine transformations between model hidden states.

References:
    - Schönemann, "A generalized solution of the orthogonal Procrustes problem"
      Psychometrika 1966
    - Lester et al., "Transferring Features Across Language Models With Model Stitching"
      arXiv:2506.06609 (June 2025)
"""

import sys
from pathlib import Path

# Import from original implementation
# This ensures correctness while maintaining modular structure
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "experimental" / "learning"))
from unified_cross_model_experiments import run_procrustes_experiment as _run_procrustes


def run_procrustes_experiment(model_a_id=None, model_b_id=None):
    """
    Run Procrustes alignment experiment.

    Tests SVD-based alignment between hidden states at layers [0, 8, 16, 24].
    Provides zero-training baseline for cross-model alignment.

    Args:
        model_a_id: Source model identifier (e.g., "meta-llama/Llama-3.1-8B")
        model_b_id: Target model identifier (e.g., "mistralai/Mistral-7B-v0.1")

    Returns:
        dict: Results containing CKA scores, generation quality, etc.

    Expected results:
        - Same-vocab models (Llama 3.1 ↔ Llama 3.2): CKA 0.7-0.8
        - Cross-vocab models (Llama ↔ Mistral): CKA 0.3-0.5
    """
    return _run_procrustes(model_a_id=model_a_id, model_b_id=model_b_id)


__all__ = ["run_procrustes_experiment"]
