"""
Learned adapter experiments (LoRA, Linear, Affine).

Trains adapter modules with InfoNCE contrastive loss to align
hidden representations between models.

References:
    - Hu et al., "LoRA: Low-Rank Adaptation" arXiv:2106.09685 (ICLR 2022)
    - Wang & Isola, "Understanding Contrastive Representation Learning" (ICML 2020)
"""

import sys
from pathlib import Path

# Import from original implementation
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "experimental" / "learning"))
from unified_cross_model_experiments import run_adapter_experiment as _run_adapter


def run_lora_adapter_experiment(model_a_id=None, model_b_id=None, gpu_id=0):
    """
    Run LoRA adapter experiment.

    Trains low-rank adapter (260K params, rank=32) with InfoNCE loss.
    Most parameter-efficient option.

    Args:
        model_a_id: Source model identifier
        model_b_id: Target model identifier
        gpu_id: GPU ID to use (default: 0)

    Returns:
        dict: Training results (loss curves, CKA scores, etc.)

    Expected results:
        - CKA 0.8-0.9 (better than Procrustes)
        - 260K params (vs 16M for full linear)
        - Generation loss 2.5-3.5
    """
    return _run_adapter(adapter_type="lora", gpu_id=gpu_id,
                       model_a_id=model_a_id, model_b_id=model_b_id)


def run_linear_adapter_experiment(model_a_id=None, model_b_id=None, gpu_id=0):
    """
    Run Linear adapter experiment.

    Trains full-rank linear projection (16M params, no bias).
    Upper bound on linear alignment capacity.

    Args:
        model_a_id: Source model identifier
        model_b_id: Target model identifier
        gpu_id: GPU ID to use (default: 0)

    Returns:
        dict: Training results

    Expected results:
        - CKA 0.8-0.9 (similar to LoRA)
        - 16M params (50× more than LoRA)
        - Tests parameter efficiency
    """
    return _run_adapter(adapter_type="linear", gpu_id=gpu_id,
                       model_a_id=model_a_id, model_b_id=model_b_id)


def run_affine_adapter_experiment(model_a_id=None, model_b_id=None, gpu_id=0):
    """
    Run Affine adapter experiment.

    Trains linear + bias transformation (16M + 3K params).
    Tests whether bias term improves alignment.

    Args:
        model_a_id: Source model identifier
        model_b_id: Target model identifier
        gpu_id: GPU ID to use (default: 0)

    Returns:
        dict: Training results

    Expected results:
        - CKA 0.8-0.9 (test if bias helps over linear)
        - Marginal improvement expected (bias often redundant after centering)
    """
    return _run_adapter(adapter_type="affine", gpu_id=gpu_id,
                       model_a_id=model_a_id, model_b_id=model_b_id)


__all__ = [
    "run_lora_adapter_experiment",
    "run_linear_adapter_experiment",
    "run_affine_adapter_experiment",
]
