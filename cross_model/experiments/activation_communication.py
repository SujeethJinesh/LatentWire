"""
Activation communication experiment.

Reproduces "Communicating Activations Between Language Model Agents" (Ramesh & Li, ICML 2025).

This experiment:
1. Trains learned projections on C4 dataset (if dimensions mismatch)
2. Tests text similarity by injecting activations across layers
3. Evaluates on real tasks (SQuAD Q&A and GSM8K math reasoning)

References:
    - Ramesh & Li, "Communicating Activations Between Language Model Agents"
      arXiv:2501.14082 (ICML 2025)
"""

import sys
from pathlib import Path

# Import from original implementation
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "experimental" / "learning"))
from unified_cross_model_experiments import run_activation_communication_experiment as _run_activation_communication


def run_activation_communication_experiment(model_a_id=None, model_b_id=None):
    """
    Run activation communication experiment.

    Method:
    1. Run source model A on prompt, extract hidden state h_A at layer j (last token)
    2. If dimensions mismatch, project: h_proj = W @ h_A (W learned on C4)
    3. Inject h_proj into target model B at layer j during generation
    4. Measure generation quality vs text baseline

    Includes:
        - Text similarity test across multiple layers
        - SQuAD Q&A evaluation (20 samples)
        - GSM8K math reasoning evaluation (20 samples)

    Args:
        model_a_id: Source model (e.g., "meta-llama/Llama-3.1-8B")
        model_b_id: Target model (e.g., "meta-llama/Llama-3.2-3B")

    Returns:
        dict: Results containing:
            - layers: Per-layer text similarity results
            - task_evaluation: SQuAD and GSM8K performance
            - config: Model configuration

    Expected results (from paper):
        - Up to 27% improvement over natural language communication
        - <1/4 compute cost vs text
        - Works across models with different vocabularies
    """
    return _run_activation_communication(model_a_id=model_a_id, model_b_id=model_b_id)


__all__ = ["run_activation_communication_experiment"]
