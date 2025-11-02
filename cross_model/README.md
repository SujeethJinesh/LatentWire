# Cross-Model Activation Communication

Refactored implementation of "Communicating Activations Between Language Model Agents" (Ramesh & Li, ICML 2025).

## Structure

```
cross_model/
├── __init__.py                 # Package initialization
├── README.md                   # This file
├── models.py                   # LearnedProjection model architecture
├── adapter_models.py           # ProcrustesAlignment, LoRA, Linear, Affine adapters
├── metrics.py                  # InfoNCE, CKA, AlignmentUniformity metrics
├── training.py                 # Projection training on C4 dataset
├── checkpointing.py            # Save/load utilities for projections
├── utils.py                    # DDP setup, device detection, helpers
├── experiments/                # Individual experiment modules
│   ├── __init__.py
│   ├── procrustes.py           # SVD-based alignment (zero-training baseline)
│   ├── activation_communication.py  # Ramesh & Li (2025) reproduction
│   └── adapters.py             # LoRA/Linear/Affine adapter training
└── run_experiments.py          # Main entry point
```

## Components

### Core Modules

- **models.py**: `LearnedProjection` class - learned linear projection for dimension mismatches (e.g., 3072 → 4096 dims)

- **adapter_models.py**: Alignment architectures:
  - `ProcrustesAlignment`: SVD-based orthogonal/affine alignment (zero-training baseline)
  - `LinearAdapter`: Full-rank linear projection (16M params)
  - `AffineAdapter`: Linear + bias (16M + 3K params)
  - `LoRAAdapter`: Low-rank adaptation (260K params, rank=32)

- **metrics.py**: Evaluation metrics and loss functions:
  - `InfoNCE`: Contrastive loss for alignment training
  - `CKA`: Centered Kernel Alignment (debiased HSIC estimator)
  - `AlignmentUniformity`: Contrastive learning quality metrics
  - `mean_pooling`: Attention-masked mean pooling

- **training.py**: `train_learned_projection()` - trains projections on C4 dataset (Ramesh & Li 2025 methodology)

- **checkpointing.py**: Save/load utilities for trained projections and experimental results

- **utils.py**: Platform detection, DDP initialization (60-min timeout for long experiments), distributed helpers

### Experiments

Individual experiment modules in `experiments/`:

- **procrustes.py**: SVD-based alignment across layers [0,8,16,24]
  - Zero-training baseline
  - Expected: CKA 0.7-0.8 (same-vocab), 0.3-0.5 (cross-vocab)

- **activation_communication.py**: Reproduces Ramesh & Li (ICML 2025)
  - Text similarity via activation injection
  - SQuAD Q&A evaluation (20 samples)
  - GSM8K math reasoning (20 samples)
  - Expected: Up to 27% improvement over natural language

- **adapters.py**: Learned adapter training with InfoNCE loss
  - `run_lora_adapter_experiment()`: 260K params, rank=32
  - `run_linear_adapter_experiment()`: 16M params, no bias
  - `run_affine_adapter_experiment()`: 16M + 3K params, with bias
  - Expected: CKA 0.8-0.9, better than Procrustes baseline

## Usage

### Full Experiment Suite

Run all experiments (Procrustes, Activation Communication, Adapters):

```bash
# Standard workflow (recommended)
git pull && rm -rf runs && PYTHONPATH=. torchrun --nproc_per_node=4 cross_model/run_experiments.py
```

This runs the complete experiment suite with:
- Procrustes alignment (Llama ↔ Mistral, Llama ↔ Llama)
- Activation communication with text similarity + task evaluation (SQuAD, GSM8K)
- Adapter training experiments (LoRA, Linear, Affine)

### Individual Experiments

You can also run specific experiments:

```python
from cross_model.experiments.procrustes import run_procrustes_experiment
from cross_model.experiments.activation_communication import run_activation_communication_experiment
from cross_model.experiments.adapters import run_lora_adapter_experiment

# Run Procrustes baseline
results = run_procrustes_experiment(
    model_a_id="meta-llama/Llama-3.1-8B",
    model_b_id="mistralai/Mistral-7B-v0.3"
)

# Run activation communication
results = run_activation_communication_experiment(
    model_a_id="meta-llama/Llama-3.1-8B",
    model_b_id="meta-llama/Llama-3.2-3B"
)

# Run LoRA adapter training
results = run_lora_adapter_experiment(
    model_a_id="meta-llama/Llama-3.1-8B",
    model_b_id="meta-llama/Llama-3.2-3B",
    gpu_id=0
)
```

## Model Support

Currently tested with:
- Llama 3.1 8B (4096 dims, 32 layers)
- Llama 3.2 3B (3072 dims, 28 layers)

## Key Features

- **Automatic dimension handling**: Trains learned projections when model dimensions mismatch
- **DDP support**: Optimized for multi-GPU training with extended timeouts
- **Checkpoint management**: Automatic save/load of trained projections
- **Platform detection**: Automatic configuration for Mac (MPS) vs HPC (CUDA)

## References

Ramesh & Li, "Communicating Activations Between Language Model Agents", ICML 2025 (arXiv:2501.14082)
