# Cross-Model Activation Communication

Refactored implementation of "Communicating Activations Between Language Model Agents" (Ramesh & Li, ICML 2025).

## Structure

```
cross_model/
├── __init__.py                 # Package initialization
├── README.md                   # This file
├── models.py                   # LearnedProjection model architecture
├── training.py                 # Projection training on C4 dataset
├── checkpointing.py            # Save/load utilities for projections
├── utils.py                    # DDP setup, device detection, helpers
├── experiments/                # Individual experiment modules
│   └── __init__.py
└── run_experiments.py          # Main entry point
```

## Components

### Core Modules

- **models.py**: Contains `LearnedProjection` class - a learned linear projection to handle dimension mismatches between models (e.g., Llama 3.2 3B's 3072 dims → Llama 3.1 8B's 4096 dims)

- **training.py**: Contains `train_learned_projection()` - trains projection matrices on C4 dataset following Ramesh & Li (2025) methodology

- **checkpointing.py**: Utilities for saving/loading trained projections and experimental results

- **utils.py**: Platform detection, DDP initialization, distributed training helpers

### Experiments

The experiments directory will contain individual experiment modules:

- **text_generation.py** (planned): Text similarity comparison across layers
- **task_evaluation_squad.py** (planned): SQuAD Q&A evaluation
- **task_evaluation_gsm8k.py** (planned): GSM8K math reasoning evaluation

## Usage

```bash
# Run all experiments
python cross_model/run_experiments.py

# With DDP on 4 GPUs
torchrun --nproc_per_node=4 cross_model/run_experiments.py
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
