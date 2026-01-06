# LatentWire Training Module (Self-Contained)

This directory contains a self-contained copy of the LatentWire training module with all its direct dependencies.

## Structure

```
finalization/
├── latentwire/           # Core training module and dependencies
│   ├── train.py         # Main training script
│   ├── models.py        # Model definitions (Encoder, Adapter, etc.)
│   ├── core_utils.py    # Core utilities for training
│   ├── losses.py        # Loss functions (k-token CE, KD losses)
│   ├── loss_bundles.py  # Loss bundling utilities
│   ├── data.py          # Dataset loading
│   ├── data_pipeline.py # Data pipeline setup
│   ├── checkpointing.py # Checkpoint saving/loading
│   ├── feature_registry.py # Feature management
│   └── features/        # Feature implementations
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Requirements

- Python 3.7 or higher (required for f-strings and modern Python features)
- CUDA-capable GPU (recommended) or Apple Silicon Mac with MPS support
- 16GB+ RAM for training
- 40GB+ GPU memory for full dual-model training

## Installation

```bash
# Ensure you're using Python 3
python3 --version  # Should be 3.7+

# Install dependencies
pip3 install -r requirements.txt
```

## Usage

### Training Example

```bash
# Set environment variables
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1  # For Mac MPS

# Quick test (small sample size)
python3 latentwire/train.py \
  --samples 100 \
  --epochs 1 \
  --batch_size 4 \
  --output_dir runs/test

# Full training
python3 latentwire/train.py \
  --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --qwen_id "Qwen/Qwen2.5-7B-Instruct" \
  --samples 87599 \
  --epochs 24 \
  --batch_size 64 \
  --latent_len 32 \
  --d_z 256 \
  --encoder_type byte \
  --dataset squad \
  --sequential_models \
  --warm_anchor_text "Answer: " \
  --first_token_ce_weight 0.5 \
  --output_dir runs/experiment_name
```

### Key Configuration Parameters

- `latent_len`: Number of soft tokens (compression vs capacity tradeoff)
- `d_z`: Latent dimension per token
- `first_token_ce_weight`: Weight for first-token cross-entropy loss
- `warm_anchor_text`: Anchor text between prefix and answer
- `sequential_models`: Process models sequentially (saves memory)

## Core Components

### Models (models.py)
- `InterlinguaEncoder`: Encodes text into shared latent space
- `Adapter`: Maps shared latent to model-specific embeddings
- `LMWrapper`: Wraps frozen language models
- `ByteTokenizer`: Byte-level tokenization

### Training (train.py)
- Implements k-token teacher-forced cross-entropy
- Knowledge distillation from text-prompted teacher
- Per-example calibration for latent scaling
- Proper tokenization alignment

### Losses (losses.py)
- `k_token_ce_from_prefix`: Supervises first K tokens
- `kd_first_k_prefix_vs_text`: Distills teacher distributions
- `kd_hidden_states_first_k`: Hidden state distillation

### Utilities (core_utils.py)
- Calibration functions
- BOS policy handling
- Anchor text management
- Tensor normalization

## Notes

- This is a self-contained copy extracted from the main LatentWire repository
- The training scripts are designed to work with frozen LLMs (Llama, Qwen)
- Implements a continuous interlingua for cross-model communication
- Checkpoints are saved to the specified output directory

## Files Included

This self-contained package includes all necessary components:
- Core training module (`train.py`) with ~5000 lines of training logic
- Model definitions (`models.py`) including encoders, adapters, and wrappers
- Loss functions (`losses.py`, `loss_bundles.py`) for k-token CE and KD
- Data loading (`data.py`, `data_pipeline.py`) for SQuAD and HotpotQA
- Utilities (`core_utils.py`) for calibration, tokenization, and anchoring
- Checkpointing system (`checkpointing.py`) for saving/resuming training
- Feature registry system for extensible architectures
- Evaluation scripts for testing trained models

Total: 30 Python files providing complete training infrastructure.
