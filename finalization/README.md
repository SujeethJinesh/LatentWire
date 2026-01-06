# LatentWire: Cross-Model Compression via Learned Interlingua

A self-contained implementation of the LatentWire training module for learning compressed representations that condition multiple heterogeneous LLMs without retokenization.

## Quick Start

```bash
# Install dependencies
pip3 install -r requirements.txt

# Set environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1  # For Mac MPS

# Run experiment
bash RUN_ALL.sh experiment --phase 1 --dataset sst2

# Or direct training
python3 latentwire/train.py --samples 1000 --epochs 1 --output_dir runs/test
```

## System Architecture

LatentWire implements a **continuous interlingua** - a learned compressed representation that conditions frozen LLMs:

- **Frozen LLMs**: Base models (Llama, Qwen) remain completely frozen
- **Shared Latent Space**: Single encoder produces `Z ∈ R^{M × d_z}` soft tokens
- **Model-Specific Adapters**: Linear adapters map shared latent to model embeddings

## Directory Structure

```
finalization/
├── RUN_ALL.sh              # Main orchestration script
├── latentwire/             # Core training module
│   ├── train.py            # Main training loop with K-token objectives
│   ├── eval.py             # Evaluation with baselines
│   ├── models.py           # Encoder, Adapter, LMWrapper
│   ├── losses.py           # K-token CE and KD losses
│   ├── data.py             # Dataset loading
│   └── checkpointing.py    # Resume capabilities
├── scripts/                # Utility scripts
├── features/               # Feature experiments
├── config.yaml            # Master configuration
└── runs/                  # Experiment outputs
```

## Core Components

### Models (models.py)
- `InterlinguaEncoder`: Encodes text into shared latent space
- `Adapter`: Maps shared latent to model-specific embeddings
- `LMWrapper`: Wraps frozen language models
- `ByteTokenizer`: Byte-level tokenization
- `LinearProbeBaseline`: Reviewer comparison baseline

### Training (train.py)
- K-token teacher-forced cross-entropy
- Knowledge distillation from text-prompted teacher
- Per-example calibration for latent scaling
- Checkpoint resume support
- DDP distributed training

### Losses (losses.py)
- `k_token_ce_from_prefix`: Supervises first K tokens
- `kd_first_k_prefix_vs_text`: Distills teacher distributions
- Proper PAD token masking and BOS alignment

### Evaluation (eval.py)
- Text baseline (upper bound)
- Latent compression
- Token-budget baseline
- LLMLingua baseline
- Linear probe baseline
- Statistical significance testing

## Configuration

Key hyperparameters from `config.yaml`:

```yaml
# Architecture
latent_len: 32       # M: number of soft tokens
d_z: 256            # Latent dimension
encoder_type: byte  # byte, simple-st, mlp, transformer

# Training - Optimized for H100 (80GB)
batch_size: 64      # Uses 81.3% GPU memory
learning_rate: 1.0e-4
epochs: 3
k_token_supervision: 4
first_token_ce_weight: 0.5

# Calibration & Anchoring
calibration: embed_rms
warm_anchor_text: "Answer: "
append_bos_after_prefix: true

# Models
llama_id: "meta-llama/Meta-Llama-3.1-8B-Instruct"
qwen_id: "Qwen/Qwen2.5-7B-Instruct"
```

## Supported Datasets

- **Classification**: SST2, AG News, TREC
- **QA**: SQuAD v1/v2, HotpotQA
- **Math**: GSM8K (with CoT)
- **Summarization**: XSum (optional)

## Experiment Phases

The `RUN_ALL.sh` script orchestrates 4 experimental phases:

### Phase 1: Statistical Rigor
- Multiple random seeds
- Bootstrap confidence intervals
- Significance testing

### Phase 2: Linear Probe Baseline
- Layer-wise probing
- Fair comparison to learned representations

### Phase 3: Compression Baselines
- LLMLingua comparison
- Token-budget baseline
- Selective context

### Phase 4: Efficiency Benchmarks
- Latency measurements
- Memory profiling
- Throughput analysis

## Running Experiments

### Main Orchestration Script
```bash
# Test system (dry run)
bash RUN_ALL.sh test --dry-run

# Run full experiment
bash RUN_ALL.sh experiment

# Specific phase/dataset
bash RUN_ALL.sh experiment --phase 1 --dataset sst2

# Submit to SLURM cluster
bash RUN_ALL.sh slurm experiment

# Monitor running experiments
bash RUN_ALL.sh monitor

# Generate final results
bash RUN_ALL.sh finalize
```

### Direct Python Usage
```bash
# Training
python3 latentwire/train.py \
  --samples 87599 \
  --epochs 24 \
  --batch_size 64 \
  --latent_len 32 \
  --d_z 256 \
  --encoder_type byte \
  --dataset squad \
  --sequential_models \
  --warm_anchor_text "Answer: " \
  --first_token_ce_weight 0.5

# Evaluation
python3 latentwire/eval.py \
  --ckpt runs/experiment/epoch_1 \
  --samples 200 \
  --max_new_tokens 128 \
  --dataset squad \
  --fresh_eval \
  --calibration embed_rms
```

## SLURM/HPC Configuration

For cluster execution:

```bash
#!/bin/bash
#SBATCH --job-name=latentwire
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=12:00:00
#SBATCH --mem=256GB

cd /projects/m000066/sujinesh/LatentWire
export PYTHONPATH=.
bash RUN_ALL.sh experiment
```

## Memory Requirements

### H100 (80GB) Budget
- Frozen Models: 31.3GB (Llama + Qwen in bf16)
- Trainable: 0.2GB (encoder + adapters)
- Optimizer: 1.2GB (AdamW states)
- Overhead: 6.4GB
- **Available for Activations: 40.7GB**
- **Optimal batch_size: 64** (81.3% utilization)

### Multi-GPU Scaling
- 2x H100: batch_size=64 per GPU, effective=128
- 4x H100: batch_size=64 per GPU, effective=256

## Key Features

### Checkpoint Recovery
- Automatic resume from interruption
- State file tracking (`recovery_state.json`)
- Preemption-aware saving

### Statistical Testing
- Bootstrap confidence intervals
- Paired t-tests
- Effect size computation
- Multiple comparison correction

### Baseline Comparisons
- Text prompt (upper bound)
- Token-budget (fair comparison)
- LLMLingua (SOTA compression)
- Linear probe (representation quality)

## Results & Metrics

### Task Metrics
- **Classification**: Accuracy, F1, Precision, Recall
- **QA**: Exact Match, F1, Has-Answer metrics
- **Generation**: BLEU, ROUGE, BERTScore

### Compression Metrics
- Compression ratio
- Wire bytes saved
- Bits per byte

### Efficiency Metrics
- Latency (ms/sample)
- Throughput (samples/sec)
- Memory usage (GB)

## Critical Implementation Details

### Bug Fixes Applied
- PAD token masking in labels (-100)
- BOS policy alignment between train/eval
- Per-example calibration (not batch-level)
- Tokenization t=0 alignment verification
- Left padding handling

### K-Token Objectives
- Supervise first K tokens (default K=4)
- First-token CE weight (default 0.5)
- Knowledge distillation with temperature τ=1.0

### Decode Hardening
- First-token nucleus sampling (p=0.95)
- First-token temperature (0.7)
- EOS ban for first 4 steps

## Development Workflow

### Local Development (MacBook)
- Code editing and analysis
- Review logs from HPC
- Documentation

### Remote Execution (HPC)
- Training runs with 4× H100 GPUs
- Jobs submitted via SLURM
- Logs pushed back via git

### Standard Workflow
```bash
# Local: Develop and push
git add -A && git commit -m "Update" && git push

# HPC: Pull and run
git pull && rm -rf runs && PYTHONPATH=. bash scripts/run_pipeline.sh

# Local: Analyze results
git pull && python scripts/analyze_results.py
```

## File Manifest

### Core Implementation (7 files)
- `train.py`: Main training loop
- `eval.py`: Evaluation pipeline
- `models.py`: Model architectures
- `losses.py`: Loss functions
- `data.py`: Dataset loading
- `prefix_utils.py`: Utilities
- `linear_probe_baseline.py`: Baselines

### Experiments (4 files)
- `unified_cross_model_experiments.py`: Cross-model tests
- `run_pipeline.sh`: Pipeline script
- `statistical_testing.py`: Significance testing
- `config.yaml`: Configuration

### Total: 48 Python modules providing complete infrastructure

## System Status

**Status: READY** ✓

All components verified:
- RUN_ALL.sh v3.1.0 executable
- 48 Python modules present
- latentwire/ subdirectory complete
- Checkpoint recovery functional
- SLURM support configured
- Error recovery mechanisms in place

## Requirements

- Python 3.7+ (for f-strings)
- CUDA GPU (H100/A100 recommended)
- 40GB+ GPU memory for dual-model training
- PyTorch 2.0+ with CUDA support

## License & Citation

Research code for the LatentWire paper. If using this code, please cite:
```
@article{latentwire2026,
  title={LatentWire: Cross-Model Compression via Learned Interlingua},
  author={...},
  year={2026}
}
```

---

**Quick Commands Reference:**
- Test: `bash RUN_ALL.sh test --dry-run`
- Train: `bash RUN_ALL.sh train --dataset squad`
- Evaluate: `bash RUN_ALL.sh eval --checkpoint runs/exp/epoch_1`
- Full experiment: `bash RUN_ALL.sh experiment`
- Monitor: `bash RUN_ALL.sh monitor`
- Finalize: `bash RUN_ALL.sh finalize`

For issues or questions, check logs in `runs/` directory.