# Evaluation Scripts Documentation

This directory contains standalone evaluation scripts copied from the LatentWire project.

## Directory Structure

```
finalization/
├── latentwire/           # Core LatentWire modules
│   ├── eval.py          # Main evaluation script
│   ├── eval_sst2.py     # SST-2 sentiment evaluation
│   ├── eval_agnews.py   # AG News classification evaluation
│   ├── gsm8k_eval.py    # GSM8K math evaluation
│   ├── models.py        # Model definitions
│   ├── core_utils.py    # Utility functions
│   ├── data.py          # Data loading utilities
│   └── features/        # Feature modules
├── telepathy/           # Telepathy evaluation suite
│   ├── eval_telepathy.py         # Main telepathy evaluation
│   ├── eval_telepathy_sst2.py    # SST-2 telepathy evaluation
│   ├── eval_telepathy_agnews.py  # AG News telepathy evaluation
│   ├── eval_telepathy_gsm8k.py   # GSM8K telepathy evaluation
│   └── eval_telepathy_trec.py    # TREC telepathy evaluation
├── scripts/             # Utility and analysis scripts
│   ├── baselines/       # Baseline evaluation scripts
│   └── statistical_testing.py    # Statistical analysis
└── test_eval_standalone.py       # Test script to verify imports

```

## Main Evaluation Scripts

### 1. Core LatentWire Evaluation (`latentwire/eval.py`)

The main evaluation script for LatentWire models. Supports multiple evaluation modes:

**Basic Usage:**
```bash
python latentwire/eval.py \
  --ckpt path/to/checkpoint \
  --samples 200 \
  --dataset squad \
  --max_new_tokens 12
```

**Key Arguments:**
- `--ckpt`: Path to checkpoint directory
- `--samples`: Number of evaluation samples
- `--dataset`: Dataset to use (squad, hotpotqa, etc.)
- `--max_new_tokens`: Maximum tokens to generate
- `--calibration`: Calibration method (embed_rms, etc.)
- `--latent_anchor_text`: Anchor text (e.g., "Answer: ")
- `--sequential_eval`: Evaluate models sequentially (memory efficient)

### 2. SST-2 Sentiment Evaluation (`latentwire/eval_sst2.py`)

Evaluates models on SST-2 sentiment classification task.

**Usage:**
```bash
python latentwire/eval_sst2.py \
  --checkpoint path/to/checkpoint \
  --num_samples 872 \
  --output_dir results/sst2
```

### 3. AG News Classification (`latentwire/eval_agnews.py`)

Evaluates models on AG News topic classification.

**Usage:**
```bash
python latentwire/eval_agnews.py \
  --checkpoint path/to/checkpoint \
  --num_samples 1000 \
  --output_dir results/agnews
```

### 4. GSM8K Math Evaluation (`latentwire/gsm8k_eval.py`)

Evaluates models on GSM8K grade school math problems.

**Usage:**
```bash
python latentwire/gsm8k_eval.py \
  --checkpoint path/to/checkpoint \
  --num_samples 100 \
  --output_dir results/gsm8k
```

## Telepathy Evaluation Scripts

The telepathy suite provides enhanced evaluation capabilities with cross-model communication.

### Main Telepathy Evaluation (`telepathy/eval_telepathy.py`)

**Usage:**
```bash
python telepathy/eval_telepathy.py \
  --checkpoint path/to/checkpoint \
  --dataset sst2 \
  --num_samples 100 \
  --output_dir results/telepathy
```

### Dataset-Specific Telepathy Scripts

- **SST-2**: `telepathy/eval_telepathy_sst2.py`
- **AG News**: `telepathy/eval_telepathy_agnews.py`
- **GSM8K**: `telepathy/eval_telepathy_gsm8k.py`
- **TREC**: `telepathy/eval_telepathy_trec.py`

## Running Tests

To verify that all evaluation scripts are properly installed:

```bash
python test_eval_standalone.py
```

This will test:
1. Module imports for all evaluation scripts
2. Basic functionality of key functions
3. Dependency resolution

## Common Environment Variables

```bash
# Required for execution
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1  # For Mac MPS

# Optional for specific hardware
export CUDA_VISIBLE_DEVICES=0,1,2,3   # Select GPUs
```

## Output Formats

All evaluation scripts produce:
- **JSON results**: Structured metrics in `{output_dir}/results.json`
- **JSONL logs**: Per-sample predictions in `{output_dir}/eval.jsonl`
- **Console logs**: Detailed progress and metrics

## Statistical Analysis

Use the statistical testing script to analyze results:

```bash
python scripts/statistical_testing.py \
  --results_dir results/ \
  --output statistical_report.txt
```

## Notes

- These scripts require PyTorch and Transformers libraries
- Checkpoint files should contain trained encoder/adapter weights
- Default seed (12345) ensures reproducible evaluation
- Use `--sequential_eval` for memory-efficient evaluation on limited hardware