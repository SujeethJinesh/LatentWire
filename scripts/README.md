# LatentWire Experiment Scripts

Clean, essential experiments for building a compressed interlingua.

## Quick Start

Standard workflow (end-to-end from scratch):
```bash
git pull && rm -rf runs && PYTHONPATH=. bash scripts/<experiment>.sh
```

All scripts:
- Work end-to-end (no dependencies on previous runs)
- Use tee logging (timestamped logs in output directories)
- Optimize for 4x H100 GPUs
- Use real data only (no synthetic tests)

## Baseline Experiments

These establish comparison points for the compressed interlingua.

### 1. Text Baseline (Upper Bound)
**What**: Full text prompts to both LLMs
**Why**: Establishes best possible performance
**Target**: LatentWire should approach this quality

```bash
PYTHONPATH=. bash scripts/baselines/text_baseline.sh
```

**Output**: `runs/baselines/text/`
- F1/EM scores for Llama and Qwen
- This is the quality ceiling we're aiming for

### 2. Token Budget Baseline (Fair Comparison)
**What**: Text truncated to M tokens (same budget as latent)
**Why**: Fair comparison - does learned compression beat simple truncation?
**Critical**: If LatentWire doesn't beat this, learning isn't working

```bash
LATENT_LEN=32 PYTHONPATH=. bash scripts/baselines/token_budget_baseline.sh
```

**Output**: `runs/baselines/token_budget/`
- F1/EM with truncated text
- **Must beat this to claim compression is learned**

### 3. PCA Baseline (Linear Compression)
**What**: PCA projection of text embeddings
**Why**: Tests if linear compression is sufficient or if we need learned encoder

```bash
LATENT_LEN=32 PYTHONPATH=. bash scripts/baselines/pca_baseline.sh
```

**Output**: `runs/baselines/pca/M32/`
- PCA explained variance
- F1/EM with linear compression
- If PCA ≈ Text → Linear is enough
- If PCA << Text → Need learned encoder

## Main Experiments

### 1. LatentWire Training
**What**: Train encoder + adapters for compressed interlingua
**Architecture**: `Text → Encoder → Z (M × d_z) → Adapter → LLM`

```bash
PYTHONPATH=. bash scripts/experiments/train_latentwire.sh
```

**Configuration** (optimized for 4x H100):
- Batch size: 32 (increased from 8 - was only 20% GPU usage)
- Samples: 87,599 (full SQuAD training set)
- Epochs: 3
- Latent: M=32, d_z=256
- K-token teacher forcing: K=4
- First token CE weight: 0.5

**Output**: `runs/experiments/latentwire/`
- Checkpoints (encoder.pt, adapter_llama.pt, adapter_qwen.pt)
- Training logs with first token accuracy curves
- Best checkpoint saved to checkpoint_best/

**Evaluation**:
```bash
python latentwire/eval.py \
    --ckpt runs/experiments/latentwire/checkpoint_best \
    --samples 1000 \
    --fresh_eval
```

### 2. Hyperparameter Sweep
**What**: Search over key hyperparameters
**Grid**:
- Latent length M ∈ {32, 48, 64}
- Latent dim d_z ∈ {256, 512}
- First token CE weight ∈ {0.5, 1.0}
- K-tokens ∈ {4, 8}

```bash
PYTHONPATH=. bash scripts/experiments/sweep_hyperparams.sh
```

**Output**: `runs/sweeps/hyperparam/`
- One directory per configuration
- sweep_summary.json (best configurations)
- sweep_results.csv (full results table)

**Estimated time**: ~6-8 hours on 4x H100 (24 configurations × ~20min each)

## Diagnostics

### Embedding Diagnostics
**What**: Analyze learned vs text embeddings to understand failure modes
**Answers**:
1. Where did "115-120× magnitude mismatch" come from?
2. What's different between text and learned embeddings?
3. Why does RMS scaling destroy everything?

```bash
PYTHONPATH=. bash scripts/run_embedding_diagnostics.sh
```

**How it works**:
1. Trains quick checkpoint (1 epoch, 1000 samples)
2. Analyzes learned embeddings from that checkpoint
3. Compares to text embeddings

**Output**: `runs/embed_diagnostics/`
- Per-token RMS distributions
- Nearest vocab alignment
- Effect of RMS scaling and batch normalization
- Compression ratio analysis

## Expected Results Hierarchy

Based on task difficulty and method sophistication:

```
Text Baseline (F1 ~0.80-0.85)          [UPPER BOUND]
    ↓
Token Budget M=64 (F1 ~0.60-0.70)      [More info retained]
    ↓
LatentWire M=32 (F1 target: 0.40-0.60) [LEARNED COMPRESSION]
    ↓
Token Budget M=32 (F1 ~0.30-0.40)      [Simple truncation]
    ↓
PCA M=32 (F1 ~0.20-0.30)               [Linear compression]
```

**Critical tests**:
1. **LatentWire > Token Budget (same M)**: Proves learned compression works
2. **LatentWire > PCA**: Proves non-linear encoding is necessary
3. **LatentWire approaches Text**: Shows minimal information loss

## GPU Optimization Notes

**Current status** (from diagnostics):
- GPU usage: 20-29% (lots of headroom!)
- Batch size: 8 → Can increase to 32+
- Peak memory: 71GB / 340GB total (4x H100s)

**Optimizations applied**:
1. Increased batch size from 8 → 32
2. Model parallelism: Llama on GPUs 0-3, Qwen on GPUs 0-3 (shared)
3. Sequential loading (models loaded one after another)

**Further optimizations possible**:
- Batch size 48-64 (still <50% memory)
- Data parallel training across GPUs
- Mixed precision training (already using fp16)

## Common Issues

### 1. "unrecognized arguments: --output_dir"
**Fix**: train.py uses `--save_dir` not `--output_dir`

### 2. "UnboundLocalError: local variable 'json' referenced"
**Fix**: Ensure `import json` at top of file (not inside functions)

### 3. Low GPU utilization (20%)
**Fix**: Increase batch size (set BATCH_SIZE=32 or higher)

### 4. "Checkpoint not found"
**Fix**: These scripts work end-to-end - they train first, then analyze

## Archived Scripts

Old experimental scripts moved to `scripts/archive/`:
- Various sweep scripts (replaced by sweep_hyperparams.sh)
- Old diagnostic scripts (replaced by embedding_diagnostics)
- Ad-hoc test scripts

These are kept for historical reference but should not be used.

## Workflow Example

Complete workflow to train and evaluate compressed interlingua:

```bash
# 1. Establish baselines
PYTHONPATH=. bash scripts/baselines/text_baseline.sh         # Upper bound
PYTHONPATH=. bash scripts/baselines/token_budget_baseline.sh # Fair comparison
PYTHONPATH=. bash scripts/baselines/pca_baseline.sh          # Linear baseline

# 2. Train LatentWire
PYTHONPATH=. bash scripts/experiments/train_latentwire.sh

# 3. Evaluate best checkpoint
python latentwire/eval.py \
    --ckpt runs/experiments/latentwire/checkpoint_best \
    --samples 10000 \
    --fresh_eval

# 4. (Optional) Run hyperparameter sweep
PYTHONPATH=. bash scripts/experiments/sweep_hyperparams.sh

# 5. (Optional) Diagnose embedding issues
PYTHONPATH=. bash scripts/run_embedding_diagnostics.sh
```

## Results Interpretation

**Good signs**:
- LatentWire F1 > Token Budget F1 (at same M)
- First token accuracy > 20%
- Training loss decreasing
- Predictions not collapsed to single token

**Bad signs**:
- LatentWire F1 < Token Budget F1 → Learning not working
- First token accuracy < 10% → Mode collapse
- All predictions = "the" → Severe optimization issue
- RMS magnitude 100× off → Calibration broken

## Contact

Questions? Check:
- CLAUDE.md (development guidelines)
- PLAN.md (research roadmap)
- LOG.md (experiment history)
