# COCONUT Reproduction Plan - Using Official Code

**Approach**: Use official COCONUT implementation directly (no custom reimplementation)
**Goal**: Reproduce their results with GPT-2 first, then try Llama 3.1 8B

---

## Quick Start

### Step 1: Setup

```bash
cd experimental/learning/reproduce_coconut/coconut

# Install dependencies
pip install -r requirements.txt

# Data already downloaded (385K examples)
ls data/gsm_train.json  # Should exist
```

### Step 2: Train Stage 0 (CoT Baseline)

```bash
# Train GPT-2 on full CoT reasoning
# NOTE: Original uses 4 GPUs, we'll use 1 (MacBook MPS)
python run.py args/gsm_cot.yaml
```

**Expected**:
- Trains for 25 epochs
- Saves checkpoints to `YOUR_PATH_TO_SAVE_THE_MODEL`
- Target: ~40% validation accuracy

### Step 3: Train Stage 1-3 (COCONUT)

```bash
# Load Stage 0 checkpoint and train with continuous thoughts
# Need to update load_model_path in args/gsm_coconut.yaml first
python run.py args/gsm_coconut.yaml
```

### Step 4: Evaluate

```bash
# Evaluate best checkpoint
# Need to update load_model_path in args/gsm_coconut_eval.yaml first
python run.py args/gsm_coconut_eval.yaml
```

---

## Configuration Changes Needed

### 1. Update `args/gsm_cot.yaml`

```yaml
# Change:
save_path: YOUR_PATH_TO_SAVE_THE_MODEL
# To:
save_path: ../runs/stage0

# Optionally reduce for MacBook (if GPU memory issues):
batch_size_training: 8  # Instead of 32
num_epochs: 6           # Instead of 25 (faster iteration)
```

### 2. Update `args/gsm_coconut.yaml`

```yaml
# Change:
save_path: YOUR_PATH_TO_SAVE_THE_MODEL
load_model_path: YOUR_PATH_TO_COT_CHECKPOINT
# To:
save_path: ../runs/stage1
load_model_path: ../runs/stage0/epoch_X  # Use best epoch from Stage 0

# Optionally reduce:
batch_size_training: 8
```

### 3. Update `args/gsm_coconut_eval.yaml`

```yaml
# Change:
load_model_path: YOUR_PATH_TO_COCONUT_CHECKPOINT
# To:
load_model_path: ../runs/stage1/epoch_X  # Use best epoch from Stage 1
```

---

## What Their Code Does

### Stage 0 (CoT Baseline) - `args/gsm_cot.yaml`

**Trains GPT-2 on**:
```
Out of 600 employees...
<<600*30/100=180>>
<<600*10/100=60>>
<<180+60=240>>
<<600-240=360>>
### 360
```

**Config**:
- Model: GPT-2 (124M params)
- Data: 385K Internalize CoT examples
- Epochs: 25
- Batch: 32 × 4 GPUs = 128
- LR: 1e-4
- c_thought: 0 (no latent)

### Stage 1-3 (COCONUT) - `args/gsm_coconut.yaml`

**Stage 1 trains on**:
```
Out of 600 employees...
<|start-latent|><|latent|><|latent|><|end-latent|>
<<600*10/100=60>>
<<180+60=240>>
<<600-240=360>>
### 360
```

**Config**:
- Model: Load Stage 0 checkpoint
- Stages: 1, 2, 3 (progressive replacement)
- Epochs per stage: 3
- c_thought: 2 (two latent tokens per step)
- Resume: 3 (starts from stage 1)
- Reset optimizer: True

### Evaluation - `args/gsm_coconut_eval.yaml`

**Evaluates** on test set with continuous thoughts

---

## File Structure

```
experimental/learning/reproduce_coconut/
├── PLAN.md                    # This file (simplified)
├── README.md                  # Quick reference
├── COCONUT_ANALYSIS.md        # Our analysis (for reference)
│
├── coconut/                   # Official repo
│   ├── run.py                 # Main training/eval script
│   ├── coconut.py             # COCONUT model
│   ├── dataset.py             # Data loading
│   ├── args/                  # Config files
│   │   ├── gsm_cot.yaml       # Stage 0 config
│   │   ├── gsm_coconut.yaml   # Stage 1-3 config
│   │   └── gsm_coconut_eval.yaml  # Eval config
│   └── data/
│       ├── gsm_train.json     # 385K examples (already downloaded)
│       ├── gsm_valid.json     # Validation set
│       └── gsm_test.json      # Test set
│
└── runs/                      # Our training outputs
    ├── stage0/                # Stage 0 checkpoints
    └── stage1/                # Stage 1-3 checkpoints
```

---

## Expected Timeline (MacBook MPS, GPT-2)

**Stage 0 (25 epochs, 385K examples)**:
- Time: ~10-12 hours (rough estimate)
- Output: Checkpoint with ~40% accuracy

**Stage 1-3 (3 epochs each, 385K examples)**:
- Time: ~1-2 hours per stage = ~3-6 hours total
- Output: Checkpoint with improved accuracy

**Total**: ~13-18 hours for full reproduction

**Fast iteration option**:
- Stage 0: 6 epochs instead of 25 (~3 hours)
- Stage 1: Skip stages 2-3 (~1 hour)
- Total: ~4 hours

---

## Success Criteria

**Stage 0**:
- ✅ Loss decreases
- ✅ Validation accuracy > 10%
- 🎯 Validation accuracy ~40% (paper's target)

**Stage 1-3**:
- ✅ Training completes
- ✅ Uses continuous thoughts (latent tokens)
- 🎯 Accuracy ≥ Stage 0 (showing continuous thoughts work)

---

## Troubleshooting

### Issue: "wandb login required"
```bash
# Login to wandb (for logging)
wandb login
# Or set debug=True in config to skip wandb
```

### Issue: GPU memory error
```yaml
# Reduce batch size in config
batch_size_training: 4  # or 2
gradient_accumulation_steps: 4  # to maintain effective batch size
```

### Issue: MPS not supported
```bash
# Their code may require CUDA
# Fallback: Use CPU or run on HPC
```

---

## After Reproduction

### If it works with GPT-2:
1. Try with Llama 3.1 8B:
   ```yaml
   model_id: meta-llama/Llama-3.1-8B
   ```
2. Compare results
3. Analyze continuous thoughts

### If it doesn't work:
1. Check logs
2. Compare with official setup (4 GPUs vs 1)
3. Try smaller subset first (10K examples)

---

## References

- **Official Code**: https://github.com/facebookresearch/coconut
- **Paper**: https://arxiv.org/abs/2412.06769
- **Data**: Already in `coconut/data/`
