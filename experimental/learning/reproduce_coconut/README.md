# COCONUT Reproduction - Quick Start

Using official COCONUT code to reproduce their results.

**Based on**: [facebookresearch/coconut](https://github.com/facebookresearch/coconut) (commit: latest as of Oct 2024)

**Modifications**:
- Modified `coconut/run.py` to support gloo backend for Mac/MPS compatibility
- Created `coconut/args/gsm_cot_test.yaml` for 3-epoch test runs
- Created `run_stage0_test.sh` wrapper script with logging

---

## Running on HPC (Recommended)

**Setup is complete!** Ready to train on HPC with CUDA GPUs.

```bash
# On HPC with CUDA GPUs
cd experimental/learning/reproduce_coconut/coconut

# Install dependencies
pip install -r requirements.txt

# Run 3-epoch test (single GPU)
torchrun --nproc_per_node=1 run.py args/gsm_cot_test.yaml

# Or run full 25-epoch training (4 GPUs)
torchrun --nproc_per_node=4 run.py args/gsm_cot.yaml
```

**Expected time**: ~1-2 hours for 3 epochs on H100.

---

## Running Locally (Mac - Not Recommended)

⚠️ **FSDP requires CUDA** - training on Mac MPS/CPU will be extremely slow or fail.

### 1. Install Dependencies

```bash
cd experimental/learning/reproduce_coconut/coconut
pip install -r requirements.txt
```

### 2. Check Data

```bash
# Should show 385620 examples
python -c "import json; data = json.load(open('data/gsm_train.json')); print(f'{len(data)} examples')"
```

### 3. (Optional) Login to WandB

```bash
# For experiment tracking
wandb login

# OR set debug=True in config to skip
```

### 4. Run Quick Test (3 epochs)

```bash
# Test config: 3 epochs instead of 25
python run.py args/gsm_cot_test.yaml
```

**This will**:
- Train GPT-2 on calculator-style reasoning
- Run for 3 epochs (~1-2 hours)
- Save checkpoint to `../runs/stage0/`
- Validate every epoch

### 5. Monitor Training

Watch the output for:
- Loss decreasing
- Validation accuracy increasing
- Checkpoint saves

---

## If Quick Test Works

### Run Full Training (25 epochs)

```bash
# Edit args/gsm_cot.yaml:
# - Change save_path to ../runs/stage0_full
# - Optionally reduce batch_size_training if memory issues

python run.py args/gsm_cot.yaml
```

### Then Train COCONUT (Stage 1-3)

```bash
# Edit args/gsm_coconut.yaml:
# - Change save_path to ../runs/stage1
# - Change load_model_path to ../runs/stage0/epoch_X

python run.py args/gsm_coconut.yaml
```

---

## Troubleshooting

### "wandb login required"
```bash
wandb login
# Or set debug: True in config
```

### GPU Memory Error
Edit config file:
```yaml
batch_size_training: 4
gradient_accumulation_steps: 8
```

### Import Errors
```bash
pip install --upgrade torch transformers datasets wandb
```

---

## What to Expect

**Stage 0 (3 epochs test)**:
- Time: ~1-2 hours
- Validation accuracy: ~10-20% (lower than full 25 epochs)
- Purpose: Verify setup works

**Stage 0 (25 epochs full)**:
- Time: ~10-12 hours
- Validation accuracy: ~40% (paper's target)
- Purpose: Strong CoT baseline

**Stage 1-3 (COCONUT)**:
- Time: ~3-6 hours
- Validation accuracy: ≥ Stage 0
- Purpose: Show continuous thoughts work

---

## Current Status

- ✅ Data downloaded (385K examples)
- ✅ Config created (gsm_cot_test.yaml)
- ⏳ Ready to train

See `PLAN.md` for full details.
