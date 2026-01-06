# SUBMISSION CHECKLIST

## Pre-Submission Verification for HPC Experiments

**Last Updated**: January 2026
**Purpose**: Final verification before submitting jobs to Marlowe HPC cluster (4× H100 GPUs)

---

## 1. Code Completeness

### What to Check
- All required evaluation scripts exist and are executable
- Import statements are correct
- No placeholder TODOs in critical paths
- Error handling is implemented

### How to Verify
```bash
# Check all evaluation scripts exist
ls -la latentwire/eval_*.py telepathy/eval_*.py

# Test imports locally
python -c "from latentwire import eval_agnews, eval_sst2, gsm8k_eval"
python -c "from telepathy import eval_telepathy_trec"

# Search for unfinished TODOs
grep -r "TODO" latentwire/eval*.py telepathy/eval*.py --include="*.py" | grep -v "# TODO: Future"
```

### Expected Outcome
- All 4 evaluation scripts present
- No import errors
- No critical TODOs remaining

---

## 2. Dependencies Installed

### What to Check
- All required packages in requirements.txt
- CUDA/PyTorch versions compatible with H100s
- Transformers library up to date

### How to Verify
```bash
# Check requirements file
cat requirements.txt | grep -E "(torch|transformers|datasets|scikit-learn)"

# Verify on HPC (after SSH)
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Expected Outcome
- PyTorch >= 2.0.0 with CUDA support
- Transformers >= 4.30.0
- All imports successful

---

## 3. Memory Calculations Correct

### What to Check
- Batch sizes fit in H100 memory (80GB per GPU)
- Model loading distributed properly
- Gradient accumulation steps configured

### How to Verify
```bash
# Check batch size configurations
grep -r "batch_size" telepathy/*.py | grep -E "=[0-9]+"

# Estimate memory usage (rough calculation)
python -c "
# Llama 3.1 8B parameters
model_params = 8e9
bytes_per_param = 2  # fp16
model_memory_gb = (model_params * bytes_per_param) / 1e9
print(f'Model memory: {model_memory_gb:.1f} GB')
print(f'With batch_size=16, seq_len=512: ~{model_memory_gb * 1.5:.1f} GB')
"
```

### Expected Outcome
- Model memory < 40GB (leaving room for activations)
- Batch sizes: 8-32 for full models
- No OOM warnings in calculations

---

## 4. Batch Sizes Optimized

### What to Check
- Batch sizes maximize GPU utilization
- Different for training vs evaluation
- Consistent across all evaluation scripts

### How to Verify
```bash
# Check all batch size settings
find telepathy -name "*.py" -exec grep -H "batch_size" {} \; | sort

# Verify evaluation batch sizes are consistent
grep "eval_batch_size\|batch_size" latentwire/eval*.py telepathy/eval*.py
```

### Expected Outcome
- Training batch_size: 16-32
- Evaluation batch_size: 32-64
- All scripts use same evaluation batch size

---

## 5. Seeds Set Properly

### What to Check
- Random seeds set for reproducibility
- Seeds consistent across scripts
- PyTorch, NumPy, and Python seeds all set

### How to Verify
```bash
# Check seed settings
grep -r "seed\|random" telepathy/*.py | grep -E "set_seed|random.seed|torch.manual_seed"

# Verify seed value
grep "SEED = \|seed = " telepathy/*.py
```

### Expected Outcome
- Seed value: 42 (or other consistent value)
- All three seeds set (random, numpy, torch)
- set_seed() called early in main()

---

## 6. Evaluation Metrics Standard

### What to Check
- Metrics match paper requirements
- Accuracy/F1 calculated correctly
- Results saved in standard format

### How to Verify
```bash
# Check metric implementations
grep -r "accuracy\|f1_score" latentwire/eval*.py telepathy/eval*.py

# Verify JSON output format
grep -r "json.dump\|to_json" telepathy/*.py
```

### Expected Outcome
- Uses sklearn.metrics for standard metrics
- Saves results as JSON with keys: accuracy, f1, latency
- Includes both per-class and overall metrics

---

## 7. Statistical Tests Ready

### What to Check
- Statistical testing script functional
- Bootstrap confidence intervals implemented
- Paired t-tests for model comparisons

### How to Verify
```bash
# Test statistical script locally with dummy data
python -c "
import numpy as np
from scripts.statistical_testing import run_statistical_tests

# Create dummy results
dummy_results = {
    'baseline': {'accuracy': list(np.random.rand(100) * 0.9)},
    'telepathy': {'accuracy': list(np.random.rand(100) * 0.85)}
}

# This should run without errors
print('Statistical tests ready')
"
```

### Expected Outcome
- No import errors
- Bootstrap CI calculated
- p-values < 0.05 for significant differences

---

## 8. SLURM Script Configured

### What to Check
- Account and partition correct
- GPU count specified
- Time limits reasonable
- Output paths correct

### How to Verify
```bash
# Check SLURM headers
head -n 20 telepathy/submit_*.slurm | grep SBATCH

# Verify critical settings
grep -E "account=marlowe-m000066|partition=preempt|gpus=4" telepathy/*.slurm

# Check paths use /projects not /home
grep "WORK_DIR\|output=" telepathy/*.slurm | grep -v "/projects" && echo "WARNING: Using wrong paths!"
```

### Expected Outcome
```
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --gpus=4
#SBATCH --time=12:00:00
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
```

---

## 9. Git Repository Updated

### What to Check
- All changes committed
- Repository pushed to remote
- No uncommitted files

### How to Verify
```bash
# Check git status
git status

# Verify no uncommitted changes
git diff --stat

# Check remote is set
git remote -v

# Ensure latest changes pushed
git log --oneline -5
```

### Expected Outcome
- "nothing to commit, working tree clean"
- Remote points to correct repository
- Latest commit has your changes

---

## 10. Results Validation Ready

### What to Check
- Results directories created
- Logging configured properly
- Analysis scripts ready

### How to Verify
```bash
# Check results directory structure
ls -la runs/ 2>/dev/null || echo "runs/ will be created on HPC"

# Verify logging in scripts
grep "tee.*LOG_FILE" telepathy/*.sh

# Test result loading locally
python -c "
import json
# This should work after experiments run
# json.load(open('runs/telepathy_results.json'))
print('Result loading code ready')
"
```

### Expected Outcome
- All scripts use tee for logging
- JSON result format verified
- Analysis scripts can parse output

---

## FINAL SUBMISSION COMMANDS

After verifying all items above:

```bash
# 1. Final local check
git status  # Should be clean
git pull    # Get latest

# 2. SSH to HPC
ssh marlowe  # or your alias

# 3. Navigate and update
cd /projects/m000066/sujinesh/LatentWire
git pull

# 4. Submit job
sbatch telepathy/submit_enhanced_arxiv.slurm

# 5. Monitor
squeue -u $USER
tail -f runs/telepathy_arxiv_*.log

# 6. After completion, locally:
git pull
python scripts/analyze_results.py
```

---

## COMMON ISSUES & FIXES

| Issue | Fix |
|-------|-----|
| OOM on GPU | Reduce batch_size or use gradient accumulation |
| Import errors | Check PYTHONPATH=. is set |
| SLURM fails immediately | Check account/partition settings |
| No results saved | Verify output_dir paths and permissions |
| Git push fails | Check remote permissions and credentials |

---

## EMERGENCY CONTACTS

- HPC Issues: Check Marlowe documentation or support ticket
- Code Issues: Review LOG.md for recent changes
- Statistical Questions: Refer to papers for standard practices

---

## SIGN-OFF

Before submitting, confirm:

- [ ] All 10 sections verified
- [ ] Test run completed locally (if possible)
- [ ] SLURM script reviewed by second pair of eyes
- [ ] Backup of current results saved
- [ ] Expected runtime calculated
- [ ] Ready to monitor job after submission

**Remember**: It's better to spend 10 minutes checking than 10 hours debugging a failed run!