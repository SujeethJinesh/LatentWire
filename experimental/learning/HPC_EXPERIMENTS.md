# Running Experiments on HPC - Resource Allocation Guide

This guide shows how to run both COCONUT and cross-model ablation experiments on HPC with optimal resource allocation.

## Resource Allocation Strategy

**Single Compute Node with 4 GPUs:**
- **GPU 0**: Cross-model ablation experiment
- **GPUs 1, 2, 3**: COCONUT Stage 0 training (3 GPUs)

Both experiments run **simultaneously on the same node** with isolated GPU assignments via `CUDA_VISIBLE_DEVICES`.

## Prerequisites

1. **Conda environment** (Python 3.11) - set up once:
```bash
module load conda/24.3.0-0
conda create -n 3_11 python=3.11 -y
conda activate 3_11
```

2. **Install dependencies** in both project directories:
```bash
# For COCONUT
cd experimental/learning/reproduce_coconut/coconut
pip install -r requirements.txt

# For cross-model ablation
cd experimental/learning
pip install transformers datasets accelerate torch
```

3. **Pull latest code**:
```bash
cd /path/to/LatentWire
git pull
```

---

## Experiment 1: COCONUT Stage 0 Training (3 GPUs)

### Location
```bash
cd experimental/learning/reproduce_coconut
```

### Quick Test (3 epochs, ~2-3 hours on 3 H100s)
```bash
conda activate 3_11
bash run_coconut_hpc.sh test

# Uses GPUs 1,2,3 (GPU 0 reserved for cross-model ablation)
# Logs saved to: runs/stage0_test/coconut_stage0_test_TIMESTAMP.log
```

### Full Training (25 epochs, ~15-20 hours on 3 H100s)
```bash
conda activate 3_11
bash run_coconut_hpc.sh full

# Uses GPUs 1,2,3 (GPU 0 reserved for cross-model ablation)
# Logs saved to: runs/stage0_full/coconut_stage0_full_TIMESTAMP.log
```

**Note**: The wrapper script sets `CUDA_VISIBLE_DEVICES=1,2,3` and defaults to `NPROC=3` for shared node execution.

### Monitor Training
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Tail logs
tail -f runs/stage0_test/coconut_*.log   # For test run
tail -f runs/stage0_full/coconut_*.log   # For full run
```

---

## Experiment 2: Cross-Model Ablation (1 GPU)

### Location
```bash
cd experimental/learning
```

### Run Experiment
```bash
# Activate environment
conda activate 3_11

# Run on GPU 0 (COC uses GPUs 1-3)
bash run_cross_model_ablation_hpc.sh
```

**What it does:**
- Loads Llama 3.1 8B and Mistral 7B
- Downloads 10,000 WikiText-2 calibration samples
- Tests 5 alignment methods:
  1. No alignment (baseline)
  2. Procrustes (SVD rotation)
  3. Centered Procrustes
  4. Scaled Procrustes
  5. L-Cross OLS
- Runs calibration for both directions:
  - Llama → Mistral
  - Mistral → Llama

**Expected runtime:** ~2-4 hours on single H100

### Monitor Progress
```bash
# Tail the log file
tail -f runs/cross_model_ablation/cross_model_ablation_hpc_*.log

# Check GPU usage
nvidia-smi
```

---

## Running Both Experiments Simultaneously on Same Node

Both scripts automatically isolate GPU usage via `CUDA_VISIBLE_DEVICES`, so they can run simultaneously on the same compute node without conflicts.

### Option 1: Separate Terminal Sessions (Recommended)

**Terminal 1** - COCONUT (GPUs 1,2,3):
```bash
cd /path/to/LatentWire/experimental/learning/reproduce_coconut
conda activate 3_11
bash run_coconut_hpc.sh full  # or 'test' for quick 3-epoch run
```

**Terminal 2** - Cross-Model (GPU 0):
```bash
cd /path/to/LatentWire/experimental/learning
conda activate 3_11
bash run_cross_model_ablation_hpc.sh
```

### Option 2: Background Jobs

```bash
# Navigate to base directory
cd /path/to/LatentWire/experimental/learning

# Start COCONUT in background (GPUs 1,2,3)
cd reproduce_coconut
nohup bash run_coconut_hpc.sh full &
COCONUT_PID=$!

# Start cross-model in background (GPU 0)
cd ..
nohup bash run_cross_model_ablation_hpc.sh &
ABLATION_PID=$!

# Monitor both experiments
tail -f reproduce_coconut/runs/stage0_full/coconut_*.log
tail -f runs/cross_model_ablation/cross_model_ablation_hpc_*.log

# Check both processes are running
ps -p $COCONUT_PID $ABLATION_PID

# Monitor GPU usage (should show all 4 GPUs in use)
watch -n 1 nvidia-smi
```

**GPU Isolation:**
- COCONUT: `CUDA_VISIBLE_DEVICES=1,2,3` (set by run_coconut_hpc.sh)
- Cross-model: `CUDA_VISIBLE_DEVICES=0` (set by run_cross_model_ablation_hpc.sh)
- No manual GPU assignment needed - handled automatically by scripts

---

## Expected Results

### COCONUT Stage 0
- **Validation accuracy**: ~40% at 25 epochs (paper target)
- **Checkpoints**: Saved to `experimental/learning/reproduce_coconut/runs/stage0/`
- **Purpose**: Establishes strong CoT baseline for comparison

### Cross-Model Ablation
- **Output**: Text generations for each alignment method
- **Comparison**: Baseline quality vs aligned cross-model transfer
- **Purpose**: Determine which alignment method enables best cross-model communication

---

## Troubleshooting

### COCONUT Issues
See `experimental/learning/reproduce_coconut/HPC_SETUP.md`

### Cross-Model Ablation Issues

**OOM (Out of Memory)**
```bash
# Models are large (8B + 7B), ensure node has enough GPU memory
# H100 (80GB) should be fine
# If issues, can reduce calibration samples in cross_model_ablation.py
```

**Slow generation**
- CUDA should be much faster than Mac MPS (~100x speedup expected)
- If slow: Check GPU is being used (`nvidia-smi`)

**Import errors**
```bash
pip install --upgrade transformers datasets accelerate torch
```

---

## Quick Reference

```bash
# Pull latest code
git pull

# Activate environment
module load conda/24.3.0-0
conda activate 3_11

# COCONUT (3 GPUs: 1,2,3)
cd experimental/learning/reproduce_coconut
bash run_coconut_hpc.sh full  # or 'test' for quick 3-epoch run

# Cross-model ablation (1 GPU: 0)
cd experimental/learning
bash run_cross_model_ablation_hpc.sh

# Both run on same node - GPU isolation handled automatically

# Monitor
nvidia-smi
tail -f reproduce_coconut/runs/stage0_full/coconut_*.log
tail -f runs/cross_model_ablation/cross_model_ablation_hpc_*.log
```

---

## After Experiments Complete

1. **Check results**:
```bash
# COCONUT: Check validation accuracy in logs
grep "validation" experimental/learning/reproduce_coconut/runs/stage0/*.log

# Cross-model: Review generations in log
cat experimental/learning/runs/cross_model_ablation/*.log
```

2. **Commit logs** (if desired):
```bash
git add experimental/learning/reproduce_coconut/runs/
git add experimental/learning/runs/
git commit -m "results: Add HPC experiment logs"
git push
```

3. **Pull locally for analysis**:
```bash
# On local machine
git pull
# Analyze logs in LOG.md or separate analysis scripts
```
