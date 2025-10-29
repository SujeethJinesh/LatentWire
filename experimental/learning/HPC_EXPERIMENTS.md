# Running Experiments on HPC - Resource Allocation Guide

This guide shows how to run both COCONUT and cross-model ablation experiments on HPC with optimal resource allocation.

## Resource Allocation Strategy

**Total: 4 nodes available**
- **3 nodes (12 GPUs)**: COCONUT Stage 0 training
- **1 node (4 GPUs)**: Cross-model ablation experiment

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

## Experiment 1: COCONUT Stage 0 Training (3 nodes)

### Location
```bash
cd experimental/learning/reproduce_coconut
```

### Quick Test (3 epochs, ~1-2 hours on 3 nodes)
```bash
conda activate 3_11
bash run_coconut_hpc.sh test

# Logs saved to: runs/stage0_test/coconut_stage0_test_TIMESTAMP.log
```

### Full Training (25 epochs, ~8-10 hours on 3 nodes)
```bash
conda activate 3_11
bash run_coconut_hpc.sh full

# Logs saved to: runs/stage0_full/coconut_stage0_full_TIMESTAMP.log
```

**Note**: The wrapper script automatically detects available GPUs and uses all of them (defaults to 12 if you have 3 nodes × 4 GPUs).

### Monitor Training
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Tail logs
tail -f runs/stage0_test/coconut_*.log   # For test run
tail -f runs/stage0_full/coconut_*.log   # For full run
```

---

## Experiment 2: Cross-Model Ablation (1 node)

### Location
```bash
cd experimental/learning
```

### Run Experiment
```bash
# Activate environment
conda activate 3_11

# Run on single node (will use 1 GPU automatically)
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

## Running Both Experiments Simultaneously

### Option 1: Separate Sessions (Recommended)

**Terminal 1** (3 nodes for COCONUT):
```bash
ssh node1  # or appropriate HPC login
cd /path/to/LatentWire/experimental/learning/reproduce_coconut/coconut
conda activate 3_11
torchrun --nproc_per_node=12 run.py args/gsm_cot.yaml
```

**Terminal 2** (1 node for cross-model):
```bash
ssh node4  # different node
cd /path/to/LatentWire/experimental/learning
conda activate 3_11
bash run_cross_model_ablation_hpc.sh
```

### Option 2: Background Jobs

```bash
# Start COCONUT in background on 3 nodes
cd experimental/learning/reproduce_coconut/coconut
nohup torchrun --nproc_per_node=12 run.py args/gsm_cot.yaml > coconut.log 2>&1 &
COCONUT_PID=$!

# Start cross-model in background on 1 node
cd ../../
nohup bash run_cross_model_ablation_hpc.sh &
ABLATION_PID=$!

# Monitor both
tail -f experimental/learning/reproduce_coconut/coconut/coconut.log
tail -f experimental/learning/runs/cross_model_ablation/cross_model_ablation_hpc_*.log

# Check status
ps -p $COCONUT_PID $ABLATION_PID
```

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

# COCONUT (3 nodes)
cd experimental/learning/reproduce_coconut
bash run_coconut_hpc.sh full  # or 'test' for quick 3-epoch run

# Cross-model ablation (1 node)
cd experimental/learning
bash run_cross_model_ablation_hpc.sh

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
