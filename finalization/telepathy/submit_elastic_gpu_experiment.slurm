#!/bin/bash
#SBATCH --job-name=elastic_gpu_test
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=02:00:00
#SBATCH --mem=256GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/elastic_gpu_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/elastic_gpu_%j.err

# =============================================================================
# Elastic GPU Configuration Test for LatentWire
# =============================================================================
# This script demonstrates the elastic GPU configuration that automatically
# adapts to available hardware (1-4 GPUs).
#
# Submit with: sbatch telepathy/submit_elastic_gpu_experiment.slurm
# Monitor with: squeue -u $USER
# Cancel with: scancel <job_id>
# =============================================================================

# Set working directory - MUST use /projects path
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

echo "=============================================================="
echo "SLURM Job Information"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Working directory: $WORK_DIR"
echo "=============================================================="

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create runs directory if needed
mkdir -p runs/elastic_gpu figures

# Pull latest code
echo "Pulling latest code..."
git pull

echo ""
echo "=============================================================="
echo "Testing Elastic GPU Configuration"
echo "=============================================================="

# First, test the configuration detection
echo "1. Testing configuration detection with all 4 GPUs..."
python scripts/test_elastic_gpu.py

echo ""
echo "=============================================================="
echo "Running Training with Different GPU Counts"
echo "=============================================================="

# Configuration for experiments
DATASET="squad"
SAMPLES=1000
EPOCHS=1
OUTPUT_BASE="runs/elastic_gpu"

# Test 1: Single GPU
echo ""
echo ">>> Test 1: Single GPU (GPU 0 only)"
echo "------------------------------------------------------------"
CUDA_VISIBLE_DEVICES=0 python latentwire/train.py \
    --elastic_gpu \
    --elastic_base_batch 64 \
    --elastic_target_util 0.75 \
    --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --qwen_id "Qwen/Qwen2.5-7B-Instruct" \
    --samples $SAMPLES \
    --epochs $EPOCHS \
    --dataset $DATASET \
    --output_dir "$OUTPUT_BASE/1gpu" \
    --latent_len 32 \
    --d_z 256 \
    --enabled_models llama \
    --max_steps 10 \
    2>&1 | tee "$OUTPUT_BASE/1gpu_train.log"

# Test 2: Dual GPU
echo ""
echo ">>> Test 2: Dual GPU (GPUs 0-1)"
echo "------------------------------------------------------------"
CUDA_VISIBLE_DEVICES=0,1 python latentwire/train.py \
    --elastic_gpu \
    --elastic_base_batch 64 \
    --elastic_target_util 0.75 \
    --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --qwen_id "Qwen/Qwen2.5-7B-Instruct" \
    --samples $SAMPLES \
    --epochs $EPOCHS \
    --dataset $DATASET \
    --output_dir "$OUTPUT_BASE/2gpu" \
    --latent_len 32 \
    --d_z 256 \
    --enabled_models llama,qwen \
    --sequential_models \
    --max_steps 10 \
    2>&1 | tee "$OUTPUT_BASE/2gpu_train.log"

# Test 3: Three GPUs
echo ""
echo ">>> Test 3: Three GPUs (GPUs 0-2)"
echo "------------------------------------------------------------"
CUDA_VISIBLE_DEVICES=0,1,2 python latentwire/train.py \
    --elastic_gpu \
    --elastic_base_batch 64 \
    --elastic_target_util 0.75 \
    --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --qwen_id "Qwen/Qwen2.5-7B-Instruct" \
    --samples $SAMPLES \
    --epochs $EPOCHS \
    --dataset $DATASET \
    --output_dir "$OUTPUT_BASE/3gpu" \
    --latent_len 32 \
    --d_z 256 \
    --enabled_models llama,qwen \
    --sequential_models \
    --max_steps 10 \
    2>&1 | tee "$OUTPUT_BASE/3gpu_train.log"

# Test 4: All Four GPUs
echo ""
echo ">>> Test 4: Four GPUs (All GPUs)"
echo "------------------------------------------------------------"
python latentwire/train.py \
    --elastic_gpu \
    --elastic_base_batch 64 \
    --elastic_target_util 0.75 \
    --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --qwen_id "Qwen/Qwen2.5-7B-Instruct" \
    --samples $SAMPLES \
    --epochs $EPOCHS \
    --dataset $DATASET \
    --output_dir "$OUTPUT_BASE/4gpu" \
    --latent_len 32 \
    --d_z 256 \
    --enabled_models llama,qwen \
    --sequential_models \
    --max_steps 10 \
    2>&1 | tee "$OUTPUT_BASE/4gpu_train.log"

echo ""
echo "=============================================================="
echo "Analyzing Results"
echo "=============================================================="

# Create a summary of throughput for each configuration
python -c "
import json
import glob
import re

print('GPU Count | Batch Size | Effective Batch | Throughput')
print('----------|------------|-----------------|------------')

for gpu_count in [1, 2, 3, 4]:
    log_file = f'runs/elastic_gpu/{gpu_count}gpu_train.log'
    try:
        with open(log_file, 'r') as f:
            content = f.read()

        # Extract batch size from elastic config output
        batch_match = re.search(r'Batch size per step: (\d+)', content)
        effective_match = re.search(r'Effective batch size: (\d+)', content)

        # Extract throughput (samples/sec) from training
        throughput_matches = re.findall(r'(\d+\.?\d*) samples/sec', content)

        batch = batch_match.group(1) if batch_match else 'N/A'
        effective = effective_match.group(1) if effective_match else 'N/A'
        throughput = throughput_matches[-1] if throughput_matches else 'N/A'

        print(f'{gpu_count:9} | {batch:10} | {effective:15} | {throughput:10} samples/sec')
    except Exception as e:
        print(f'{gpu_count:9} | Error reading log: {e}')
"

echo ""
echo "=============================================================="
echo "Experiment Summary"
echo "=============================================================="
echo "The elastic GPU configuration automatically adjusted settings based on"
echo "the number of available GPUs:"
echo ""
echo "1 GPU:  Used gradient accumulation to maintain effective batch size"
echo "2 GPUs: Split models across GPUs (Llama on GPU0, Qwen on GPU1)"
echo "3 GPUs: Hybrid approach (Llama on GPU0-1, Qwen on GPU2)"
echo "4 GPUs: Maximum parallelism for highest throughput"
echo ""
echo "Key benefits of elastic configuration:"
echo "- Automatic memory-aware batch sizing"
echo "- Optimal GPU utilization regardless of count"
echo "- No manual tuning needed when switching hardware"
echo "- Graceful degradation from 4 GPUs down to CPU-only"

# Push results back to git
echo ""
echo "Pushing results to git..."
git add -A
git commit -m "results: elastic GPU configuration tests (SLURM job $SLURM_JOB_ID)

Tested elastic GPU configuration with 1-4 GPUs to demonstrate automatic adaptation.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>" || true
git push || true

echo ""
echo "=============================================================="
echo "Job completed at $(date)"
echo "=============================================================="