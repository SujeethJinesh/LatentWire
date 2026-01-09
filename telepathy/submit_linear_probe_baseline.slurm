#!/bin/bash
#SBATCH --job-name=linear_probe_baseline
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/linear_probe_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/linear_probe_%j.err

# =============================================================================
# Linear Probe Baseline for Telepathy Paper
# =============================================================================
# This script runs a comprehensive linear probe baseline evaluation on frozen
# Llama hidden states to demonstrate that the Perceiver architecture adds
# value beyond simple linear projection.
#
# Evaluated datasets:
# - SST-2 (2-class sentiment)
# - AG News (4-class topic)
# - TREC (6-class question type)
# - Banking77 (77-class intent)
# - IMDB (2-class sentiment)
# - MNLI (3-class NLI)
# - 20 Newsgroups (20-class topic)
#
# Evaluated layers: 16, 20, 24, 28, 31
# Seeds: 42, 123, 456 (for statistical significance)
# =============================================================================
# Submit with: sbatch telepathy/submit_linear_probe_baseline.slurm
# Monitor with: squeue -u $USER
# Cancel with: scancel <job_id>
# =============================================================================

# Set working directory
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

# Create output directory
OUTPUT_DIR="runs/linear_probe_baseline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
mkdir -p runs figures

# Pull latest code
echo ""
echo "Pulling latest code..."
git pull

# Check GPU availability
echo ""
echo "Checking GPU..."
nvidia-smi || echo "nvidia-smi not available"

# Run the linear probe baseline
echo ""
echo "=============================================================="
echo "Starting Linear Probe Baseline Experiment"
echo "=============================================================="

python telepathy/linear_probe_baseline.py \
    --datasets sst2 agnews trec banking77 imdb mnli 20newsgroups \
    --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --layers 16 20 24 28 31 \
    --pooling last_token \
    --max_train_samples 5000 \
    --max_test_samples 1000 \
    --batch_size 8 \
    --cv_folds 5 \
    --seeds 42 123 456 \
    --output_dir "$OUTPUT_DIR" \
    --device cuda

# Check if experiment completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================================="
    echo "Experiment completed successfully!"
    echo "=============================================================="
else
    echo ""
    echo "=============================================================="
    echo "ERROR: Experiment failed!"
    echo "=============================================================="
fi

# Print results summary
echo ""
echo "Results saved to: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"

# Push results back to git
echo ""
echo "Pushing results to git..."
git add -A
git commit -m "results: linear probe baseline across 7 datasets (SLURM job $SLURM_JOB_ID)

Evaluated layers: 16, 20, 24, 28, 31
Datasets: SST-2, AG News, TREC, Banking77, IMDB, MNLI, 20 Newsgroups
Seeds: 42, 123, 456

This baseline demonstrates whether the Perceiver architecture adds value
beyond simple linear projection of frozen Llama hidden states.

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>" || true
git push || true

echo ""
echo "=============================================================="
echo "Job completed at $(date)"
echo "=============================================================="
