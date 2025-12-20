#!/bin/bash
#SBATCH --job-name=enhanced_arxiv
#SBATCH --output=runs/slurm_%j.log
#SBATCH --error=runs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

# =============================================================================
# SLURM Submission Script for Enhanced ArXiv Experiments
# =============================================================================
# Submit with: sbatch telepathy/submit_enhanced_arxiv.slurm
# Monitor with: squeue -u $USER
# Cancel with: scancel <job_id>
# =============================================================================

echo "=============================================================="
echo "SLURM Job Information"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=============================================================="

# Navigate to project directory
cd /Users/sujeethjinesh/Desktop/LatentWire

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create runs directory if needed
mkdir -p runs

# Pull latest code
echo "Pulling latest code..."
git pull

# Clean previous runs (optional - comment out to keep)
# rm -rf runs/*

# Run the enhanced arxiv suite
echo "Starting enhanced arxiv experiments..."
bash telepathy/run_enhanced_arxiv_suite.sh

# Push results back to git
echo "Pushing results to git..."
git add -A
git commit -m "results: enhanced arxiv experiments (SLURM job $SLURM_JOB_ID)" || true
git push || true

echo "=============================================================="
echo "Job completed at $(date)"
echo "=============================================================="
