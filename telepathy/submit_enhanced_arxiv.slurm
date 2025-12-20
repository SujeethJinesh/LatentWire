#!/bin/bash
#SBATCH --job-name=enhanced_arxiv
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=12:00:00
#SBATCH --mem=256GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/slurm_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/slurm_%j.err

# =============================================================================
# SLURM Submission Script for Enhanced ArXiv Experiments
# =============================================================================
# Submit with: sbatch telepathy/submit_enhanced_arxiv.slurm
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

# Create runs directory if needed
mkdir -p runs

# Pull latest code
echo "Pulling latest code..."
git pull

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
