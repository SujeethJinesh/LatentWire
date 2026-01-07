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

# =========================================================================
# Git Setup and Authentication Check
# =========================================================================
echo ""
echo "Git Configuration Check..."

# Configure git identity if not set (required for commits)
git config user.name > /dev/null 2>&1 || git config user.name "SLURM Job $SLURM_JOB_ID"
git config user.email > /dev/null 2>&1 || git config user.email "slurm@hpc.cluster"

# Pull latest code with error handling
echo "Pulling latest code..."
if ! git pull; then
    echo "WARNING: git pull failed. Attempting to stash and retry..."
    git stash push -m "SLURM job $SLURM_JOB_ID auto-stash"
    git pull || echo "Pull failed - continuing with existing code"
    git stash pop 2>/dev/null || true
fi

# Run the enhanced arxiv suite
echo "Starting enhanced arxiv experiments..."
bash telepathy/run_enhanced_arxiv_suite.sh

# =========================================================================
# Git Commit and Push Results
# =========================================================================
echo ""
echo "Saving results to git..."

# Only add log files and JSON results (not large checkpoints)
# Use find to ensure we get all log files recursively
find runs -name "*.log" -o -name "*.err" -o -name "*.json" | xargs git add 2>/dev/null || true
git add figures/*.png figures/*.pdf 2>/dev/null || true

# Also explicitly check for the enhanced arxiv log
if [ -f "runs/*/enhanced_arxiv.log" ]; then
    git add runs/*/enhanced_arxiv.log
    echo "Found and added enhanced_arxiv.log"
fi

# Commit with proper message format
COMMIT_MSG="results: enhanced arxiv experiments (SLURM job $SLURM_JOB_ID)

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>"

if git commit -m "$COMMIT_MSG"; then
    echo "Created commit, attempting to push..."

    # Push with retry logic
    RETRY_COUNT=0
    while [ $RETRY_COUNT -lt 3 ]; do
        if git push; then
            echo "Successfully pushed to remote"
            break
        else
            RETRY_COUNT=$((RETRY_COUNT + 1))
            echo "Push attempt $RETRY_COUNT failed"
            [ $RETRY_COUNT -lt 3 ] && sleep 5 && git pull --rebase=false || true
        fi
    done

    [ $RETRY_COUNT -eq 3 ] && echo "WARNING: Could not push after 3 attempts"
else
    echo "No changes to commit"
fi

echo "=============================================================="
echo "Job completed at $(date)"
echo "=============================================================="
