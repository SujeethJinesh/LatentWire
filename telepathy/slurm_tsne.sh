#!/bin/bash
#SBATCH --job-name=tsne_viz
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/tsne_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/tsne_%j.err

# t-SNE Visualization for AG News Latent Space
# Generates figure showing category separation in Bridge latents
#
# Submit with: sbatch telepathy/slurm_tsne.sh
# Monitor with: squeue -u $USER
# Cancel with: scancel <job_id>

# Force unbuffered output
export PYTHONUNBUFFERED=1

# Write immediately to a marker file to confirm job started
date > /projects/m000066/sujinesh/LatentWire/runs/tsne_started_$SLURM_JOB_ID.txt
echo "Job $SLURM_JOB_ID started" >> /projects/m000066/sujinesh/LatentWire/runs/tsne_started_$SLURM_JOB_ID.txt

# Don't use set -e - handle errors explicitly so we can see what fails

# Set working directory
WORK_DIR="/projects/m000066/sujinesh/LatentWire"

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo "=========================================="

echo "Changing to: $WORK_DIR"
cd "$WORK_DIR" || { echo "FATAL: Cannot cd to $WORK_DIR"; exit 1; }
echo "Current directory: $(pwd)"

# Load conda and activate environment
echo "Loading conda..."
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

echo "Activating latentwire environment..."
conda activate latentwire || echo "WARNING: conda activate failed, continuing with system Python"

export PYTHONPATH=.

# Show Python info
echo "Python: $(which python)"
python --version

# Show GPU info
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "nvidia-smi failed"

# Create directories
mkdir -p runs figures

# Pull latest code
echo ""
echo "Pulling latest code..."
git pull || echo "WARNING: git pull failed"

# Find AG News checkpoint
echo ""
echo "Searching for AG News checkpoint..."
echo "Looking for: runs/*agnews*/bridge.pt"

# Show all bridge.pt files
echo "All bridge.pt files found:"
find runs -name "bridge.pt" 2>/dev/null || echo "  (none)"

CHECKPOINT=$(find runs -name "bridge.pt" -path "*agnews*" 2>/dev/null | head -1)

if [ -z "$CHECKPOINT" ]; then
    echo ""
    echo "ERROR: No AG News checkpoint found in runs/"
    echo ""
    echo "Directory structure:"
    ls -la runs/ 2>/dev/null || echo "runs/ does not exist"
    echo ""
    # Still push logs so we can see what happened
    git add runs/tsne_*.log runs/tsne_*.err 2>/dev/null || true
    git commit -m "logs: t-SNE failed - no checkpoint (SLURM job $SLURM_JOB_ID)" 2>/dev/null || true
    git push 2>/dev/null || true
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT"

# Run t-SNE visualization
echo ""
echo "Generating t-SNE visualization..."
if python telepathy/generate_tsne_visualization.py \
    --checkpoint "$CHECKPOINT" \
    --output figures/agnews_tsne.pdf \
    --samples_per_class 100; then
    echo "Python script succeeded!"
    echo ""
    echo "Generated files:"
    ls -la figures/agnews_tsne.* 2>/dev/null || echo "No output files found"
    COMMIT_MSG="figures: AG News t-SNE visualization (SLURM job $SLURM_JOB_ID)

Shows semantic category separation in Bridge latent space.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
else
    echo "Python script FAILED with exit code $?"
    COMMIT_MSG="logs: t-SNE visualization failed (SLURM job $SLURM_JOB_ID)"
fi

# Push results to git (always, even on failure)
echo ""
echo "Pushing results to git..."
git add figures/*.pdf figures/*.png runs/tsne_*.log runs/tsne_*.err 2>/dev/null || true
git commit -m "$COMMIT_MSG" 2>/dev/null || echo "No changes to commit"
git push 2>/dev/null || echo "Push failed - may need manual push"

echo ""
echo "=========================================="
echo "Completed at: $(date)"
echo "=========================================="
