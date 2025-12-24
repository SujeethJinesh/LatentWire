#!/bin/bash
#SBATCH --job-name=tsne_viz
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/tsne_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/tsne_%j.err

# t-SNE Visualization for AG News Latent Space
# Generates figure showing category separation in Bridge latents

set -e

# Set working directory
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo "Working dir: $WORK_DIR"
echo "=========================================="

export PYTHONPATH=.

# Create directories
mkdir -p runs figures

# Pull latest code
echo "Pulling latest code..."
git pull

# Find AG News checkpoint
CHECKPOINT=$(find runs -name "bridge.pt" -path "*agnews*" 2>/dev/null | head -1)

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: No AG News checkpoint found. Run training first."
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT"

# Activate conda environment if needed
# source ~/miniconda3/bin/activate latentwire

# Run t-SNE visualization
echo ""
echo "Generating t-SNE visualization..."
python telepathy/generate_tsne_visualization.py \
    --checkpoint "$CHECKPOINT" \
    --output figures/agnews_tsne.pdf \
    --samples_per_class 100

echo ""
echo "Generated files:"
ls -la figures/agnews_tsne.*

# Push results to git
echo ""
echo "Pushing results to git..."
git add figures/*.pdf figures/*.png logs/tsne_*.out logs/tsne_*.err 2>/dev/null || true
git commit -m "figures: AG News t-SNE visualization (SLURM job $SLURM_JOB_ID)

Shows semantic category separation in Bridge latent space.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>" 2>/dev/null || echo "No changes to commit"

git push 2>/dev/null || echo "Push failed - may need manual push"

echo ""
echo "=========================================="
echo "Completed at: $(date)"
echo "=========================================="
