#!/bin/bash
#SBATCH --job-name=experiment_name
#SBATCH --nodes=1
#SBATCH --gpus=4                                    # Use 4 for full experiments, 1 for small jobs
#SBATCH --account=marlowe-m000066                   # REQUIRED - correct account
#SBATCH --partition=preempt                         # REQUIRED - correct partition
#SBATCH --time=12:00:00                             # Adjust based on expected runtime
#SBATCH --mem=256GB                                 # Adjust based on needs (64GB for small, 256GB for large)
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/experiment_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/experiment_%j.err

# =============================================================================
# TEMPLATE: Description of what this script does
# =============================================================================
# Submit with: sbatch telepathy/submit_template_with_git.slurm
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
mkdir -p runs figures

# =========================================================================
# Git Setup and Authentication Check
# =========================================================================
echo ""
echo "=============================================================="
echo "Git Configuration and Authentication Check"
echo "=============================================================="

# Configure git identity if not set (required for commits)
if ! git config user.name > /dev/null 2>&1; then
    echo "Setting git user.name..."
    git config user.name "SLURM Job $SLURM_JOB_ID"
fi

if ! git config user.email > /dev/null 2>&1; then
    echo "Setting git user.email..."
    git config user.email "slurm@hpc.cluster"
fi

# Show current git status
echo "Current git status:"
git status --short

# Pull latest code with error handling
echo ""
echo "Pulling latest code..."
if git pull; then
    echo "Successfully pulled latest changes"
else
    echo "WARNING: git pull failed. Continuing with local code."
    echo "This may be due to:"
    echo "  1. SSH key authentication issues"
    echo "  2. Network connectivity"
    echo "  3. Uncommitted local changes"
    echo ""
    echo "Attempting to stash and pull..."
    git stash push -m "SLURM job $SLURM_JOB_ID auto-stash"
    if git pull; then
        echo "Pull successful after stash"
        git stash pop || echo "Could not pop stash - manual intervention may be needed"
    else
        echo "Pull still failed - continuing with existing code"
    fi
fi

echo "=============================================================="
echo ""

# =========================================================================
# Main Experiment Execution
# =========================================================================
echo "Starting experiment..."

# YOUR EXPERIMENT COMMAND HERE
# Example:
# python latentwire/train.py \
#     --samples 1000 \
#     --epochs 1 \
#     --output_dir runs/experiment

# =========================================================================
# Git Commit and Push Results
# =========================================================================
echo ""
echo "=============================================================="
echo "Saving Results to Git"
echo "=============================================================="

# Check what files would be added
echo "Files to be committed:"
git status --short runs/ figures/ 2>/dev/null || echo "No runs/figures to commit"

# Only commit if there are changes in runs/ or figures/
if git status --short runs/ figures/ 2>/dev/null | grep -q .; then
    echo "Adding results to git..."

    # Add only log files and JSON results (not checkpoints)
    git add runs/*.log runs/*.err runs/*.json runs/**/*.log runs/**/*.json 2>/dev/null || true
    git add figures/*.png figures/*.pdf 2>/dev/null || true

    # Check size of staged files to avoid large commits
    STAGED_SIZE=$(git diff --cached --stat | tail -1 | awk '{print $1}')
    echo "Staged files statistics: $STAGED_SIZE"

    # Create commit with proper message format
    COMMIT_MSG="results: experiment description (SLURM job $SLURM_JOB_ID)

Automated commit from HPC cluster
Node: $SLURMD_NODENAME
Time: $(date)

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>"

    if git commit -m "$COMMIT_MSG"; then
        echo "Successfully created commit"

        # Attempt to push with retry logic
        MAX_RETRIES=3
        RETRY_COUNT=0

        while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
            if git push; then
                echo "Successfully pushed results to remote"
                break
            else
                RETRY_COUNT=$((RETRY_COUNT + 1))
                echo "Push attempt $RETRY_COUNT failed"

                if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
                    echo "Retrying in 5 seconds..."
                    sleep 5

                    # Try to pull and merge in case of conflicts
                    echo "Attempting to pull and merge..."
                    git pull --rebase=false || true
                else
                    echo "ERROR: Could not push after $MAX_RETRIES attempts"
                    echo "Results are committed locally but not pushed"
                    echo "Manual intervention required to push:"
                    echo "  cd $WORK_DIR"
                    echo "  git push"
                fi
            fi
        done
    else
        echo "WARNING: No changes to commit (this is normal if no new results were generated)"
    fi
else
    echo "No results to commit in runs/ or figures/"
fi

echo "=============================================================="
echo ""

# =========================================================================
# Final Summary
# =========================================================================
echo "=============================================================="
echo "Job Summary"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Completed at: $(date)"
echo "Total runtime: $SECONDS seconds"

# List generated files
echo ""
echo "Generated files:"
ls -lh runs/*${SLURM_JOB_ID}* 2>/dev/null || echo "  No job-specific files in runs/"
ls -lh runs/experiment* 2>/dev/null | head -5 || echo "  No experiment files"

echo ""
echo "To view logs on HPC:"
echo "  tail -f $WORK_DIR/runs/*_${SLURM_JOB_ID}.log"
echo ""
echo "To pull results locally:"
echo "  git pull"
echo "=============================================================="