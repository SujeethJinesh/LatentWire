#!/bin/bash
#SBATCH --job-name=latentwire_paper_revision
#SBATCH --nodes=1
#SBATCH --gpus=1                                    # Use 1 GPU for efficiency
#SBATCH --account=marlowe-m000066                   # REQUIRED - correct account
#SBATCH --partition=preempt                         # REQUIRED - correct partition
#SBATCH --time=48:00:00                             # 48 hours for all experiments
#SBATCH --mem=128GB                                 # Sufficient for all phases
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/paper_revision_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/paper_revision_%j.err
#SBATCH --signal=SIGTERM@120                        # 2 minute grace period for cleanup

# =============================================================================
# COMPREHENSIVE PAPER REVISION EXPERIMENTS FOR LATENTWIRE
# =============================================================================
# This SLURM script runs ALL experiments needed for the paper revision:
# - Phase 1: Statistical rigor (3 seeds, bootstrap CI)
# - Phase 2: Linear probe baseline
# - Phase 3: Fair baseline comparisons (LLMLingua, etc.)
# - Phase 4: Efficiency measurements
#
# Resilient to preemption with automatic checkpoint resumption
# =============================================================================
# Submit with: sbatch finalization/submit_all_experiments.slurm
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
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export TOKENIZERS_PARALLELISM=true

# Create runs directory if needed
mkdir -p runs

# Pull latest code
echo "Pulling latest code..."
git pull

# Set experiment configuration
export EXP_NAME="paper_revision_job${SLURM_JOB_ID}"
export BASE_OUTPUT_DIR="runs/${EXP_NAME}"
export NUM_GPUS=1
export BATCH_SIZE=8
export EVAL_BATCH_SIZE=16

# Configure datasets and seeds for comprehensive evaluation
export SEEDS="42 123 456"
export DATASETS="sst2 agnews trec xsum"
export BOOTSTRAP_SAMPLES=10000

# Check if we should resume from existing checkpoint
if [ -d "runs/paper_revision_latest/checkpoint" ]; then
    echo "Found existing checkpoint, will resume..."
    export SKIP_TRAINING=yes
    export CHECKPOINT_PATH="runs/paper_revision_latest/checkpoint"
else
    echo "No existing checkpoint found, will train from scratch..."
    export SKIP_TRAINING=no
fi

# Function to handle preemption
handle_preemption() {
    echo "=============================================================="
    echo "Handling preemption signal at $(date)"
    echo "=============================================================="

    # Create symlink to latest results for easy resumption
    ln -sfn "$EXP_NAME" runs/paper_revision_latest

    # Push intermediate results to git
    echo "Saving intermediate results..."
    git add -f "$BASE_OUTPUT_DIR/logs"/*.log \
              "$BASE_OUTPUT_DIR/results"/**/*.json \
              "$BASE_OUTPUT_DIR"/*.md \
              "$BASE_OUTPUT_DIR"/*.json 2>/dev/null || true

    git commit -m "results: paper revision intermediate (SLURM job $SLURM_JOB_ID)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>" || true

    git push || true

    echo "Preemption handling complete"
    exit 0
}

# Set up signal handler for preemption
trap handle_preemption SIGTERM

# Run the main experiment script
echo "Starting comprehensive experiments..."
bash finalization/run_all_experiments.sh &

# Store the PID of the background process
EXPERIMENT_PID=$!

# Wait for the experiment to complete or for a signal
wait $EXPERIMENT_PID
EXPERIMENT_EXIT_CODE=$?

echo "=============================================================="
echo "Experiment completed with exit code: $EXPERIMENT_EXIT_CODE"
echo "=============================================================="

# Push final results to git
echo "Pushing final results to git..."
git add -f "$BASE_OUTPUT_DIR/logs"/*.log \
          "$BASE_OUTPUT_DIR/results"/**/*.json \
          "$BASE_OUTPUT_DIR/paper_assets"/* \
          "$BASE_OUTPUT_DIR"/*.md \
          "$BASE_OUTPUT_DIR"/*.json 2>/dev/null || true

if [ $EXPERIMENT_EXIT_CODE -eq 0 ]; then
    commit_msg="results: paper revision complete (SLURM job $SLURM_JOB_ID)"
else
    commit_msg="results: paper revision partial (exit code $EXPERIMENT_EXIT_CODE, SLURM job $SLURM_JOB_ID)"
fi

git commit -m "$commit_msg

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>" || true

git push || true

# Create final symlink for easy access
ln -sfn "$EXP_NAME" runs/paper_revision_latest

echo "=============================================================="
echo "Job completed at $(date)"
echo "Results available at: $BASE_OUTPUT_DIR"
echo "=============================================================="

# Print summary if available
if [ -f "$BASE_OUTPUT_DIR/experiment_summary.md" ]; then
    echo ""
    echo "Experiment Summary:"
    echo "-------------------"
    cat "$BASE_OUTPUT_DIR/experiment_summary.md"
fi

exit $EXPERIMENT_EXIT_CODE