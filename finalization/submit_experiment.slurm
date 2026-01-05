#!/bin/bash
#SBATCH --job-name=latentwire_experiment
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=12:00:00
#SBATCH --mem=256GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/%x_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/%x_%j.err
#SBATCH --signal=SIGUSR1@90  # Send signal 90 seconds before timeout
#SBATCH --requeue  # Automatically requeue on preemption

# =============================================================================
# SLURM Submission Script for LatentWire Experiments
# =============================================================================
# This script wraps the main orchestrator for SLURM batch submission.
#
# Submit with:
#   sbatch finalization/submit_experiment.slurm
#
# Or with custom parameters:
#   sbatch --gpus=2 --time=06:00:00 finalization/submit_experiment.slurm
#
# Monitor with:
#   squeue -u $USER
#   tail -f runs/latentwire_experiment_*.log
#
# Cancel with:
#   scancel <job_id>
# =============================================================================

# Set working directory
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

echo "=============================================================="
echo "SLURM Job Information"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "GPUs requested: $SLURM_GPUS"
echo "GPUs visible: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Working directory: $WORK_DIR"
echo "=============================================================="
echo ""

# Configure git for automatic commits
git config user.name > /dev/null 2>&1 || git config user.name "SLURM Job $SLURM_JOB_ID"
git config user.email > /dev/null 2>&1 || git config user.email "slurm@marlowe.cluster"

# Pull latest code
echo "Pulling latest code..."
if ! git pull; then
    echo "WARNING: git pull failed. Attempting to stash and retry..."
    git stash push -m "SLURM job $SLURM_JOB_ID auto-stash $(date +%Y%m%d_%H%M%S)"
    git pull || echo "Pull failed - continuing with existing code"
    git stash pop 2>/dev/null || true
fi

# Critical environment variable for Python output buffering
export PYTHONUNBUFFERED=1  # Critical: Immediate output flushing to prevent log loss

# Set experiment name based on SLURM job
export EXP_NAME="${EXP_NAME:-slurm_job_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)}"
export WORK_DIR="$WORK_DIR"

# Configure experiment parameters (can be overridden via environment)
export DATASET="${DATASET:-squad}"
export SAMPLES="${SAMPLES:-10000}"
export EPOCHS="${EPOCHS:-3}"
export LATENT_LEN="${LATENT_LEN:-32}"
export D_Z="${D_Z:-256}"

# Hardware optimization settings
export TARGET_GPU_UTIL="${TARGET_GPU_UTIL:-0.75}"
export BASE_BATCH_SIZE="${BASE_BATCH_SIZE:-64}"
export NUM_WORKERS="${NUM_WORKERS:-8}"  # More workers on HPC
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"  # More prefetching on HPC

# Enable automatic resumption on preemption
export RESUME="${RESUME:-auto}"
export MAX_RETRIES="${MAX_RETRIES:-10}"

echo "Experiment Configuration:"
echo "  Name: $EXP_NAME"
echo "  Dataset: $DATASET"
echo "  Samples: $SAMPLES"
echo "  Epochs: $EPOCHS"
echo "  Latent dimensions: ${LATENT_LEN}x${D_Z}"
echo ""

# =============================================================================
# PREEMPTION SIGNAL HANDLER
# =============================================================================

handle_preemption_signal() {
    echo ""
    echo "=============================================================="
    echo "PREEMPTION SIGNAL RECEIVED - Initiating graceful shutdown"
    echo "=============================================================="

    # Send SIGTERM to the orchestrator script to trigger cleanup
    if [[ -n "${ORCHESTRATOR_PID:-}" ]]; then
        echo "Signaling orchestrator process (PID: $ORCHESTRATOR_PID) to save checkpoint..."
        kill -TERM "$ORCHESTRATOR_PID" 2>/dev/null || true

        # Wait for orchestrator to finish cleanup (max 60 seconds)
        local wait_count=0
        while kill -0 "$ORCHESTRATOR_PID" 2>/dev/null && [[ $wait_count -lt 60 ]]; do
            sleep 1
            wait_count=$((wait_count + 1))
        done

        if [[ $wait_count -eq 60 ]]; then
            echo "WARNING: Orchestrator did not terminate cleanly, forcing..."
            kill -KILL "$ORCHESTRATOR_PID" 2>/dev/null || true
        fi
    fi

    # Ensure logs are saved
    echo "Saving logs to git..."
    save_logs_to_git

    echo "Preemption handling complete. Job will be requeued automatically."
    exit 0
}

# Register signal handler for SLURM preemption
trap handle_preemption_signal SIGUSR1 SIGTERM

# =============================================================================
# LOG SAVING FUNCTION
# =============================================================================

save_logs_to_git() {
    echo "Committing logs and results to git..."

    # Add logs and small result files
    git add -f runs/*.log runs/*.err runs/*.json runs/**/*.log runs/**/*.json 2>/dev/null || true
    git add -f runs/$EXP_NAME/.orchestrator_state runs/$EXP_NAME/.retry_count 2>/dev/null || true
    git add -f runs/$EXP_NAME/logs/*.log runs/$EXP_NAME/*.jsonl 2>/dev/null || true

    # Create commit message
    local commit_msg="results: $EXP_NAME experiment logs (SLURM job $SLURM_JOB_ID)

Experiment: $EXP_NAME
Dataset: $DATASET
Samples: $SAMPLES
Epochs: $EPOCHS
Node: $SLURMD_NODENAME
Time: $(date)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>"

    if git diff --staged --quiet; then
        echo "No changes to commit"
    else
        if git commit -m "$commit_msg"; then
            echo "Created commit, attempting to push..."

            # Push with retry logic
            local retry_count=0
            while [[ $retry_count -lt 3 ]]; do
                if git push; then
                    echo "Successfully pushed to remote"
                    break
                else
                    retry_count=$((retry_count + 1))
                    echo "Push attempt $retry_count failed"
                    [[ $retry_count -lt 3 ]] && sleep 5 && git pull --rebase=false || true
                fi
            done

            [[ $retry_count -eq 3 ]] && echo "WARNING: Could not push after 3 attempts"
        else
            echo "WARNING: Could not create commit"
        fi
    fi
}

# =============================================================================
# RUN THE MAIN ORCHESTRATOR
# =============================================================================

echo ""
echo "Launching main orchestrator script..."
echo "=============================================================="

# Run orchestrator in background so we can handle signals
bash finalization/run_experiment.sh &
ORCHESTRATOR_PID=$!

echo "Orchestrator launched with PID: $ORCHESTRATOR_PID"

# Wait for orchestrator to complete or for preemption signal
wait $ORCHESTRATOR_PID
ORCHESTRATOR_EXIT_CODE=$?

echo ""
echo "=============================================================="
echo "Orchestrator exited with code: $ORCHESTRATOR_EXIT_CODE"

# Save final logs
save_logs_to_git

echo "=============================================================="
echo "Job completed at $(date)"
echo "Exit code: $ORCHESTRATOR_EXIT_CODE"
echo "=============================================================="

# Exit with orchestrator's exit code
exit $ORCHESTRATOR_EXIT_CODE