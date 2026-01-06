#!/bin/bash
#SBATCH --job-name=preemptible_orchestrator
#SBATCH --nodes=1
#SBATCH --gpus=4                                    # Use all 4 GPUs for maximum throughput
#SBATCH --account=marlowe-m000066                   # REQUIRED - correct account
#SBATCH --partition=preempt                         # REQUIRED - preemptible partition
#SBATCH --time=24:00:00                             # Maximum time (will be preempted before)
#SBATCH --mem=256GB                                 # Large memory for multiple experiments
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/orchestrator_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/orchestrator_%j.err
#SBATCH --signal=TERM@120                           # Send SIGTERM 120 seconds before job ends
#SBATCH --requeue                                   # Allow job to be requeued after preemption
#SBATCH --open-mode=append                         # Append to log files on requeue

# =============================================================================
# Preemptible Experiment Orchestrator
# =============================================================================
# This script runs the main orchestrator that manages all experiments with
# full preemption support, automatic checkpointing, and resumption.
#
# Submit with: sbatch telepathy/submit_preemptible_orchestrator.slurm
# Monitor with: squeue -u $USER
# Watch logs: tail -f runs/orchestrator_*.log
# Cancel with: scancel <job_id>
# =============================================================================

# Set working directory - MUST use /projects path
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

echo "=============================================================="
echo "PREEMPTIBLE ORCHESTRATOR - SLURM Job Information"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Requeue count: ${SLURM_RESTART_COUNT:-0}"
echo "Start time: $(date)"
echo "Working directory: $WORK_DIR"
echo "=============================================================="

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Enable NCCL optimizations for multi-GPU
export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0

# Create necessary directories
mkdir -p runs/orchestrated/{checkpoints,logs,results,figures}

# Determine if this is a restart
if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
    echo "=============================================================="
    echo "RESUMING FROM PREEMPTION (restart count: ${SLURM_RESTART_COUNT})"
    echo "=============================================================="
    RESUME_FLAG="--resume"
else
    echo "=============================================================="
    echo "STARTING FRESH RUN"
    echo "=============================================================="
    RESUME_FLAG=""

    # Pull latest code only on fresh start
    echo "Pulling latest code..."
    git pull
fi

# Function to handle cleanup on exit
cleanup() {
    echo "=============================================================="
    echo "Cleanup initiated at $(date)"
    echo "=============================================================="

    # Push any results to git
    echo "Pushing results to git..."
    git add -A runs/orchestrated/
    git commit -m "results: orchestrator checkpoint (job $SLURM_JOB_ID, restart ${SLURM_RESTART_COUNT:-0})

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>" || true
    git push || true

    echo "Cleanup complete"
}

# Set trap for cleanup
trap cleanup EXIT

# Run the orchestrator with full experiment suite
echo "=============================================================="
echo "Starting orchestrator..."
echo "=============================================================="

python telepathy/run_preemptible_experiments.py \
    --experiment all \
    --output_dir runs/orchestrated \
    $RESUME_FLAG \
    2>&1 | tee -a "runs/orchestrated/logs/orchestrator_${SLURM_JOB_ID}.log"

EXIT_CODE=${PIPESTATUS[0]}

echo "=============================================================="
echo "Orchestrator exit code: $EXIT_CODE"
echo "=============================================================="

# Check if we were preempted (exit code 0 from preemption handler)
if [ $EXIT_CODE -eq 0 ]; then
    echo "Clean exit (likely preempted) - ready for requeue"
else
    echo "Experiment completed or failed with code $EXIT_CODE"
fi

echo "=============================================================="
echo "Job completed at $(date)"
echo "Total runtime: $SECONDS seconds"
echo "=============================================================="

# Exit with the same code as the orchestrator
exit $EXIT_CODE