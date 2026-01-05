#!/bin/bash
#SBATCH --job-name=unified_experiments
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=04:00:00
#SBATCH --mem=40GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/unified_experiments_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/unified_experiments_%j.err
#SBATCH --requeue                                    # Auto-requeue on preemption
#SBATCH --open-mode=append                          # Append to logs on restart

# =============================================================================
# SLURM WRAPPER FOR UNIFIED EXPERIMENTS
# =============================================================================
# This script automatically handles preemption and restarts.
# Submit with: sbatch telepathy/submit_experiments.slurm
# Monitor with: squeue -u $USER
# Cancel with: scancel <job_id>
#
# The script will automatically:
# - Resume from where it left off if preempted
# - Save state between runs
# - Aggregate results when complete
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
echo "Restart count: ${SLURM_RESTART_COUNT:-0}"
echo "=============================================================="

# Check if this is a restart
if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
    echo "This is restart #${SLURM_RESTART_COUNT} after preemption"
fi

# Run the main experiment script
# It handles all state management and recovery internally
bash telepathy/run_experiments.sh

# Exit code will be 0 if completed, non-zero if preempted
EXIT_CODE=$?

echo "=============================================================="
echo "Job ended at $(date) with exit code $EXIT_CODE"
echo "=============================================================="

# If preempted (exit code != 0), SLURM will automatically requeue
exit $EXIT_CODE