#!/bin/bash
#SBATCH --job-name=preempt_latentwire
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=24:00:00
#SBATCH --mem=256GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/preemptible_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/preemptible_%j.err
#SBATCH --signal=TERM@120                           # Send SIGTERM 120 seconds before job ends
#SBATCH --requeue                                   # Allow job to be requeued after preemption
#SBATCH --open-mode=append                          # Append to log files on requeue

# =============================================================================
# PREEMPTION-SAFE TRAINING FOR LATENTWIRE
# =============================================================================
# This script runs training with automatic checkpoint saving and resumption
# on preemption. The job will automatically requeue itself if preempted.
#
# Submit with: sbatch telepathy/submit_preemptible.slurm
# Monitor with: squeue -u $USER
# Cancel with: scancel <job_id>
#
# The training will:
# - Save checkpoints every 5 minutes
# - Save immediately on preemption signal
# - Resume from exact point when requeued
# - Handle mid-batch interruptions
# =============================================================================

# Set working directory - MUST use /projects path
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

echo "=============================================================="
echo "SLURM Preemptible Job Information"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Working directory: $WORK_DIR"

# Check if this is a requeued job
if [ -n "$SLURM_RESTART_COUNT" ]; then
    echo "REQUEUE: This is restart #$SLURM_RESTART_COUNT"
fi
echo "=============================================================="

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create runs directory if needed
mkdir -p runs figures

# Pull latest code
echo "Pulling latest code..."
git pull

# Configure experiment
EXPERIMENT_NAME="preemptible_exp"
SAVE_DIR="runs/$EXPERIMENT_NAME"
CHECKPOINT_INTERVAL=300  # Save every 5 minutes
GRACE_PERIOD=120        # 120 seconds to save on preemption

# Run preemptible training
echo "Starting preemptible training..."
echo "Save directory: $SAVE_DIR"
echo "Checkpoint interval: ${CHECKPOINT_INTERVAL}s"
echo "Grace period: ${GRACE_PERIOD}s"
echo ""

python telepathy/preemptible_training.py \
    --auto_resume \
    --checkpoint_interval $CHECKPOINT_INTERVAL \
    --grace_period $GRACE_PERIOD \
    --save_dir "$SAVE_DIR" \
    --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --qwen_id "Qwen/Qwen2.5-7B-Instruct" \
    --samples 87599 \
    --epochs 24 \
    --batch_size 64 \
    --latent_len 32 \
    --d_z 256 \
    --encoder_type byte \
    --dataset squad \
    --sequential_models \
    --warm_anchor_text "Answer: " \
    --first_token_ce_weight 0.5 \
    --learning_rate 5e-4 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0

# Check exit status
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"

    # Push results back to git
    echo "Pushing results to git..."
    git add -A
    git commit -m "results: preemptible training completed (SLURM job $SLURM_JOB_ID)

Experiment: $EXPERIMENT_NAME
Checkpoint interval: ${CHECKPOINT_INTERVAL}s
Final status: SUCCESS

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>" || true
    git push || true

elif [ $EXIT_CODE -eq 99 ]; then
    echo "🔄 Job preempted - will be requeued automatically"

    # Commit intermediate results
    git add -A
    git commit -m "checkpoint: preemptible training interrupted (SLURM job $SLURM_JOB_ID)

Experiment: $EXPERIMENT_NAME
Status: PREEMPTED - will resume
Restart count: $SLURM_RESTART_COUNT

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>" || true
    git push || true

else
    echo "❌ Training failed with exit code $EXIT_CODE"

    # Commit error state
    git add -A
    git commit -m "error: preemptible training failed (SLURM job $SLURM_JOB_ID)

Experiment: $EXPERIMENT_NAME
Exit code: $EXIT_CODE
Check logs for details

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>" || true
    git push || true
fi

echo "=============================================================="
echo "Job ended at $(date)"
echo "Exit code: $EXIT_CODE"
echo "=============================================================="

# Exit with the same code for proper SLURM handling
exit $EXIT_CODE
