#!/bin/bash
#SBATCH --job-name=latentwire-finalize
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=12:00:00
#SBATCH --mem=256GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/finalize_%A_%a.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/finalize_%A_%a.err
#SBATCH --signal=TERM@120
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-9%1

# =============================================================================
# LatentWire Finalization Training with Preemption Handling
# =============================================================================
# This script runs the finalization training pipeline with:
# - Automatic resumption from checkpoints on preemption
# - Array job support for multiple resumption attempts
# - Dynamic GPU allocation (tries 4, then 2, then 1)
# - State management and progress tracking
# - Git integration for results
#
# Submit with: sbatch finalization/submit.slurm
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
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs Allocated: $SLURM_GPUS"
echo "GPU IDs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Working directory: $WORK_DIR"
echo "=============================================================="

# Set up environment
export PYTHONUNBUFFERED=1  # Critical: Immediate output flushing to prevent log loss
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HOME="/projects/m000066/sujinesh/.cache/huggingface"
export TORCH_HOME="/projects/m000066/sujinesh/.cache/torch"

# Configuration
BASE_OUTPUT_DIR="runs/finalization"
STATE_FILE="$BASE_OUTPUT_DIR/state.json"
CHECKPOINT_DIR="$BASE_OUTPUT_DIR/checkpoint"
LOG_DIR="$BASE_OUTPUT_DIR/logs"

# Create necessary directories
mkdir -p "$BASE_OUTPUT_DIR" "$LOG_DIR"

# Pull latest code
echo "Pulling latest code..."
git pull

# Function to save state
save_state() {
    local status="$1"
    local checkpoint="$2"
    local epoch="$3"

    cat > "$STATE_FILE" <<EOF
{
    "status": "$status",
    "checkpoint": "$checkpoint",
    "epoch": $epoch,
    "timestamp": "$(date -Iseconds)",
    "job_id": "$SLURM_JOB_ID",
    "array_task": $SLURM_ARRAY_TASK_ID,
    "node": "$SLURMD_NODENAME",
    "gpus": $SLURM_GPUS
}
EOF
}

# Function to handle preemption signal
handle_preemption() {
    echo ""
    echo "=============================================================="
    echo "PREEMPTION SIGNAL RECEIVED at $(date)"
    echo "=============================================================="

    # Find latest checkpoint
    if [ -d "$CHECKPOINT_DIR" ]; then
        LATEST_CHECKPOINT=$(ls -d "$CHECKPOINT_DIR"/epoch* 2>/dev/null | sort -V | tail -1)
        if [ -n "$LATEST_CHECKPOINT" ]; then
            EPOCH=$(basename "$LATEST_CHECKPOINT" | sed 's/epoch//')
            echo "Saving state with checkpoint: $LATEST_CHECKPOINT (epoch $EPOCH)"
            save_state "preempted" "$LATEST_CHECKPOINT" "$EPOCH"
        else
            echo "No checkpoint found, saving preempted state"
            save_state "preempted" "none" "0"
        fi
    else
        save_state "preempted" "none" "0"
    fi

    # Push current results to git
    echo "Pushing current results to git..."
    git add -A
    git commit -m "checkpoint: preemption save at epoch ${EPOCH:-0} (job $SLURM_JOB_ID)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>" || true
    git push || true

    echo "Gracefully exiting for requeue..."
    exit 0
}

# Set up signal handler for preemption
trap handle_preemption TERM

# Determine GPU configuration based on array task ID
# First attempts (0-3): Try with 4 GPUs
# Next attempts (4-6): Try with 2 GPUs
# Final attempts (7-9): Try with 1 GPU
if [ $SLURM_ARRAY_TASK_ID -lt 4 ]; then
    REQUESTED_GPUS=4
    BATCH_SIZE=64
elif [ $SLURM_ARRAY_TASK_ID -lt 7 ]; then
    REQUESTED_GPUS=2
    BATCH_SIZE=32
else
    REQUESTED_GPUS=1
    BATCH_SIZE=16
fi

echo "Array task $SLURM_ARRAY_TASK_ID requesting $REQUESTED_GPUS GPUs with batch size $BATCH_SIZE"

# Check if we have a previous state to resume from
RESUME_CHECKPOINT=""
RESUME_EPOCH=0

if [ -f "$STATE_FILE" ]; then
    echo "Found previous state file:"
    cat "$STATE_FILE"

    # Extract checkpoint path and epoch (using Python for robust JSON parsing)
    RESUME_INFO=$(python3 -c "
import json
with open('$STATE_FILE') as f:
    state = json.load(f)
    print(f\"{state.get('checkpoint', 'none')}:{state.get('epoch', 0)}\")
")

    RESUME_CHECKPOINT=$(echo "$RESUME_INFO" | cut -d: -f1)
    RESUME_EPOCH=$(echo "$RESUME_INFO" | cut -d: -f2)

    if [ "$RESUME_CHECKPOINT" != "none" ] && [ -d "$RESUME_CHECKPOINT" ]; then
        echo "Will resume from checkpoint: $RESUME_CHECKPOINT (epoch $RESUME_EPOCH)"
    else
        echo "No valid checkpoint to resume from, starting fresh"
        RESUME_CHECKPOINT=""
        RESUME_EPOCH=0
    fi
else
    echo "No previous state found, starting fresh"
fi

# Determine number of epochs remaining
TOTAL_EPOCHS=24
if [ $RESUME_EPOCH -gt 0 ]; then
    REMAINING_EPOCHS=$((TOTAL_EPOCHS - RESUME_EPOCH))
    echo "Resuming training: $REMAINING_EPOCHS epochs remaining (already completed $RESUME_EPOCH)"
else
    REMAINING_EPOCHS=$TOTAL_EPOCHS
    echo "Starting fresh training: $TOTAL_EPOCHS epochs"
fi

# Main training command
echo ""
echo "=============================================================="
echo "Starting Training Pipeline"
echo "=============================================================="
echo "Configuration:"
echo "  - GPUs requested: $REQUESTED_GPUS (allocated: $SLURM_GPUS)"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Epochs to run: $REMAINING_EPOCHS"
echo "  - Resume checkpoint: ${RESUME_CHECKPOINT:-none}"
echo "=============================================================="

# Create timestamped log for this specific run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TRAIN_LOG="$LOG_DIR/train_${TIMESTAMP}_task${SLURM_ARRAY_TASK_ID}.log"

# Build training command
TRAIN_CMD="python latentwire/train.py \
    --llama_id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --qwen_id Qwen/Qwen2.5-7B-Instruct \
    --samples 87599 \
    --epochs $REMAINING_EPOCHS \
    --batch_size $BATCH_SIZE \
    --latent_len 32 \
    --d_z 256 \
    --encoder_type byte \
    --dataset squad \
    --sequential_models \
    --warm_anchor_text 'Answer: ' \
    --first_token_ce_weight 0.5 \
    --output_dir $CHECKPOINT_DIR"

# Add resume flag if we have a checkpoint
if [ -n "$RESUME_CHECKPOINT" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume_from $RESUME_CHECKPOINT"
fi

# Run training with output capture (backgrounded to allow signal handling)
echo "Executing training command..."
{
    eval "$TRAIN_CMD"
} 2>&1 | tee "$TRAIN_LOG" &

TRAIN_PID=$!

# Wait for training to complete or be interrupted
wait $TRAIN_PID
TRAIN_EXIT_CODE=$?

echo ""
echo "=============================================================="
echo "Training completed with exit code: $TRAIN_EXIT_CODE"
echo "=============================================================="

# Check if training completed successfully
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"

    # Find final checkpoint
    FINAL_CHECKPOINT=$(ls -d "$CHECKPOINT_DIR"/epoch* 2>/dev/null | sort -V | tail -1)
    if [ -n "$FINAL_CHECKPOINT" ]; then
        FINAL_EPOCH=$(basename "$FINAL_CHECKPOINT" | sed 's/epoch//')
        save_state "completed" "$FINAL_CHECKPOINT" "$FINAL_EPOCH"

        # Run evaluation on final checkpoint
        echo ""
        echo "Running evaluation on final checkpoint..."
        EVAL_LOG="$LOG_DIR/eval_${TIMESTAMP}_task${SLURM_ARRAY_TASK_ID}.log"

        {
            python latentwire/eval.py \
                --ckpt "$FINAL_CHECKPOINT" \
                --samples 200 \
                --max_new_tokens 12 \
                --dataset squad \
                --sequential_eval \
                --fresh_eval \
                --calibration embed_rms \
                --latent_anchor_mode text \
                --latent_anchor_text "Answer: " \
                --append_bos_after_prefix yes
        } 2>&1 | tee "$EVAL_LOG"

        echo "Evaluation complete!"
    else
        echo "Warning: No checkpoint found after successful training"
        save_state "completed_no_checkpoint" "none" "0"
    fi

    # Mark job as complete to prevent further requeueing
    touch "$BASE_OUTPUT_DIR/.complete"

elif [ $TRAIN_EXIT_CODE -eq 143 ]; then
    # SIGTERM exit code - handled by trap
    echo "Training terminated by SIGTERM (handled by trap)"
else
    echo "Training failed with error code $TRAIN_EXIT_CODE"
    save_state "failed" "none" "0"

    # Don't requeue on actual failures
    touch "$BASE_OUTPUT_DIR/.failed"
fi

# Push final results to git
echo ""
echo "Pushing final results to git..."
git add -A
git commit -m "results: finalization training (job $SLURM_JOB_ID, task $SLURM_ARRAY_TASK_ID)

Status: ${TRAIN_EXIT_CODE}
Epochs completed: ${FINAL_EPOCH:-unknown}

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>" || true
git push || true

echo ""
echo "=============================================================="
echo "Job completed at $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "Logs saved to:"
echo "  - $TRAIN_LOG"
[ -n "$EVAL_LOG" ] && echo "  - $EVAL_LOG"
echo "=============================================================="

# Check if we should prevent requeue
if [ -f "$BASE_OUTPUT_DIR/.complete" ]; then
    echo "Training complete - preventing requeue"
    exit 0
elif [ -f "$BASE_OUTPUT_DIR/.failed" ]; then
    echo "Training failed - preventing requeue"
    exit 1
fi

# Exit with appropriate code for requeue
exit $TRAIN_EXIT_CODE