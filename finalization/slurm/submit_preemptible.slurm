#!/bin/bash
#SBATCH --job-name=latentwire-preempt
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=12:00:00
#SBATCH --mem=256GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/preempt_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/preempt_%j.err
#SBATCH --signal=TERM@120  # Send SIGTERM 120 seconds before job ends
#SBATCH --requeue          # Allow job to be requeued after preemption

# =============================================================================
# Preemptible LatentWire Training
# =============================================================================
# This script handles preemptible training with automatic checkpoint/resume
# Submit with: sbatch finalization/slurm/submit_preemptible.slurm
# Monitor with: squeue -u $USER
# Cancel with: scancel <job_id>
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
echo "Preemption signal configured: TERM@120"
echo "=============================================================="

# Set up environment
export PYTHONUNBUFFERED=1  # Critical: Immediate output flushing to prevent log loss
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"  # H100 support

# Create necessary directories
mkdir -p runs/preemptible figures logs

# Pull latest code
echo "Pulling latest code..."
git pull

# Configuration
EXPERIMENT_NAME="latentwire_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="runs/preemptible/${EXPERIMENT_NAME}"
mkdir -p "$OUTPUT_DIR"

# Log configuration
cat > "$OUTPUT_DIR/config.json" << EOF
{
    "slurm_job_id": "$SLURM_JOB_ID",
    "experiment_name": "$EXPERIMENT_NAME",
    "node": "$SLURMD_NODENAME",
    "gpus": "$CUDA_VISIBLE_DEVICES",
    "start_time": "$(date -Iseconds)"
}
EOF

# Run preemptible training with auto-resume
echo "Starting preemptible training..."
python finalization/training/preemptible_trainer.py \
    --save_dir "$OUTPUT_DIR" \
    --checkpoint_interval 300 \
    --auto_resume \
    --monitor_gpu \
    --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --samples 87599 \
    --epochs 24 \
    --batch_size 64 \
    --latent_len 32 \
    --d_z 256 \
    --encoder_type byte \
    --dataset squad \
    --warm_anchor_text "Answer: " \
    --first_token_ce_weight 0.5 \
    2>&1 | tee "$OUTPUT_DIR/training.log"

TRAINING_EXIT_CODE=$?

# Handle exit codes
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully"
    STATUS="completed"
elif [ $TRAINING_EXIT_CODE -eq 143 ]; then  # SIGTERM
    echo "Job preempted - will be requeued"
    STATUS="preempted"
else
    echo "Training failed with exit code $TRAINING_EXIT_CODE"
    STATUS="failed"
fi

# Save final status
cat > "$OUTPUT_DIR/status.json" << EOF
{
    "slurm_job_id": "$SLURM_JOB_ID",
    "exit_code": $TRAINING_EXIT_CODE,
    "status": "$STATUS",
    "end_time": "$(date -Iseconds)"
}
EOF

# Push results to git (if completed or checkpointed)
if [ -f "$OUTPUT_DIR/checkpoint_latest.pt" ]; then
    echo "Pushing checkpoint to git..."
    git add -A
    git commit -m "checkpoint: $EXPERIMENT_NAME (SLURM job $SLURM_JOB_ID, status: $STATUS)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>" || true
    git push || true
fi

echo "=============================================================="
echo "Job $STATUS at $(date)"
echo "Exit code: $TRAINING_EXIT_CODE"
echo "=============================================================="

# Exit with original code to trigger requeue if needed
exit $TRAINING_EXIT_CODEexport PYTHONUNBUFFERED=1
