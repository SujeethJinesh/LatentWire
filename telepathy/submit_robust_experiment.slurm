#!/bin/bash
#SBATCH --job-name=robust_training
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=48:00:00
#SBATCH --mem=256GB
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/robust_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/robust_%j.err

# =============================================================================
# Robust Training with Automatic Recovery
# =============================================================================
# This script runs training with automatic recovery from:
# - OOM errors (reduces batch size automatically)
# - Gradient explosions (clips and continues)
# - Network errors (retries with exponential backoff)
# - Checkpoint corruption (uses backups)
# - Interrupts (saves emergency checkpoint)
#
# Submit with: sbatch telepathy/submit_robust_experiment.slurm
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

# Pull latest code
echo "Pulling latest code..."
git pull

# Define experiment configuration
EXPERIMENT_NAME="robust_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="runs/${EXPERIMENT_NAME}"

echo "Starting robust training experiment..."
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run the robust training wrapper
# The wrapper will handle all failures and retries automatically
python telepathy/robust_training.py \
    --checkpoint_dir "$OUTPUT_DIR" \
    --max_retries 3 \
    --max_oom_retries 5 \
    --batch_size 64 \
    --min_batch_size 4 \
    --memory_threshold_gb 70.0 \
    --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --qwen_id "Qwen/Qwen2.5-7B-Instruct" \
    --samples 87599 \
    --epochs 24 \
    --latent_len 32 \
    --d_z 256 \
    --encoder_type byte \
    --dataset squad \
    --sequential_models \
    --warm_anchor_text "Answer: " \
    --first_token_ce_weight 0.5

# Check exit status
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"

    # Run evaluation on the final checkpoint
    echo "Running evaluation..."
    LATEST_CHECKPOINT=$(ls -td "$OUTPUT_DIR"/epoch* | head -1)

    python latentwire/eval.py \
        --ckpt "$LATEST_CHECKPOINT" \
        --samples 200 \
        --max_new_tokens 12 \
        --dataset squad \
        --sequential_eval \
        --fresh_eval \
        --calibration embed_rms \
        --latent_anchor_mode text \
        --latent_anchor_text "Answer: " \
        --append_bos_after_prefix yes \
        2>&1 | tee "${OUTPUT_DIR}/eval_results.log"
else
    echo "Training failed after all retries. Check logs for details."
fi

# Push results back to git
echo "Pushing results to git..."
git add -A
git commit -m "results: robust training experiment ${EXPERIMENT_NAME} (SLURM job $SLURM_JOB_ID)

Exit code: $EXIT_CODE
Node: $SLURMD_NODENAME

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>" || true
git push || true

echo "=============================================================="
echo "Job completed at $(date)"
echo "Exit code: $EXIT_CODE"
echo "=============================================================="

exit $EXIT_CODE