#!/bin/bash
#SBATCH --job-name=mixed_precision_h100
#SBATCH --nodes=1
#SBATCH --gpus=4                                    # Use all 4 H100s
#SBATCH --account=marlowe-m000066                   # REQUIRED - correct account
#SBATCH --partition=preempt                         # REQUIRED - correct partition
#SBATCH --time=12:00:00                             # Adjust based on expected runtime
#SBATCH --mem=256GB                                 # Can reduce with mixed precision
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/mixed_precision_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/mixed_precision_%j.err

# =============================================================================
# Mixed Precision Training on H100 GPUs
# =============================================================================
# This script runs training with PyTorch Automatic Mixed Precision (AMP)
# optimized for H100 Tensor Cores. Expects 2-3x speedup and 30-50% memory
# savings compared to FP32 training.
#
# Submit with: sbatch telepathy/submit_mixed_precision.slurm
# Monitor with: squeue -u $USER
# Cancel with: scancel <job_id>
# =============================================================================

# Set working directory - MUST use /projects path
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

echo "=============================================================="
echo "Mixed Precision Training on H100 GPUs"
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

# Enable NCCL optimizations for H100
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=2

# Create runs directory if needed
mkdir -p runs figures

# Pull latest code
echo "Pulling latest code..."
git pull

# Configuration
PRECISION="${PRECISION:-bf16}"  # bf16 recommended for H100
BATCH_SIZE="${BATCH_SIZE:-256}"  # Larger batch with mixed precision
SAMPLES="${SAMPLES:-87599}"      # Full SQuAD dataset
EPOCHS="${EPOCHS:-10}"

echo ""
echo "Configuration:"
echo "  Precision: $PRECISION"
echo "  Batch size: $BATCH_SIZE"
echo "  Samples: $SAMPLES"
echo "  Epochs: $EPOCHS"
echo ""

# Run mixed precision experiment
echo "Starting mixed precision training..."
python latentwire/train.py \
    --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --qwen_id "Qwen/Qwen2.5-7B-Instruct" \
    --mixed_precision "$PRECISION" \
    --amp_opt_level "O1" \
    --grad_scaler_init 65536 \
    --grad_scaler_growth_interval 2000 \
    --samples "$SAMPLES" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum_steps 1 \
    --latent_len 32 \
    --d_z 256 \
    --encoder_type byte \
    --dataset squad \
    --sequential_models \
    --warm_anchor_text "Answer: " \
    --first_token_ce_weight 0.5 \
    --k_ce_weight 1.0 \
    --kd_first_k_weight 0.1 \
    --K 4 \
    --lr 1e-4 \
    --max_grad_norm 1.0 \
    --grad_ckpt \
    --save_dir "runs/mixed_precision_${PRECISION}_${SLURM_JOB_ID}" \
    --save_every 1000 \
    --auto_resume

echo ""
echo "Training completed. Running evaluation..."

# Evaluate the final checkpoint
python latentwire/eval.py \
    --ckpt "runs/mixed_precision_${PRECISION}_${SLURM_JOB_ID}/last.pt" \
    --samples 500 \
    --max_new_tokens 12 \
    --dataset squad \
    --sequential_eval \
    --fresh_eval \
    --calibration embed_rms \
    --latent_anchor_mode text \
    --latent_anchor_text "Answer: " \
    --append_bos_after_prefix yes \
    --output_dir "runs/mixed_precision_${PRECISION}_${SLURM_JOB_ID}/eval"

# Generate performance comparison report
echo ""
echo "=============================================================="
echo "Performance Report"
echo "=============================================================="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv

# Push results back to git
echo ""
echo "Pushing results to git..."
git add -A
git commit -m "results: mixed precision experiment ($PRECISION on H100, job $SLURM_JOB_ID)

Configuration:
- Precision: $PRECISION
- Batch size: $BATCH_SIZE
- Samples: $SAMPLES
- Epochs: $EPOCHS
- H100 GPUs: 4

Expected improvements:
- 2-3x training speedup
- 30-50% memory reduction
- Higher GPU utilization

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.1 <noreply@anthropic.com>" || true
git push || true

echo "=============================================================="
echo "Job completed at $(date)"
echo "Results saved to: runs/mixed_precision_${PRECISION}_${SLURM_JOB_ID}/"
echo "=============================================================="