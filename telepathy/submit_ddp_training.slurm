#!/bin/bash
#SBATCH --job-name=ddp_training
#SBATCH --nodes=1
#SBATCH --gpus=4                                    # Request 4 H100 GPUs
#SBATCH --account=marlowe-m000066                   # REQUIRED - correct account
#SBATCH --partition=preempt                         # REQUIRED - correct partition
#SBATCH --time=12:00:00                             # 12 hours should be enough
#SBATCH --mem=256GB                                 # 256GB for large models
#SBATCH --output=/projects/m000066/sujinesh/LatentWire/runs/ddp_training_%j.log
#SBATCH --error=/projects/m000066/sujinesh/LatentWire/runs/ddp_training_%j.err

# =============================================================================
# DDP (Distributed Data Parallel) Training on 4x H100 GPUs
# =============================================================================
# This script runs LatentWire training with DDP for maximum throughput on
# multiple GPUs. It automatically detects the number of available GPUs and
# configures DDP accordingly.
#
# Submit with: sbatch telepathy/submit_ddp_training.slurm
# Monitor with: squeue -u $USER
# Cancel with: scancel <job_id>
# =============================================================================

# Set working directory - MUST use /projects path
WORK_DIR="/projects/m000066/sujinesh/LatentWire"
cd "$WORK_DIR"

echo "=============================================================="
echo "SLURM DDP Training Job"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $SLURM_GPUS"
echo "Start time: $(date)"
echo "Working directory: $WORK_DIR"
echo "=============================================================="

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# DDP-specific environment variables
export MASTER_PORT=29500
export MASTER_ADDR=localhost

# Optimize for H100s
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export TORCH_DTYPE=bfloat16

# Enable NCCL optimizations for H100
export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

# Create runs directory if needed
mkdir -p runs figures

# Pull latest code
echo "Pulling latest code..."
git pull

# Configuration for training
SAMPLES=87599          # Full SQuAD dataset
EPOCHS=24              # Full training
BATCH_SIZE=64          # Will be divided by number of GPUs
OUTPUT_DIR="runs/ddp_h100_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "Training configuration:"
echo "  Dataset samples: $SAMPLES"
echo "  Epochs: $EPOCHS"
echo "  Base batch size: $BATCH_SIZE"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Run training with DDP using torchrun
echo "Starting DDP training on $SLURM_GPUS GPUs..."
echo "=============================================================="

torchrun \
    --nproc_per_node=$SLURM_GPUS \
    --master_port=$MASTER_PORT \
    --nnodes=1 \
    --node_rank=0 \
    latentwire/train.py \
    --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --qwen_id "Qwen/Qwen2.5-7B-Instruct" \
    --samples $SAMPLES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --grad_accum_steps 1 \
    --latent_len 32 \
    --d_z 256 \
    --encoder_type byte \
    --dataset squad \
    --warm_anchor_text "Answer: " \
    --first_token_ce_weight 0.5 \
    --elastic_gpu \
    --elastic_base_batch 64 \
    --elastic_target_util 0.85 \
    --save_dir "$OUTPUT_DIR" \
    --save_every 500 \
    --diagnostic_log "$OUTPUT_DIR/diagnostics.jsonl" \
    --save_training_stats \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32

echo "=============================================================="
echo "Training complete!"
echo ""

# Run evaluation on the final checkpoint
echo "Running evaluation on final checkpoint..."
python latentwire/eval.py \
    --ckpt "$OUTPUT_DIR" \
    --samples 200 \
    --max_new_tokens 12 \
    --dataset squad \
    --fresh_eval \
    --calibration embed_rms \
    --latent_anchor_mode text \
    --latent_anchor_text "Answer: " \
    --append_bos_after_prefix yes \
    > "$OUTPUT_DIR/eval_results.txt" 2>&1

# Push results back to git
echo ""
echo "Pushing results to git..."
git add -A
git commit -m "results: DDP training on ${SLURM_GPUS}x H100 GPUs (SLURM job $SLURM_JOB_ID)

Training configuration:
- Samples: $SAMPLES
- Epochs: $EPOCHS
- Batch size: $BATCH_SIZE (per GPU)
- GPUs: $SLURM_GPUS x H100
- DDP: Enabled with torchrun

Results saved to: $OUTPUT_DIR

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>" || true
git push || true

echo "=============================================================="
echo "Job completed at $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================================="

# Print summary statistics
if [ -f "$OUTPUT_DIR/training_stats.json" ]; then
    echo ""
    echo "Training Statistics:"
    python -c "import json; print(json.dumps(json.load(open('$OUTPUT_DIR/training_stats.json')), indent=2))"
fi

if [ -f "$OUTPUT_DIR/eval_results.txt" ]; then
    echo ""
    echo "Evaluation Results (last 50 lines):"
    tail -50 "$OUTPUT_DIR/eval_results.txt"
fi