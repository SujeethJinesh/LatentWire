#!/usr/bin/env bash
set -e

# Mixed Precision Training Script for H100 GPUs
# This script demonstrates how to use the new mixed precision features
# for maximum performance on H100 Tensor Cores

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/mixed_precision_experiment}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"  # bf16 recommended for H100
BATCH_SIZE="${BATCH_SIZE:-64}"  # Safe for single GPU or per-GPU in DDP (81.3% memory)
GRAD_ACCUM="${GRAD_ACCUM:-1}"  # Use 4 for effective batch=256 on single GPU

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use all 4 H100s

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/mixed_precision_${TIMESTAMP}.log"

echo "=========================================="
echo "Mixed Precision Training on H100 GPUs"
echo "=========================================="
echo "Precision: $MIXED_PRECISION"
echo "Batch size: $BATCH_SIZE"
echo "Gradient accumulation: $GRAD_ACCUM"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""
echo "Expected benefits:"
echo "  - 2-3x training speedup"
echo "  - 30-50% memory savings"
echo "  - Larger effective batch sizes"
echo "=========================================="
echo ""

# Run training with mixed precision
{
    python latentwire/train.py \
        --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --qwen_id "Qwen/Qwen2.5-7B-Instruct" \
        --mixed_precision "$MIXED_PRECISION" \
        --amp_opt_level "O1" \
        --samples 10000 \
        --epochs 3 \
        --batch_size "$BATCH_SIZE" \
        --grad_accum_steps "$GRAD_ACCUM" \
        --latent_len 32 \
        --d_z 256 \
        --encoder_type byte \
        --dataset squad \
        --sequential_models \
        --warm_anchor_text "Answer: " \
        --first_token_ce_weight 0.5 \
        --k_ce_weight 1.0 \
        --K 4 \
        --lr 1e-4 \
        --save_dir "$OUTPUT_DIR/checkpoints" \
        --save_every 500 \
        --grad_ckpt \
        --max_grad_norm 1.0

    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "=========================================="

    # Run evaluation to verify quality
    echo ""
    echo "Running evaluation on latest checkpoint..."

    python latentwire/eval.py \
        --ckpt "$OUTPUT_DIR/checkpoints" \
        --samples 100 \
        --max_new_tokens 12 \
        --dataset squad \
        --sequential_eval \
        --fresh_eval \
        --calibration embed_rms \
        --latent_anchor_mode text \
        --latent_anchor_text "Answer: " \
        --append_bos_after_prefix yes \
        --output_dir "$OUTPUT_DIR/eval"

} 2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Complete! Results saved to:"
echo "  - Training log: $LOG_FILE"
echo "  - Checkpoints: $OUTPUT_DIR/checkpoints"
echo "  - Evaluation: $OUTPUT_DIR/eval"
echo "=========================================="
echo ""
echo "Performance Tips:"
echo "  1. Use bf16 for H100 (better stability than fp16)"
echo "  2. Increase batch size by 2-3x vs fp32"
echo "  3. Monitor GPU utilization with nvidia-smi"
echo "  4. Check for overflow warnings in fp16 mode"
echo "=========================================="