#!/usr/bin/env bash
set -euo pipefail

# VALIDATION RUN: Re-run experiment 1c with fixed gold answer extraction
# Purpose: Get TRUE accuracy numbers after fixing the extraction bug
# Expected: Peak accuracy at step 750 may be significantly higher than 21.5%

# Load HPC modules if available
if command -v module >/dev/null 2>&1; then
    module purge
    module load gcc/13.1.0
    module load conda/24.3.0-0
    module load stockcuda/12.6.2
    module load cudnn/cuda12/9.3.0.75
fi

echo "=========================================="
echo "1C VALIDATION RUN (Fixed Gold Extraction)"
echo "=========================================="
echo "Start time: $(date)"
echo "Expected runtime: ~2 hours"
echo ""
echo "Changes from original run:"
echo "  - Gold answer extraction fixed (now extracts from test answer only)"
echo "  - Early stopping disabled (patience=999, will run full 2000 steps)"
echo "  - Will reveal TRUE accuracy numbers"
echo ""

# Base configuration (same as ablation script)
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY_SCRIPT=paper_writing/cross_attention.py
SOURCE_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
TARGET_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
PER_DEVICE_BATCH=4
EVAL_EVERY=250
EVAL_SAMPLES=200
MAX_NEW_TOKENS=512

# Create output directory
OUTPUT_DIR="paper_writing/runs/1c_validation_$(date +"%Y%m%d_%H%M%S")"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/train.log"

echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""

# Use random port to avoid conflicts
RANDOM_PORT=$((29500 + RANDOM % 1000))

echo "Starting training..."
{
    torchrun --standalone --nproc_per_node=4 --master_port "$RANDOM_PORT" "$PY_SCRIPT" \
        --source_model "$SOURCE_MODEL" \
        --target_model "$TARGET_MODEL" \
        --per_device_batch "$PER_DEVICE_BATCH" \
        --eval_every "$EVAL_EVERY" \
        --eval_samples "$EVAL_SAMPLES" \
        --eval_batch_size 36 \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --bf16 \
        --no_compile \
        --save_path "$OUTPUT_DIR/checkpoint.pt" \
        --log_dir "$OUTPUT_DIR" \
        --dataset gsm8k \
        --bridge dit \
        --lr 1e-4 \
        --dit_dim 512 \
        --soft_tokens 64 \
        --dit_depth 6 \
        --dit_heads 8 \
        --dit_steps_train 2 \
        --dit_steps_eval 4 \
        --dit_dropout 0.1 \
        --dit_pool attn \
        --dit_loss_weight 0.1 \
        --weight_decay 0.01 \
        --train_steps 2000 \
        --warmup_steps 200 \
        --info_nce_weight 0.05 \
        --early_stop_patience 999 \
        --seed 1234
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "VALIDATION RUN COMPLETE"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Results:"
grep "Final.*acc:" "$LOG_FILE" | tail -1

echo ""
echo "Accuracy progression (all eval steps):"
grep "Eval.*Bridged acc:" "$LOG_FILE" | \
    awk '{print $2, $3, $4, $5, $6, $7, $8, $9, $10}'

echo ""
echo "Peak bridged accuracy:"
grep "Eval.*Bridged acc:" "$LOG_FILE" | \
    awk -F'Bridged acc: ' '{print $2}' | \
    sort -rn | head -1 | \
    awk '{printf "  %.3f (%.1f%%)\n", $1, $1 * 100}'

echo ""
echo "Full results saved to:"
echo "  - $OUTPUT_DIR"
echo "  - $LOG_FILE"
echo ""
echo "Next steps:"
echo "  1. Compare peak accuracy to original 21.5%"
echo "  2. Analyze eval_samples_step_*.jsonl files for gold answer diversity"
echo "  3. If accuracy is high (>40%), focus on stabilization"
echo "  4. If accuracy is low (<10%), implement architectural changes"
