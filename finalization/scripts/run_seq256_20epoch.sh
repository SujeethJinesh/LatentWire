#!/usr/bin/env bash
set -e

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/seq256_r16_l8_20epoch}"

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/train_${TIMESTAMP}.log"
DIAGNOSTIC_LOG="$OUTPUT_DIR/diagnostics.jsonl"

echo "================================================"
echo "Sequence Compression: 20 Epoch Extended Training"
echo "================================================"
echo "Configuration: seq256_r16_l8 (best from sweep)"
echo "  Sequence: 256 (compression: 1.17x)"
echo "  Pooling: learned_attention"
echo "  LoRA: r=16, alpha=32, first 8 layers"
echo "  Epochs: 20"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "Diagnostic log: $DIAGNOSTIC_LOG"
echo "================================================"
echo ""

# Run training with tee to capture ALL output
{
    python train_sequence_compression.py \
        --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --target_sequence_length 256 \
        --source_length 300 \
        --pooling_method learned_attention \
        --use_lora \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_layers 8 \
        --dataset squad \
        --samples 10000 \
        --eval_samples 100 \
        --epochs 20 \
        --batch_size 48 \
        --lr 5e-4 \
        --max_new_tokens 12 \
        --save_dir "$OUTPUT_DIR" \
        --diagnostic_log "$DIAGNOSTIC_LOG"
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "================================================"
echo "TRAINING COMPLETE"
echo "================================================"
echo "Results saved to:"
echo "  - $OUTPUT_DIR/best_checkpoint.pt"
echo "  - $LOG_FILE"
echo "  - $DIAGNOSTIC_LOG"
echo ""

# Print final results summary
if [ -f "$DIAGNOSTIC_LOG" ]; then
    echo "Final epoch results:"
    tail -n 1 "$DIAGNOSTIC_LOG" | python -m json.tool
    echo ""
fi
