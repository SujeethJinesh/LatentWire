#!/usr/bin/env bash
# scripts/run_optimized_training.sh
# Demonstrate optimized dataloader usage with LatentWire training

set -e

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/optimized_dataloader_test}"
DATASET="${DATASET:-squad}"
SAMPLES="${SAMPLES:-1000}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/training_optimized_${TIMESTAMP}.log"

echo "=========================================="
echo "LatentWire Training with Optimized DataLoader"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Samples: $SAMPLES"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Output dir: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""

# Run training with optimized dataloader
echo "Running training with optimized DataLoader..."
{
    python latentwire/train.py \
        --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --qwen_id "Qwen/Qwen2.5-7B-Instruct" \
        --models "llama" \
        --samples "$SAMPLES" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --latent_len 32 \
        --d_z 256 \
        --encoder_type byte \
        --dataset "$DATASET" \
        --sequential_models \
        --warm_anchor_text "Answer: " \
        --first_token_ce_weight 0.5 \
        --output_dir "$OUTPUT_DIR" \
        --use_optimized_dataloader \
        --num_dataloader_workers 4 \
        --dataloader_prefetch_factor 2 \
        --dataloader_cache_tokenization \
        --dataloader_pin_memory \
        --lr 1e-3 \
        --max_grad_norm 1.0
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "Training complete! Results saved to:"
echo "  - Checkpoints: $OUTPUT_DIR/"
echo "  - Log file: $LOG_FILE"

# Extract and display performance metrics
echo ""
echo "Performance Summary:"
echo "==================="
grep -E "(batch/s|ms/batch|GPU idle|throughput)" "$LOG_FILE" | tail -20 || true

echo ""
echo "To compare with original dataloader, run:"
echo "  bash scripts/run_optimized_training.sh  # With optimization"
echo "  Then run without the --use_optimized_dataloader flag to compare"