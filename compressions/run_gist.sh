#!/usr/bin/env bash
#
# Run Gist Tokens Faithful Reproduction
#
# Usage:
#   bash run_gist.sh test       # Quick test (100 samples)
#   bash run_gist.sh validate   # Validation (2K samples)
#   bash run_gist.sh full       # Full reproduction (52K samples)

set -e

# Configuration
SCRIPT="compressions/train_gist_faithful.py"
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
NUM_GIST_TOKENS=1
BATCH_SIZE=1  # REQUIRED (per paper)
LR=1e-4

# Set PYTHONPATH
export PYTHONPATH=.

# Parse mode
MODE=${1:-validate}

case $MODE in
    test)
        echo "Running QUICK TEST (100 samples, 1 epoch)"
        SAMPLES=100
        EPOCHS=1
        OUTPUT_DIR="runs/gist_test"
        ;;
    validate)
        echo "Running VALIDATION (2K samples, 2 epochs)"
        SAMPLES=2000
        EPOCHS=2
        OUTPUT_DIR="runs/gist_validate"
        ;;
    full)
        echo "Running FULL REPRODUCTION (52K samples, 3 epochs)"
        SAMPLES=52000
        EPOCHS=3
        OUTPUT_DIR="runs/gist_full"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: bash run_gist.sh [test|validate|full]"
        exit 1
        ;;
esac

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/train_${TIMESTAMP}.log"

echo "Starting Gist training..."
echo "Log file: $LOG_FILE"
echo ""

# Run training with tee to capture output
{
    python $SCRIPT \
        --model_id "$MODEL" \
        --num_gist_tokens $NUM_GIST_TOKENS \
        --samples $SAMPLES \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --output_dir "$OUTPUT_DIR" \
        --device cuda:0
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "Complete! Results saved to:"
echo "  - $OUTPUT_DIR/pytorch_model.bin"
echo "  - $OUTPUT_DIR/metrics.json"
echo "  - $LOG_FILE"
