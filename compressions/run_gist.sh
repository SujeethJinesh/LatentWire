#!/usr/bin/env bash
#
# Run Gist Tokens Faithful Reproduction with Multi-GPU Support
#
# Usage:
#   bash run_gist.sh test       # Quick test (100 samples)
#   bash run_gist.sh validate   # Validation (2K samples)
#   bash run_gist.sh full       # Full reproduction (52K samples)

set -e

# Configuration
SCRIPT="compressions/train_gist_faithful.py"
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"  # Instruct model - already trained for instruction following
NUM_GIST_TOKENS=5  # Paper uses 5-10 tokens for effective compression
BATCH_SIZE=12  # Per-GPU batch size (conservative for H100 80GB)
GRAD_ACCUM_STEPS=2  # Gradient accumulation (effective batch size multiplier)
LR=1e-5  # Conservative for full 52K sample run (paper uses 2e-5, we used 5e-5 for 2K validation)
WARMUP_RATIO=0.03  # Match paper: 3% warmup
LR_SCHEDULER="cosine"  # Match paper: cosine decay
NUM_GPUS=4    # Use all 4 GPUs

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
        echo "Running VALIDATION (2K samples, 5 epochs)"
        SAMPLES=2000
        EPOCHS=5
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

echo "Starting Gist training on $NUM_GPUS GPUs..."
echo "Per-GPU batch size: $BATCH_SIZE"
echo "Gradient accumulation steps: $GRAD_ACCUM_STEPS"
echo "Effective batch size: $((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM_STEPS))"
echo "Log file: $LOG_FILE"
echo ""

# Run training with torchrun for multi-GPU (DDP)
{
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        $SCRIPT \
            --model_id "$MODEL" \
            --num_gist_tokens $NUM_GIST_TOKENS \
            --samples $SAMPLES \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
            --lr $LR \
            --warmup_ratio $WARMUP_RATIO \
            --lr_scheduler $LR_SCHEDULER \
            --output_dir "$OUTPUT_DIR" \
            --device auto
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "Training complete! Results saved to:"
echo "  - $OUTPUT_DIR/pytorch_model.bin"
echo "  - $OUTPUT_DIR/metrics.json"
echo "  - $LOG_FILE"
echo ""

# Run evaluation with baselines
echo "Starting evaluation with baselines..."
EVAL_LOG_FILE="$OUTPUT_DIR/eval_${TIMESTAMP}.log"

{
    python compressions/eval_gist.py \
        --checkpoint "$OUTPUT_DIR" \
        --samples 200 \
        --batch_size 16 \
        --max_new_tokens 128 \
        --device cuda:0
} 2>&1 | tee "$EVAL_LOG_FILE"

echo ""
echo "="$(printf '=%.0s' {1..78})
echo "COMPLETE! All results saved to:"
echo "  Training:"
echo "    - $OUTPUT_DIR/pytorch_model.bin"
echo "    - $OUTPUT_DIR/metrics.json"
echo "    - $LOG_FILE"
echo "  Evaluation:"
echo "    - $OUTPUT_DIR/eval_results.json"
echo "    - $OUTPUT_DIR/sample_outputs.json"
echo "    - $EVAL_LOG_FILE"
echo ""
echo "Configuration:"
echo "  GPUs: $NUM_GPUS"
echo "  Per-GPU batch size: $BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM_STEPS"
echo "  Effective batch size: $((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM_STEPS))"
echo "="$(printf '=%.0s' {1..78})
