#!/usr/bin/env bash
set -e

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/llama_baselines}"
DATASETS="${DATASETS:-sst2 agnews}"
EVAL_SAMPLES="${EVAL_SAMPLES:-}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/llama_baselines_${TIMESTAMP}.log"

echo "Starting Llama 3.1 8B Zero-Shot Baseline Evaluation..."
echo "Model: $MODEL"
echo "Datasets: $DATASETS"
echo "Eval samples: ${EVAL_SAMPLES:-full dataset}"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""

# Build command
CMD="python telepathy/run_llama_baselines.py --datasets $DATASETS --output_dir $OUTPUT_DIR --model $MODEL"

# Add eval_samples if specified
if [ -n "$EVAL_SAMPLES" ]; then
    CMD="$CMD --eval_samples $EVAL_SAMPLES"
fi

# Add save_predictions flag
CMD="$CMD --save_predictions"

# Run command with tee to capture ALL output
{
    eval $CMD
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "Complete! Results saved to:"
echo "  - $OUTPUT_DIR/llama_baseline_results_*.json"
echo "  - $LOG_FILE"
