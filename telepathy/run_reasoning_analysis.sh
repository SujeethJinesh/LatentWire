#!/usr/bin/env bash
# telepathy/run_reasoning_analysis.sh
#
# Local wrapper for reasoning failure analysis
# For full analysis on HPC, use submit_reasoning_analysis.slurm
#
# Usage:
#   bash telepathy/run_reasoning_analysis.sh
#
# Or with custom checkpoint:
#   CHECKPOINT=runs/my_bridge/bridge.pt bash telepathy/run_reasoning_analysis.sh

set -e

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/reasoning_analysis_$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT="${CHECKPOINT:-runs/gsm8k_bridge/bridge.pt}"
CHECKPOINT_LAYER16="${CHECKPOINT_LAYER16:-}"  # Optional
SOURCE_LAYER="${SOURCE_LAYER:-31}"
SOFT_TOKENS="${SOFT_TOKENS:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"  # Reduced for local testing

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set up logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/analysis_${TIMESTAMP}.log"

echo "=============================================================="
echo "Telepathy Reasoning Failure Analysis (Local)"
echo "=============================================================="
echo "Checkpoint: $CHECKPOINT"
if [ -n "$CHECKPOINT_LAYER16" ]; then
    echo "Layer 16 checkpoint: $CHECKPOINT_LAYER16"
fi
echo "Source layer: $SOURCE_LAYER"
echo "Soft tokens: $SOFT_TOKENS"
echo "Samples: $NUM_SAMPLES"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "=============================================================="
echo ""

# Build command
CMD="python telepathy/analyze_reasoning_failure.py \
    --checkpoint $CHECKPOINT \
    --source_layer $SOURCE_LAYER \
    --soft_tokens $SOFT_TOKENS \
    --num_samples $NUM_SAMPLES \
    --output_dir $OUTPUT_DIR \
    --bf16"

# Add Layer 16 checkpoint if provided
if [ -n "$CHECKPOINT_LAYER16" ]; then
    CMD="$CMD --checkpoint_layer16 $CHECKPOINT_LAYER16"
fi

# Run with logging
{
    echo "Starting analysis at $(date)"
    echo ""

    $CMD

    echo ""
    echo "Analysis completed at $(date)"
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "=============================================================="
echo "Complete! Results saved to:"
echo "  - $OUTPUT_DIR/results.json"
echo "  - $OUTPUT_DIR/*.png (visualizations)"
echo "  - $LOG_FILE"
echo "=============================================================="
