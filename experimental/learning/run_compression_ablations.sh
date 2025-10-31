#!/usr/bin/env bash
set -e

# ============================================================================
# SQuAD Compression Ablations Runner
# Tests different compression ratios (32, 64, 128), architectures, and losses
# ============================================================================

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/compression_ablations}"
PYTHON_ENV="${PYTHON_ENV:-../../venv_arm64/bin/activate}"

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# For HPC: Use CUDA
export USE_CUDA=1
export DISABLE_FLASH_ATTENTION=0

# For Mac: Use MPS
if [[ "$(uname)" == "Darwin" ]]; then
    export USE_MPS=1
    export USE_CUDA=0
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/compression_ablations_${TIMESTAMP}.log"

echo "Starting SQuAD Compression Ablations..."
echo "Log file: $LOG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Function to run single ablation
run_single_ablation() {
    local TARGET_LENGTH=$1
    local ARCHITECTURE=$2
    local GPU=$3
    local SAMPLES=${4:-10000}

    echo "Running ablation: M=${TARGET_LENGTH}, arch=${ARCHITECTURE}, GPU=${GPU}"

    {
        python compression_ablations.py \
            --target_length "$TARGET_LENGTH" \
            --architecture "$ARCHITECTURE" \
            --gpu "$GPU" \
            --num_train_samples "$SAMPLES" \
            --num_eval_samples 1000 \
            --epochs 10 \
            --batch_size 8
    } 2>&1 | tee -a "$LOG_FILE"
}

# Parse command line arguments
MODE="${1:-single}"  # single, parallel, or full
TARGET_LENGTH="${2:-64}"
ARCHITECTURE="${3:-cross_attention}"
GPU="${4:-0}"

if [[ "$MODE" == "single" ]]; then
    # Run single configuration
    echo "Running single ablation: M=$TARGET_LENGTH, $ARCHITECTURE on GPU $GPU"
    {
        python compression_ablations.py \
            --target_length "$TARGET_LENGTH" \
            --architecture "$ARCHITECTURE" \
            --gpu "$GPU" \
            --num_train_samples 10000 \
            --num_eval_samples 1000 \
            --epochs 10 \
            --batch_size 8
    } 2>&1 | tee "$LOG_FILE"

elif [[ "$MODE" == "parallel" ]]; then
    # Run 4 configurations in parallel (one per GPU)
    echo "Running parallel ablations on 4 GPUs..."

    # Start background processes
    run_single_ablation 32 cross_attention 0 10000 &
    PID1=$!

    run_single_ablation 64 cross_attention 1 10000 &
    PID2=$!

    run_single_ablation 128 cross_attention 2 10000 &
    PID3=$!

    run_single_ablation 64 conv 3 10000 &
    PID4=$!

    # Wait for all to complete
    echo "Waiting for parallel jobs to complete..."
    wait $PID1 $PID2 $PID3 $PID4

    echo "All parallel ablations complete!"

elif [[ "$MODE" == "full" ]]; then
    # Run full ablation suite
    echo "Running full ablation suite (27 configurations)..."
    echo "This will take approximately 54 hours with 4 GPUs"

    {
        python compression_ablations.py --run_all
    } 2>&1 | tee "$LOG_FILE"

elif [[ "$MODE" == "test" ]]; then
    # Quick test run with minimal samples
    echo "Running test ablation with reduced samples..."
    {
        python compression_ablations.py \
            --target_length 64 \
            --architecture cross_attention \
            --gpu 0 \
            --num_train_samples 100 \
            --num_eval_samples 50 \
            --epochs 2 \
            --batch_size 4
    } 2>&1 | tee "$LOG_FILE"

else
    echo "Usage: $0 [mode] [target_length] [architecture] [gpu]"
    echo ""
    echo "Modes:"
    echo "  single    - Run single configuration (default)"
    echo "  parallel  - Run 4 configurations in parallel"
    echo "  full      - Run all 27 ablations"
    echo "  test      - Quick test with minimal samples"
    echo ""
    echo "Examples:"
    echo "  $0 single 64 cross_attention 0    # Single run"
    echo "  $0 parallel                        # Run 4 in parallel"
    echo "  $0 full                            # Run all ablations"
    echo "  $0 test                            # Quick test"
    exit 1
fi

echo ""
echo "Complete! Results saved to:"
echo "  - $OUTPUT_DIR/"
echo "  - $LOG_FILE"

# Generate summary if results exist
if [[ -f "$OUTPUT_DIR/metrics.json" ]]; then
    echo ""
    echo "Summary of results:"
    python -c "
import json
with open('$OUTPUT_DIR/metrics.json') as f:
    metrics = json.load(f)
    print(f'  Best F1: {max(metrics[\"eval_f1\"]):.4f}')
    print(f'  Best EM: {max(metrics[\"eval_em\"]):.4f}')
    print(f'  Compression: {metrics[\"compression_ratio\"]:.1f}x')
"
fi