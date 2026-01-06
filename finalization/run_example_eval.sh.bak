#!/bin/bash
# Example script showing how to run evaluation scripts standalone

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "=========================================="
echo "Example Evaluation Script Runner"
echo "=========================================="
echo "Directory: $SCRIPT_DIR"
echo ""

# Function to show usage
show_usage() {
    echo "Usage: $0 <eval_type> <checkpoint_path> [options]"
    echo ""
    echo "Available evaluation types:"
    echo "  main     - Run main eval.py script"
    echo "  sst2     - Run SST-2 sentiment evaluation"
    echo "  agnews   - Run AG News classification evaluation"
    echo "  gsm8k    - Run GSM8K math evaluation"
    echo ""
    echo "Example:"
    echo "  $0 main /path/to/checkpoint --samples 100 --dataset squad"
    echo "  $0 sst2 /path/to/checkpoint --num_samples 100"
    echo ""
    exit 1
}

# Check arguments
if [ $# -lt 2 ]; then
    show_usage
fi

EVAL_TYPE="$1"
CHECKPOINT="$2"
shift 2  # Remove first two arguments, keep the rest

# Create output directory
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/eval_results}"
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

case "$EVAL_TYPE" in
    main)
        echo "Running main evaluation..."
        LOG_FILE="$OUTPUT_DIR/eval_main_${TIMESTAMP}.log"

        {
            python3 "${SCRIPT_DIR}/latentwire/eval.py" \
                --ckpt "$CHECKPOINT" \
                --samples 100 \
                --dataset squad \
                --max_new_tokens 12 \
                --sequential_eval \
                --calibration embed_rms \
                --latent_anchor_text "Answer: " \
                "$@"
        } 2>&1 | tee "$LOG_FILE"

        echo ""
        echo "Results saved to: $LOG_FILE"
        ;;

    sst2)
        echo "Running SST-2 evaluation..."
        LOG_FILE="$OUTPUT_DIR/eval_sst2_${TIMESTAMP}.log"

        {
            python3 "${SCRIPT_DIR}/latentwire/eval_sst2.py" \
                --checkpoint "$CHECKPOINT" \
                --num_samples 100 \
                --output_dir "$OUTPUT_DIR/sst2_${TIMESTAMP}" \
                "$@"
        } 2>&1 | tee "$LOG_FILE"

        echo ""
        echo "Results saved to: $OUTPUT_DIR/sst2_${TIMESTAMP}/"
        ;;

    agnews)
        echo "Running AG News evaluation..."
        LOG_FILE="$OUTPUT_DIR/eval_agnews_${TIMESTAMP}.log"

        {
            python3 "${SCRIPT_DIR}/latentwire/eval_agnews.py" \
                --checkpoint "$CHECKPOINT" \
                --num_samples 100 \
                --output_dir "$OUTPUT_DIR/agnews_${TIMESTAMP}" \
                "$@"
        } 2>&1 | tee "$LOG_FILE"

        echo ""
        echo "Results saved to: $OUTPUT_DIR/agnews_${TIMESTAMP}/"
        ;;

    gsm8k)
        echo "Running GSM8K evaluation..."
        LOG_FILE="$OUTPUT_DIR/eval_gsm8k_${TIMESTAMP}.log"

        {
            python3 "${SCRIPT_DIR}/latentwire/gsm8k_eval.py" \
                --checkpoint "$CHECKPOINT" \
                --num_samples 50 \
                --output_dir "$OUTPUT_DIR/gsm8k_${TIMESTAMP}" \
                "$@"
        } 2>&1 | tee "$LOG_FILE"

        echo ""
        echo "Results saved to: $OUTPUT_DIR/gsm8k_${TIMESTAMP}/"
        ;;

    *)
        echo "Error: Unknown evaluation type '$EVAL_TYPE'"
        echo ""
        show_usage
        ;;
esac

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="