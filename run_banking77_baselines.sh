#!/bin/bash
# run_banking77_baselines.sh
#
# Run Banking77 text baselines (Mistral and Llama direct classification)
# Expected runtime: ~30-45 minutes on 1 GPU

set -e

OUTPUT_DIR="${OUTPUT_DIR:-runs/banking77_baselines_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/banking77_baselines.log"

echo "============================================================"
echo "BANKING77 TEXT BASELINES"
echo "============================================================"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo "============================================================"

{
    python telepathy/eval_text_relay_baseline.py \
        --banking77 \
        --output_dir "$OUTPUT_DIR" \
        --num_samples 200 \
        --gpu 0
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "COMPLETE"
echo "============================================================"
echo "Results: $OUTPUT_DIR/banking77_baselines.json"
echo "Log: $LOG_FILE"
echo "============================================================"
