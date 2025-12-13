#!/bin/bash
# run_banking77_relay.sh
#
# Run Banking77 text-relay baseline (Llama summarizes → Mistral classifies)
# This completes the comparison table for Banking77
# Expected runtime: ~20-30 minutes on 1 GPU

set -e

OUTPUT_DIR="${OUTPUT_DIR:-runs/banking77_relay_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/banking77_relay.log"

echo "============================================================"
echo "BANKING77 TEXT-RELAY BASELINE"
echo "============================================================"
echo "Pipeline: Llama summarizes → text → Mistral classifies"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo "============================================================"

{
    python telepathy/eval_text_relay_baseline.py \
        --banking77_relay \
        --output_dir "$OUTPUT_DIR" \
        --num_samples 200 \
        --gpu 0
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "COMPLETE"
echo "============================================================"
echo "Results: $OUTPUT_DIR/banking77_relay_results.json"
echo "Log: $LOG_FILE"
echo "============================================================"
