#!/bin/bash
# run_trec_baselines.sh
#
# Run TREC text baselines and text-relay to complete comparison table
# Expected runtime: ~20-30 minutes on 1 GPU

set -e

OUTPUT_DIR="${OUTPUT_DIR:-runs/trec_baselines_$(date +%Y%m%d_%H%M%S)}"
GPU="${GPU:-0}"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/trec_baselines.log"

echo "============================================================"
echo "TREC BASELINES (Text + Text-Relay)"
echo "============================================================"
echo "Part 1: Direct text baselines (Mistral, Llama)"
echo "Part 2: Text-relay (Llama summarizes -> Mistral classifies)"
echo "GPU: $GPU"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo "============================================================"

{
    echo "=== PART 1: TREC TEXT BASELINES ==="
    python telepathy/eval_text_relay_baseline.py \
        --trec \
        --output_dir "$OUTPUT_DIR" \
        --num_samples 200 \
        --gpu "$GPU"

    echo ""
    echo "=== PART 2: TREC TEXT-RELAY ==="
    python telepathy/eval_text_relay_baseline.py \
        --trec_relay \
        --output_dir "$OUTPUT_DIR" \
        --num_samples 200 \
        --gpu "$GPU"

} 2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "COMPLETE"
echo "============================================================"
echo "Results:"
echo "  - $OUTPUT_DIR/trec_baselines.json (direct text)"
echo "  - $OUTPUT_DIR/trec_relay_results.json (text-relay)"
echo "Log: $LOG_FILE"
echo "============================================================"
