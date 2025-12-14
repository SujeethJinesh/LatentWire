#!/bin/bash
# run_latency_benchmark.sh
#
# Measure inference latency for Bridge vs Text-Relay vs Direct Text
#
# Usage: git pull && PYTHONPATH=. bash run_latency_benchmark.sh
#
# Expected output:
#   - Bridge: ~50-100ms (8 soft tokens, no generation)
#   - Text-Relay: ~500-1000ms (generation required)
#   - Direct Text: ~100-200ms (Mistral only)
#
# Expected speedup: Bridge is 5-10x faster than Text-Relay

set -e

OUTPUT_BASE="${OUTPUT_BASE:-runs}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_BASE}/latency_benchmark_${TIMESTAMP}"
LOG_FILE="${OUTPUT_DIR}/latency_benchmark.log"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "LATENCY BENCHMARK"
echo "=============================================="
echo "Output: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""

{
    echo "[$(date)] Starting latency benchmark..."
    echo ""

    # Find the SST-2 checkpoint (if available)
    SST2_CKPT=""
    if [ -d "preserved_data/phase21_overnight_sst2_agnews_2025-12-12" ]; then
        SST2_CKPT=$(find preserved_data/phase21_overnight_sst2_agnews_2025-12-12 -name "best_checkpoint.pt" | head -1)
    fi

    if [ -n "$SST2_CKPT" ] && [ -f "$SST2_CKPT" ]; then
        echo "Found SST-2 checkpoint: $SST2_CKPT"
        python telepathy/benchmark_latency.py \
            --checkpoint "$SST2_CKPT" \
            --num_trials 50 \
            --output_dir "$OUTPUT_DIR" \
            --gpu 0
    else
        echo "No checkpoint found, running without Bridge timing"
        python telepathy/benchmark_latency.py \
            --num_trials 50 \
            --output_dir "$OUTPUT_DIR" \
            --gpu 0
    fi

    echo ""
    echo "=============================================="
    echo "COMPLETE"
    echo "=============================================="
    echo ""
    echo "Results saved to: $OUTPUT_DIR/latency_benchmark.json"
    echo ""

    # Print summary
    if [ -f "$OUTPUT_DIR/latency_benchmark.json" ]; then
        echo "SUMMARY:"
        python -c "
import json
with open('$OUTPUT_DIR/latency_benchmark.json') as f:
    results = json.load(f)
print(f\"Direct Text: {results['direct_text']['total_ms']:.1f}ms\")
print(f\"Text-Relay: {results['text_relay']['total_ms']:.1f}ms\")
if 'bridge' in results:
    print(f\"Bridge: {results['bridge']['total_ms']:.1f}ms\")
    speedup = results['text_relay']['total_ms'] / results['bridge']['total_ms']
    print(f'\\nBridge is {speedup:.1f}x faster than Text-Relay')
"
    fi

    echo ""
    echo "[$(date)] Done!"

} 2>&1 | tee "$LOG_FILE"

echo ""
echo "Full log saved to: $LOG_FILE"
