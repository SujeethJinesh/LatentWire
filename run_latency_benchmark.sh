#!/bin/bash
# run_latency_benchmark.sh
#
# Measure inference latency for Bridge vs Text-Relay vs Direct Text
#
# Usage: git pull && PYTHONPATH=. bash run_latency_benchmark.sh
#
# This script will:
#   1. Train a quick SST-2 Bridge checkpoint (~3.5 min) if none exists
#   2. Run latency benchmark with the checkpoint
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
TRAIN_DIR="${OUTPUT_DIR}/sst2_for_latency"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "LATENCY BENCHMARK (with Bridge Training)"
echo "=============================================="
echo "Output: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""

{
    echo "[$(date)] Starting latency benchmark..."
    echo ""

    # Find existing SST-2 checkpoint (if available)
    # Note: train_telepathy_sst2.py saves as bridge_sst2.pt
    SST2_CKPT=""
    for dir in "preserved_data/phase21_overnight_sst2_agnews_2025-12-12" "runs" "."; do
        if [ -d "$dir" ]; then
            # Try bridge_sst2.pt first (what train_telepathy_sst2.py creates)
            FOUND=$(find "$dir" -name "bridge_sst2.pt" 2>/dev/null | head -1)
            if [ -z "$FOUND" ]; then
                # Fallback to best_checkpoint.pt
                FOUND=$(find "$dir" -name "best_checkpoint.pt" -path "*sst2*" 2>/dev/null | head -1)
            fi
            if [ -n "$FOUND" ] && [ -f "$FOUND" ]; then
                SST2_CKPT="$FOUND"
                break
            fi
        fi
    done

    # If no checkpoint found, train one quickly
    if [ -z "$SST2_CKPT" ] || [ ! -f "$SST2_CKPT" ]; then
        echo "=============================================="
        echo "PHASE 1: Training SST-2 Bridge (~3.5 min)"
        echo "=============================================="
        echo ""

        CUDA_VISIBLE_DEVICES=0 python telepathy/train_telepathy_sst2.py \
            --source_layer 31 \
            --soft_tokens 8 \
            --steps 2000 \
            --batch_size 8 \
            --lr 1e-4 \
            --diversity_weight 0.1 \
            --output_dir "$TRAIN_DIR"

        # train_telepathy_sst2.py saves as bridge_sst2.pt, not best_checkpoint.pt
        SST2_CKPT="${TRAIN_DIR}/bridge_sst2.pt"
        echo ""
        echo "Training complete. Checkpoint: $SST2_CKPT"
    else
        echo "Found existing SST-2 checkpoint: $SST2_CKPT"
    fi

    echo ""
    echo "=============================================="
    echo "PHASE 2: Running Latency Benchmark"
    echo "=============================================="
    echo ""

    python telepathy/benchmark_latency.py \
        --checkpoint "$SST2_CKPT" \
        --num_trials 50 \
        --output_dir "$OUTPUT_DIR" \
        --gpu 0

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
