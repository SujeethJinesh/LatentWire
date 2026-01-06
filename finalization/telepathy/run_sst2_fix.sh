#!/usr/bin/env bash
set -e

OUTPUT_DIR="runs/arxiv_sst2_fix_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/run.log"

echo "Running SST-2 fix experiment..."
echo "Output: $OUTPUT_DIR"

{
    PYTHONPATH=. python telepathy/run_unified_comparison.py \
        --datasets sst2 \
        --output_dir "$OUTPUT_DIR" \
        --train_steps 2000 \
        --eval_samples 200 \
        --seed 42
} 2>&1 | tee "$LOG_FILE"

echo "Done! Results in $OUTPUT_DIR"
