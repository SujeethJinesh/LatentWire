#!/usr/bin/env bash
set -e

# =============================================================================
# FULL ARXIV EXPERIMENTS WITH SST-2 FIX
# =============================================================================
# Runs all 3 datasets (SST-2, AG News, TREC) with the fixed Bridge code
#
# SST-2 Fixes applied:
# 1. Disabled diversity loss for binary classification (was counterproductive)
# 2. Added class-balanced sampling for binary tasks
# 3. Adaptive hyperparameters: 4 tokens, lr=5e-4, 4000 steps for binary
# 4. Improved prompt format: "Classify sentiment as positive or negative:"
# =============================================================================

OUTPUT_DIR="runs/arxiv_experiments_fixed_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/run.log"

echo "=============================================================="
echo "ARXIV EXPERIMENTS WITH SST-2 FIX"
echo "=============================================================="
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""
echo "SST-2 Fixes:"
echo "  - Diversity loss disabled for binary classification"
echo "  - Class-balanced sampling enabled"
echo "  - Adaptive hyperparams: 4 tokens, lr=5e-4, 4000 steps"
echo "  - Improved prompt format"
echo "=============================================================="

{
    echo "Starting experiments at $(date)"
    echo ""

    # Run unified comparison on all datasets
    PYTHONPATH=. python telepathy/run_unified_comparison.py \
        --datasets sst2 agnews trec \
        --output_dir "$OUTPUT_DIR/comparison" \
        --train_steps 2000 \
        --eval_samples 200 \
        --seed 42

    echo ""
    echo "=============================================================="
    echo "MAIN EXPERIMENTS COMPLETE"
    echo "=============================================================="

    # Run latency benchmark if it exists
    if [ -f "telepathy/batched_latency_benchmark.py" ]; then
        echo ""
        echo "Running batched latency benchmark..."
        mkdir -p "$OUTPUT_DIR/latency"
        PYTHONPATH=. python telepathy/batched_latency_benchmark.py \
            --output_dir "$OUTPUT_DIR/latency" \
            2>/dev/null || echo "Latency benchmark skipped (optional)"
    fi

    echo ""
    echo "=============================================================="
    echo "ALL EXPERIMENTS COMPLETE"
    echo "=============================================================="
    echo "Finished at $(date)"
    echo ""
    echo "Results saved to:"
    echo "  - $OUTPUT_DIR/comparison/"
    echo "  - $OUTPUT_DIR/latency/ (if available)"
    echo ""

    # Print summary if results file exists
    RESULTS_FILE=$(ls -t "$OUTPUT_DIR/comparison"/unified_results_*.json 2>/dev/null | head -1)
    if [ -n "$RESULTS_FILE" ]; then
        echo "=============================================================="
        echo "RESULTS SUMMARY"
        echo "=============================================================="
        python -c "
import json
with open('$RESULTS_FILE') as f:
    data = json.load(f)
for ds in ['sst2', 'agnews', 'trec']:
    if ds in data['results']:
        r = data['results'][ds]
        print(f'{ds.upper()}:')
        print(f'  Bridge:        {r[\"bridge\"][\"accuracy\"]:.1f}%')
        print(f'  Prompt-Tuning: {r[\"prompt_tuning\"][\"accuracy\"]:.1f}%')
        print(f'  Llama 0-shot:  {r[\"llama_zeroshot\"][\"accuracy\"]:.1f}%')
        if 'text_relay' in r and r['text_relay'].get('accuracy'):
            print(f'  Text-Relay:    {r[\"text_relay\"][\"accuracy\"]:.1f}%')
        print()
"
    fi

} 2>&1 | tee "$LOG_FILE"

echo "Done! Full log saved to: $LOG_FILE"
