#!/usr/bin/env bash
# run_telepathy_v10_eval_only.sh
# Re-run evaluation on existing V10 checkpoint with the fixed eval script
# (Added attention_mask and BOS token to fix empty outputs)

set -euo pipefail

# HPC Environment Setup
if command -v module >/dev/null 2>&1; then
    module purge 2>/dev/null || true
    module load gcc/13.1.0 2>/dev/null || true
    module load conda/24.3.0-0 2>/dev/null || true
    module load stockcuda/12.6.2 2>/dev/null || true
fi

export PYTHONPATH="${PYTHONPATH:-.}:."
export TOKENIZERS_PARALLELISM=false

# Find most recent V10 run
LATEST_RUN=$(ls -td runs/telepathy_v10_* 2>/dev/null | head -1 || echo "")

if [[ -z "$LATEST_RUN" ]]; then
    echo "ERROR: No V10 run found in runs/"
    exit 1
fi

echo "=========================================================================="
echo " Re-evaluating V10 with fixed eval script"
echo "=========================================================================="
echo " Run dir: $LATEST_RUN"
echo ""
echo " FIX: Added attention_mask and BOS token to generation"
echo "=========================================================================="

CHECKPOINT="${LATEST_RUN}/bridge_v10_final.pt"
STATS_FILE="${LATEST_RUN}/stats.pt"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

if [[ ! -f "$STATS_FILE" ]]; then
    echo "ERROR: Stats file not found: $STATS_FILE"
    exit 1
fi

EVAL_LOG="${LATEST_RUN}/eval_fixed_$(date +%Y%m%d_%H%M%S).log"

{
    python telepathy/eval_telepathy_v10.py \
        --checkpoint "$CHECKPOINT" \
        --stats_path "$STATS_FILE" \
        --source_layer 16 \
        --soft_tokens 128 \
        --depth 4 \
        --heads 8 \
        --num_samples 20 \
        --max_new_tokens 200 \
        --output_dir "$LATEST_RUN"
} 2>&1 | tee "$EVAL_LOG"

echo ""
echo "=========================================================================="
echo " Evaluation complete!"
echo "=========================================================================="
echo " Results: ${LATEST_RUN}/eval_v10_results.json"
echo " Log:     $EVAL_LOG"
echo "=========================================================================="
