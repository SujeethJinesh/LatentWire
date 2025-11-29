#!/usr/bin/env bash
# run_telepathy_eval.sh
# Evaluate trained Telepathy bridge on held-out test data
#
# Usage:
#   bash run_telepathy_eval.sh runs/telepathy_*/bridge_final.pt
#
# Or with defaults (finds most recent run):
#   bash run_telepathy_eval.sh

set -euo pipefail

# =============================================================================
# HPC Environment Setup
# =============================================================================
if command -v module >/dev/null 2>&1; then
    module purge 2>/dev/null || true
    module load gcc/13.1.0 2>/dev/null || true
    module load conda/24.3.0-0 2>/dev/null || true
    module load stockcuda/12.6.2 2>/dev/null || true
fi

export PYTHONPATH="${PYTHONPATH:-.}:."
export TOKENIZERS_PARALLELISM=false

# =============================================================================
# Find Checkpoint
# =============================================================================
if [[ $# -ge 1 ]]; then
    CHECKPOINT="$1"
else
    # Find most recent telepathy run
    LATEST_RUN=$(ls -td runs/telepathy_* 2>/dev/null | head -1)
    if [[ -z "$LATEST_RUN" ]]; then
        echo "ERROR: No telepathy runs found in runs/"
        echo "Usage: bash run_telepathy_eval.sh <checkpoint_path>"
        exit 1
    fi
    CHECKPOINT="${LATEST_RUN}/bridge_final.pt"
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

RUN_DIR=$(dirname "$CHECKPOINT")
STATS_PATH="${RUN_DIR}/stats.pt"
LOG_FILE="${RUN_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"

# =============================================================================
# Configuration
# =============================================================================
NUM_SAMPLES="${NUM_SAMPLES:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-150}"

echo "=========================================================================="
echo " Telepathy Evaluation"
echo "=========================================================================="
echo " Checkpoint:  $CHECKPOINT"
echo " Stats:       $STATS_PATH"
echo " Samples:     $NUM_SAMPLES"
echo " Log:         $LOG_FILE"
echo "=========================================================================="
echo ""

# =============================================================================
# Run Evaluation
# =============================================================================
{
    python telepathy/eval_telepathy.py \
        --checkpoint "$CHECKPOINT" \
        --stats_path "$STATS_PATH" \
        --num_samples "$NUM_SAMPLES" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --output_dir "$RUN_DIR"
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================================================="
echo " Evaluation Complete!"
echo "=========================================================================="
echo " Results: ${RUN_DIR}/eval_results.json"
echo " Log:     $LOG_FILE"
echo "=========================================================================="
