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
    # Try different checkpoint names (v2 uses bridge_v2_final.pt)
    if [[ -f "${LATEST_RUN}/bridge_v2_final.pt" ]]; then
        CHECKPOINT="${LATEST_RUN}/bridge_v2_final.pt"
    elif [[ -f "${LATEST_RUN}/bridge_final.pt" ]]; then
        CHECKPOINT="${LATEST_RUN}/bridge_final.pt"
    else
        echo "ERROR: No checkpoint found in ${LATEST_RUN}"
        echo "Looking for: bridge_final.pt or bridge_v2_final.pt"
        exit 1
    fi
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

RUN_DIR=$(dirname "$CHECKPOINT")
STATS_PATH="${RUN_DIR}/stats.pt"
CONFIG_FILE="${RUN_DIR}/config.json"
LOG_FILE="${RUN_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"

# =============================================================================
# Configuration (auto-detect from config.json if available)
# =============================================================================
NUM_SAMPLES="${NUM_SAMPLES:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-150}"

# Read architecture params from config.json
if [[ -f "$CONFIG_FILE" ]]; then
    SOFT_TOKENS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('soft_tokens', 64))")
    DEPTH=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('depth', 4))")
    HEADS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('heads', 8))")
    echo "Auto-detected from config.json: soft_tokens=$SOFT_TOKENS, depth=$DEPTH, heads=$HEADS"
else
    SOFT_TOKENS="${SOFT_TOKENS:-64}"
    DEPTH="${DEPTH:-4}"
    HEADS="${HEADS:-8}"
    echo "Using defaults: soft_tokens=$SOFT_TOKENS, depth=$DEPTH, heads=$HEADS"
fi

echo "=========================================================================="
echo " Telepathy Evaluation"
echo "=========================================================================="
echo " Checkpoint:   $CHECKPOINT"
echo " Stats:        $STATS_PATH"
echo " Soft Tokens:  $SOFT_TOKENS"
echo " Samples:      $NUM_SAMPLES"
echo " Log:          $LOG_FILE"
echo "=========================================================================="
echo ""

# =============================================================================
# Run Evaluation
# =============================================================================
{
    python telepathy/eval_telepathy.py \
        --checkpoint "$CHECKPOINT" \
        --stats_path "$STATS_PATH" \
        --soft_tokens "$SOFT_TOKENS" \
        --depth "$DEPTH" \
        --heads "$HEADS" \
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
