#!/usr/bin/env bash
# run_telepathy_v6.sh
# Phase 6: The Translator Pivot
#
# KEY INSIGHT: V5 failed because we anchored to ANSWER embeddings,
# forcing the small bridge to solve math (Q->A). It learned generic
# "math vectors" and discarded entities.
#
# FIX: Anchor to QUESTION embeddings. Bridge only translates Q->Q,
# letting Mistral's 7B parameters do the reasoning.
#
# Expected Result: Entities preserved (ducks stay ducks, not apples)

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# =============================================================================
# GPU Detection
# =============================================================================
detect_nproc() {
    if [[ -n "${NUM_GPUS:-}" ]]; then
        echo "$NUM_GPUS"
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        local count
        count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
        if [[ "$count" -gt 0 ]]; then
            echo "$count"
            return
        fi
    fi
    echo 1
}

NPROC=$(detect_nproc)

# =============================================================================
# Configuration
# =============================================================================
SOURCE_MODEL="${SOURCE_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
TARGET_MODEL="${TARGET_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"

# Same architecture as V5
SOURCE_LAYER="${SOURCE_LAYER:-16}"
SOFT_TOKENS="${SOFT_TOKENS:-256}"
DEPTH="${DEPTH:-4}"
HEADS="${HEADS:-8}"

# Loss weights
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-2.0}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.1}"
CONTRASTIVE_TEMP="${CONTRASTIVE_TEMP:-0.07}"

# Training
STEPS="${STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-4}"  # Reduced from 8 due to extra memory for question embeddings
LR="${LR:-1e-4}"

# Output
RUN_ID="telepathy_v6_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="runs/${RUN_ID}"
STATS_FILE="${OUTPUT_DIR}/stats.pt"
LOG_FILE="${OUTPUT_DIR}/train.log"

mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Banner
# =============================================================================
echo "=========================================================================="
echo " Latent Telepathy Phase 6: The Translator Pivot"
echo "=========================================================================="
echo " Run ID:             $RUN_ID"
echo " Output Dir:         $OUTPUT_DIR"
echo " GPUs:               $NPROC"
echo ""
echo " KEY CHANGE FROM V5:"
echo "   Anchor Target:    QUESTION embeddings (was: ANSWER embeddings)"
echo "   Bridge Role:      Translator (Q->Q), not Solver (Q->A)"
echo ""
echo " Architecture (unchanged):"
echo "   Source Layer:     $SOURCE_LAYER"
echo "   Soft Tokens:      $SOFT_TOKENS"
echo "   Anchor Weight:    $ANCHOR_WEIGHT"
echo "   Steps:            $STEPS"
echo ""
echo " Expected Result: Entities preserved (ducks->ducks, not ducks->apples)"
echo "=========================================================================="
echo ""

# Save configuration
cat > "${OUTPUT_DIR}/config.json" << EOF
{
    "run_id": "$RUN_ID",
    "phase": 6,
    "key_change": "Anchor to QUESTION embeddings instead of ANSWER embeddings",
    "source_model": "$SOURCE_MODEL",
    "target_model": "$TARGET_MODEL",
    "source_layer": $SOURCE_LAYER,
    "soft_tokens": $SOFT_TOKENS,
    "depth": $DEPTH,
    "heads": $HEADS,
    "steps": $STEPS,
    "batch_size": $BATCH_SIZE,
    "lr": $LR,
    "anchor_weight": $ANCHOR_WEIGHT,
    "contrastive_weight": $CONTRASTIVE_WEIGHT,
    "contrastive_temp": $CONTRASTIVE_TEMP,
    "num_gpus": $NPROC
}
EOF

# =============================================================================
# Phase 1: Calibration
# =============================================================================
echo "[Phase 1/3] Calibrating for Layer $SOURCE_LAYER..."
echo ""

{
    python telepathy/phase1_calibration.py \
        --source_model "$SOURCE_MODEL" \
        --target_model "$TARGET_MODEL" \
        --source_layer "$SOURCE_LAYER" \
        --num_samples 500 \
        --batch_size 4 \
        --output_file "$STATS_FILE"
} 2>&1 | tee "${OUTPUT_DIR}/calibration.log"

if [ ! -f "$STATS_FILE" ]; then
    echo "CRITICAL ERROR: stats.pt not found after calibration"
    exit 1
fi

echo ""

# =============================================================================
# Phase 2: DDP Training with V6 (Question Anchor)
# =============================================================================
echo "[Phase 2/3] Training V6 Bridge (Question Anchor) on $NPROC GPU(s)..."
echo "  Log file: $LOG_FILE"
echo ""

RANDOM_PORT=$((29500 + RANDOM % 1000))

{
    torchrun \
        --standalone \
        --nproc_per_node="$NPROC" \
        --master_port "$RANDOM_PORT" \
        telepathy/train_telepathy_v6.py \
        --source_model "$SOURCE_MODEL" \
        --target_model "$TARGET_MODEL" \
        --stats_path "$STATS_FILE" \
        --source_layer "$SOURCE_LAYER" \
        --soft_tokens "$SOFT_TOKENS" \
        --depth "$DEPTH" \
        --heads "$HEADS" \
        --steps "$STEPS" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --anchor_weight "$ANCHOR_WEIGHT" \
        --contrastive_weight "$CONTRASTIVE_WEIGHT" \
        --contrastive_temp "$CONTRASTIVE_TEMP" \
        --bf16 \
        --save_every 500 \
        --save_path "${OUTPUT_DIR}/bridge_v6_final.pt"
} 2>&1 | tee "$LOG_FILE"

# =============================================================================
# Phase 3: Evaluation
# =============================================================================
CHECKPOINT="${OUTPUT_DIR}/bridge_v6_final.pt"

if [[ -f "$CHECKPOINT" ]]; then
    echo ""
    echo "[Phase 3/3] Evaluating V6 Bridge..."

    EVAL_LOG="${OUTPUT_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"

    {
        python telepathy/eval_telepathy.py \
            --checkpoint "$CHECKPOINT" \
            --stats_path "$STATS_FILE" \
            --source_layer "$SOURCE_LAYER" \
            --soft_tokens "$SOFT_TOKENS" \
            --depth "$DEPTH" \
            --heads "$HEADS" \
            --num_samples 20 \
            --max_new_tokens 150 \
            --output_dir "$OUTPUT_DIR" \
            --bridge_version 1
    } 2>&1 | tee "$EVAL_LOG"

    echo ""
    echo "=========================================================================="
    echo " Phase 6 Complete!"
    echo "=========================================================================="
    echo " Results: ${OUTPUT_DIR}/eval_results.json"
    echo " Log:     $EVAL_LOG"
    echo ""
    echo " CHECK: Do outputs mention 'ducks' when question has 'ducks'?"
    echo "        If yes -> Entity preservation working!"
    echo "        If still scrambled -> Need different approach"
    echo "=========================================================================="
else
    echo ""
    echo "WARNING: Checkpoint not found, skipping evaluation"
    echo "Expected: $CHECKPOINT"
fi
