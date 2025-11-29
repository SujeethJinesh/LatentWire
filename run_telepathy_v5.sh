#!/usr/bin/env bash
# run_telepathy_v5.sh
# Phase 5: High-Resolution Telepathy
#
# Changes from Phase 4:
#   1. Soft Tokens: 128 -> 256 (More bandwidth for entity details)
#   2. Source Layer: 20 -> 16 (More concrete features, less abstract)
#   3. Anchor Weight: 1.0 -> 2.0 (Force stronger semantic alignment)
#   4. Steps: 2000 -> 2500 (More training for larger capacity)
#
# Goal: Fix entity scrambling (ducks->chickens, cherries->apples)

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
# Configuration - PHASE 5 CHANGES
# =============================================================================
SOURCE_MODEL="${SOURCE_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
TARGET_MODEL="${TARGET_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"

# PHASE 5: Key changes for high-resolution telepathy
SOURCE_LAYER="${SOURCE_LAYER:-16}"           # Changed from 20 -> 16 (more concrete)
SOFT_TOKENS="${SOFT_TOKENS:-256}"            # Changed from 128 -> 256 (more bandwidth)
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-2.0}"        # Changed from 1.0 -> 2.0 (stronger alignment)
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.1}"  # Keep low to prevent drift

# Architecture (unchanged)
DEPTH="${DEPTH:-4}"
HEADS="${HEADS:-8}"

# Training
STEPS="${STEPS:-2500}"                       # Increased from 2000 for larger capacity
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-4}"
CONTRASTIVE_TEMP="${CONTRASTIVE_TEMP:-0.07}"

# Output
RUN_ID="telepathy_v5_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="runs/${RUN_ID}"
STATS_FILE="${OUTPUT_DIR}/stats.pt"
LOG_FILE="${OUTPUT_DIR}/train.log"

mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Banner
# =============================================================================
echo "=========================================================================="
echo " Latent Telepathy Phase 5: High-Resolution Telepathy"
echo "=========================================================================="
echo " Run ID:             $RUN_ID"
echo " Output Dir:         $OUTPUT_DIR"
echo " GPUs:               $NPROC"
echo ""
echo " PHASE 5 CHANGES (vs Phase 4):"
echo "   Source Layer:     $SOURCE_LAYER (was 20 - more concrete features)"
echo "   Soft Tokens:      $SOFT_TOKENS (was 128 - doubled bandwidth)"
echo "   Anchor Weight:    $ANCHOR_WEIGHT (was 1.0 - stronger alignment)"
echo "   Steps:            $STEPS (was 2000)"
echo ""
echo " Goal: Fix entity scrambling (ducks->ducks, not ducks->chickens)"
echo "=========================================================================="
echo ""

# Save configuration
cat > "${OUTPUT_DIR}/config.json" << EOF
{
    "run_id": "$RUN_ID",
    "phase": 5,
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
    "num_gpus": $NPROC,
    "changes": "Layer 20->16, Tokens 128->256, Anchor 1.0->2.0"
}
EOF

# =============================================================================
# Phase 1: Calibration (MUST re-run because layer changed!)
# =============================================================================
echo "[Phase 1/3] Re-calibrating for Layer $SOURCE_LAYER (was 20)..."
echo "  This is required because hidden state statistics differ per layer."
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
# Phase 2: DDP Training
# =============================================================================
echo "[Phase 2/3] Training High-Res Bridge on $NPROC GPU(s)..."
echo "  Log file: $LOG_FILE"
echo ""

RANDOM_PORT=$((29500 + RANDOM % 1000))

{
    torchrun \
        --standalone \
        --nproc_per_node="$NPROC" \
        --master_port "$RANDOM_PORT" \
        telepathy/train_telepathy_v3.py \
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
        --save_path "${OUTPUT_DIR}/bridge_v5_final.pt"
} 2>&1 | tee "$LOG_FILE"

# =============================================================================
# Phase 3: Evaluation
# =============================================================================
CHECKPOINT="${OUTPUT_DIR}/bridge_v5_final.pt"

if [[ -f "$CHECKPOINT" ]]; then
    echo ""
    echo "[Phase 3/3] Evaluating High-Res Bridge..."

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
            --bridge_version 3
    } 2>&1 | tee "$EVAL_LOG"

    echo ""
    echo "=========================================================================="
    echo " Phase 5 Complete!"
    echo "=========================================================================="
    echo " Results: ${OUTPUT_DIR}/eval_results.json"
    echo " Log:     $EVAL_LOG"
    echo ""
    echo " If you see 'Janet' and 'ducks' in output (not 'chickens'), we succeeded!"
    echo "=========================================================================="
else
    echo ""
    echo "WARNING: Checkpoint not found, skipping evaluation"
    echo "Expected: $CHECKPOINT"
fi
