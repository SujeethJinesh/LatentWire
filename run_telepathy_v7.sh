#!/usr/bin/env bash
# run_telepathy_v7.sh
# Phase 7: Scale-Corrected Telepathy
#
# KEY FIXES:
# 1. Output scaling: Perceiver outputs ~1.0 std, Mistral expects ~0.03
#    The 33x overload saturated attention, causing degenerate loops
# 2. Reverted to Answer anchoring (V5 logic)
#    V6's Question anchor created conflicting objectives

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

# Same architecture as V5/V6
SOURCE_LAYER="${SOURCE_LAYER:-16}"
SOFT_TOKENS="${SOFT_TOKENS:-256}"
DEPTH="${DEPTH:-4}"
HEADS="${HEADS:-8}"

# Loss weights (V5 settings)
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-2.0}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.1}"
CONTRASTIVE_TEMP="${CONTRASTIVE_TEMP:-0.07}"

# Training
STEPS="${STEPS:-2500}"
BATCH_SIZE="${BATCH_SIZE:-4}"  # Reduced from 8 due to OOM on H100s
LR="${LR:-1e-4}"

# Output
RUN_ID="telepathy_v7_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="runs/${RUN_ID}"
STATS_FILE="${OUTPUT_DIR}/stats.pt"
LOG_FILE="${OUTPUT_DIR}/train.log"

mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Banner
# =============================================================================
echo "=========================================================================="
echo " Latent Telepathy Phase 7: Scale-Corrected"
echo "=========================================================================="
echo " Run ID:             $RUN_ID"
echo " Output Dir:         $OUTPUT_DIR"
echo " GPUs:               $NPROC"
echo ""
echo " KEY FIXES FROM V6:"
echo "   1. Output scaling:  tanh + learnable scale (~0.03)"
echo "   2. Answer anchor:   Reverted to V5 logic (aligned objectives)"
echo ""
echo " THE MAGNITUDE BUG:"
echo "   Perceiver outputs ~1.0 std, Mistral expects ~0.03"
echo "   33x overload saturated attention → degenerate loops"
echo ""
echo " Architecture:"
echo "   Source Layer:     $SOURCE_LAYER"
echo "   Soft Tokens:      $SOFT_TOKENS"
echo "   Anchor Weight:    $ANCHOR_WEIGHT"
echo "   Steps:            $STEPS"
echo "=========================================================================="
echo ""

# Save configuration
cat > "${OUTPUT_DIR}/config.json" << EOF
{
    "run_id": "$RUN_ID",
    "phase": 7,
    "key_fixes": [
        "Output scaling (tanh + learnable scale)",
        "Reverted to Answer anchoring (V5 logic)"
    ],
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
# Phase 2: DDP Training with V7 (Scale-Corrected)
# =============================================================================
echo "[Phase 2/3] Training V7 Bridge (Scale-Corrected) on $NPROC GPU(s)..."
echo "  Log file: $LOG_FILE"
echo ""

RANDOM_PORT=$((29500 + RANDOM % 1000))

{
    torchrun \
        --standalone \
        --nproc_per_node="$NPROC" \
        --master_port "$RANDOM_PORT" \
        telepathy/train_telepathy_v7.py \
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
        --save_path "${OUTPUT_DIR}/bridge_v7_final.pt"
} 2>&1 | tee "$LOG_FILE"

# =============================================================================
# Phase 3: Evaluation
# =============================================================================
CHECKPOINT="${OUTPUT_DIR}/bridge_v7_final.pt"

if [[ -f "$CHECKPOINT" ]]; then
    echo ""
    echo "[Phase 3/3] Evaluating V7 Bridge..."

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
            --bridge_version 7
    } 2>&1 | tee "$EVAL_LOG"

    echo ""
    echo "=========================================================================="
    echo " Phase 7 Complete!"
    echo "=========================================================================="
    echo " Results: ${OUTPUT_DIR}/eval_results.json"
    echo " Log:     $EVAL_LOG"
    echo ""
    echo " CHECK: Are outputs coherent numbers (not 10*10*10... or 1000000...)?"
    echo "        If yes → Scale fix working!"
    echo "        If still degenerate → Need further magnitude debugging"
    echo "=========================================================================="
else
    echo ""
    echo "WARNING: Checkpoint not found, skipping evaluation"
    echo "Expected: $CHECKPOINT"
fi
