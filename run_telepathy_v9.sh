#!/usr/bin/env bash
# run_telepathy_v9.sh
# Phase 9: Bag-of-Words Supervision
#
# KEY FIX: V8's mean-pooling reconstruction was too easy.
# "Ducks" and "chickens" have similar average vectors.
#
# Solution: Force the bridge to predict WHICH SPECIFIC TOKENS were in the input.
# Multi-hot BCE loss: If Llama reads "Janet" and "ducks", activate those classifiers.

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

# Architecture (same as V7/V8)
SOURCE_LAYER="${SOURCE_LAYER:-16}"
SOFT_TOKENS="${SOFT_TOKENS:-256}"
DEPTH="${DEPTH:-4}"
HEADS="${HEADS:-8}"

# Loss weights
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-2.0}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.1}"
CONTRASTIVE_TEMP="${CONTRASTIVE_TEMP:-0.07}"
BOW_WEIGHT="${BOW_WEIGHT:-5.0}"  # NEW: Strong signal for entity forcing

# Training
STEPS="${STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-4}"  # Keep at 4 to avoid OOM (BoW head adds memory)
LR="${LR:-1e-4}"

# Output
RUN_ID="telepathy_v9_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="runs/${RUN_ID}"
STATS_FILE="${OUTPUT_DIR}/stats.pt"
LOG_FILE="${OUTPUT_DIR}/train.log"

mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Banner
# =============================================================================
echo "=========================================================================="
echo " Latent Telepathy Phase 9: Bag-of-Words Supervision"
echo "=========================================================================="
echo " Run ID:             $RUN_ID"
echo " Output Dir:         $OUTPUT_DIR"
echo " GPUs:               $NPROC"
echo ""
echo " KEY FIX FROM V8:"
echo "   V8's mean-pooling reconstruction was too easy"
echo "   'ducks' and 'chickens' have similar average vectors"
echo ""
echo " BAG-OF-WORDS SUPERVISION:"
echo "   Force bridge to predict which specific tokens were in input"
echo "   Multi-hot BCE: If Llama reads 'ducks', activate 'ducks' classifier"
echo ""
echo " Architecture:"
echo "   Source Layer:     $SOURCE_LAYER"
echo "   Soft Tokens:      $SOFT_TOKENS"
echo "   Anchor Weight:    $ANCHOR_WEIGHT"
echo "   BoW Weight:       $BOW_WEIGHT"
echo "   Steps:            $STEPS"
echo "=========================================================================="
echo ""

# Save configuration
cat > "${OUTPUT_DIR}/config.json" << EOF
{
    "run_id": "$RUN_ID",
    "phase": 9,
    "key_fix": "Bag-of-Words supervision for specific entity transfer",
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
    "bow_weight": $BOW_WEIGHT,
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
# Phase 2: DDP Training with V9 (Bag-of-Words)
# =============================================================================
echo "[Phase 2/3] Training V9 Bridge (BoW Supervision) on $NPROC GPU(s)..."
echo "  Log file: $LOG_FILE"
echo ""

RANDOM_PORT=$((29500 + RANDOM % 1000))

{
    torchrun \
        --standalone \
        --nproc_per_node="$NPROC" \
        --master_port "$RANDOM_PORT" \
        telepathy/train_telepathy_v9.py \
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
        --bow_weight "$BOW_WEIGHT" \
        --bf16 \
        --save_every 500 \
        --save_path "${OUTPUT_DIR}/bridge_v9_final.pt"
} 2>&1 | tee "$LOG_FILE"

# =============================================================================
# Phase 3: Evaluation
# =============================================================================
CHECKPOINT="${OUTPUT_DIR}/bridge_v9_final.pt"

if [[ -f "$CHECKPOINT" ]]; then
    echo ""
    echo "[Phase 3/3] Evaluating V9 Bridge..."

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
            --bridge_version 9
    } 2>&1 | tee "$EVAL_LOG"

    echo ""
    echo "=========================================================================="
    echo " Phase 9 Complete!"
    echo "=========================================================================="
    echo " Results: ${OUTPUT_DIR}/eval_results.json"
    echo " Log:     $EVAL_LOG"
    echo ""
    echo " CHECK: Do outputs mention 'ducks' when question has 'ducks'?"
    echo "        If yes -> BoW supervision working!"
    echo "        If still 'students' -> Need stronger BoW weight or more training"
    echo "=========================================================================="
else
    echo ""
    echo "WARNING: Checkpoint not found, skipping evaluation"
    echo "Expected: $CHECKPOINT"
fi
