#!/usr/bin/env bash
# run_telepathy_v2.sh
# Latent Telepathy Phase 2: Contrastive Learning Fix
#
# Phase 1 achieved 0% accuracy due to Posterior Collapse.
# Phase 2 adds InfoNCE contrastive loss to force unique latent representations.
#
# Key changes from Phase 1:
#   - Soft tokens: 64 → 128 (more capacity)
#   - Batch size: 4 → 8 (more negatives for contrastive)
#   - Contrastive weight: 0.5 (prevents collapse)
#
# Usage:
#   bash run_telepathy_v2.sh

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
# Configuration (Phase 2 defaults)
# =============================================================================
SOURCE_MODEL="${SOURCE_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
TARGET_MODEL="${TARGET_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
SOURCE_LAYER="${SOURCE_LAYER:-20}"

# Increased capacity for Phase 2
SOFT_TOKENS="${SOFT_TOKENS:-128}"
DEPTH="${DEPTH:-4}"
HEADS="${HEADS:-8}"

# Training hyperparameters
STEPS="${STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-2e-4}"

# Contrastive learning
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.5}"
CONTRASTIVE_TEMP="${CONTRASTIVE_TEMP:-0.07}"

# Output
RUN_ID="telepathy_v2_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="runs/${RUN_ID}"
STATS_FILE="${OUTPUT_DIR}/stats.pt"
LOG_FILE="${OUTPUT_DIR}/train.log"

mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Banner
# =============================================================================
echo "=========================================================================="
echo " Latent Telepathy Phase 2: Contrastive Learning"
echo "=========================================================================="
echo " Run ID:             $RUN_ID"
echo " Output Dir:         $OUTPUT_DIR"
echo " GPUs:               $NPROC"
echo ""
echo " Source Model:       $SOURCE_MODEL"
echo " Target Model:       $TARGET_MODEL"
echo " Source Layer:       $SOURCE_LAYER"
echo ""
echo " Phase 2 Changes:"
echo "   Soft Tokens:      $SOFT_TOKENS (was 64)"
echo "   Batch Size:       $BATCH_SIZE (was 4)"
echo "   Contrastive Wt:   $CONTRASTIVE_WEIGHT"
echo "   Contrastive Temp: $CONTRASTIVE_TEMP"
echo ""
echo " Steps:              $STEPS"
echo " Learning Rate:      $LR"
echo "=========================================================================="
echo ""

# Save configuration
cat > "${OUTPUT_DIR}/config.json" << EOF
{
    "run_id": "$RUN_ID",
    "phase": 2,
    "source_model": "$SOURCE_MODEL",
    "target_model": "$TARGET_MODEL",
    "source_layer": $SOURCE_LAYER,
    "soft_tokens": $SOFT_TOKENS,
    "depth": $DEPTH,
    "heads": $HEADS,
    "steps": $STEPS,
    "batch_size": $BATCH_SIZE,
    "lr": $LR,
    "contrastive_weight": $CONTRASTIVE_WEIGHT,
    "contrastive_temp": $CONTRASTIVE_TEMP,
    "num_gpus": $NPROC
}
EOF

# =============================================================================
# Phase 1: Calibration (Reuse if possible, else run)
# =============================================================================
# Check if we have a recent calibration file we can reuse
EXISTING_STATS=$(ls -t runs/telepathy_*/stats.pt 2>/dev/null | head -1 || true)

if [[ -n "$EXISTING_STATS" ]] && [[ -f "$EXISTING_STATS" ]]; then
    echo "[Phase 1/2] Reusing existing calibration: $EXISTING_STATS"
    cp "$EXISTING_STATS" "$STATS_FILE"
else
    echo "[Phase 1/2] Running Calibration..."
    {
        python telepathy/phase1_calibration.py \
            --source_model "$SOURCE_MODEL" \
            --target_model "$TARGET_MODEL" \
            --source_layer "$SOURCE_LAYER" \
            --num_samples 500 \
            --batch_size 4 \
            --output_file "$STATS_FILE"
    } 2>&1 | tee "${OUTPUT_DIR}/calibration.log"
fi

if [ ! -f "$STATS_FILE" ]; then
    echo "CRITICAL ERROR: stats.pt not found"
    exit 1
fi

echo ""

# =============================================================================
# Phase 2: DDP Training with Contrastive Loss
# =============================================================================
echo "[Phase 2/2] Launching Phase 2 Training on $NPROC GPU(s)..."
echo "  Log file: $LOG_FILE"
echo ""

RANDOM_PORT=$((29500 + RANDOM % 1000))

{
    torchrun \
        --standalone \
        --nproc_per_node="$NPROC" \
        --master_port "$RANDOM_PORT" \
        telepathy/train_telepathy_v2.py \
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
        --contrastive_weight "$CONTRASTIVE_WEIGHT" \
        --contrastive_temp "$CONTRASTIVE_TEMP" \
        --bf16 \
        --save_every 500 \
        --save_path "${OUTPUT_DIR}/bridge_v2_final.pt"
} 2>&1 | tee "$LOG_FILE"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=========================================================================="
echo " Phase 2 Training Complete!"
echo "=========================================================================="
echo " Artifacts saved to: $OUTPUT_DIR"
echo ""
echo " Files:"
echo "   - config.json:         Run configuration"
echo "   - stats.pt:            Calibration statistics"
echo "   - train.log:           Training output"
echo "   - bridge_v2_final.pt:  Final checkpoint"
echo ""
echo " Next step: Evaluate with"
echo "   bash run_telepathy_eval.sh ${OUTPUT_DIR}/bridge_v2_final.pt"
echo "=========================================================================="
