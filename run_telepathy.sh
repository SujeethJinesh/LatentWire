#!/usr/bin/env bash
# run_telepathy.sh
# Latent Telepathy: Llama 3.1 8B -> Mistral 0.3 7B
#
# Phases:
#   1. Calibration: Collect distribution statistics from both models
#   2. Training: Train Perceiver Resampler with reconstruction loss
#
# Usage:
#   bash run_telepathy.sh
#
# Environment variables:
#   NUM_GPUS: Override GPU count detection
#   STEPS: Override training steps (default: 3000)
#   BATCH_SIZE: Override batch size (default: 4)

set -euo pipefail

# =============================================================================
# HPC Environment Setup (if available)
# =============================================================================
if command -v module >/dev/null 2>&1; then
    module purge 2>/dev/null || true
    module load gcc/13.1.0 2>/dev/null || true
    module load conda/24.3.0-0 2>/dev/null || true
    module load stockcuda/12.6.2 2>/dev/null || true
fi

# =============================================================================
# Environment Configuration
# =============================================================================
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
SOURCE_LAYER="${SOURCE_LAYER:-20}"
SOFT_TOKENS="${SOFT_TOKENS:-64}"
DEPTH="${DEPTH:-4}"
HEADS="${HEADS:-8}"
STEPS="${STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-4}"
RECON_WEIGHT="${RECON_WEIGHT:-1.0}"

# Output directory with timestamp
RUN_ID="telepathy_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="runs/${RUN_ID}"
STATS_FILE="${OUTPUT_DIR}/stats.pt"
LOG_FILE="${OUTPUT_DIR}/train.log"

mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Banner
# =============================================================================
echo "=========================================================================="
echo " Latent Telepathy: Cross-Model Latent Communication"
echo "=========================================================================="
echo " Run ID:        $RUN_ID"
echo " Output Dir:    $OUTPUT_DIR"
echo " GPUs:          $NPROC"
echo ""
echo " Source Model:  $SOURCE_MODEL"
echo " Target Model:  $TARGET_MODEL"
echo " Source Layer:  $SOURCE_LAYER"
echo ""
echo " Soft Tokens:   $SOFT_TOKENS"
echo " Depth:         $DEPTH"
echo " Heads:         $HEADS"
echo ""
echo " Steps:         $STEPS"
echo " Batch Size:    $BATCH_SIZE (per GPU)"
echo " Learning Rate: $LR"
echo " Recon Weight:  $RECON_WEIGHT"
echo "=========================================================================="
echo ""

# Save configuration
cat > "${OUTPUT_DIR}/config.json" << EOF
{
    "run_id": "$RUN_ID",
    "source_model": "$SOURCE_MODEL",
    "target_model": "$TARGET_MODEL",
    "source_layer": $SOURCE_LAYER,
    "soft_tokens": $SOFT_TOKENS,
    "depth": $DEPTH,
    "heads": $HEADS,
    "steps": $STEPS,
    "batch_size": $BATCH_SIZE,
    "lr": $LR,
    "recon_weight": $RECON_WEIGHT,
    "num_gpus": $NPROC
}
EOF

# =============================================================================
# Phase 1: Calibration
# =============================================================================
echo "[Phase 1/2] Running Calibration..."
echo "  Collecting distribution statistics from both models..."
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

# Verify calibration completed
if [ ! -f "$STATS_FILE" ]; then
    echo ""
    echo "CRITICAL ERROR: Calibration failed - stats.pt was not generated"
    echo "Check ${OUTPUT_DIR}/calibration.log for details"
    exit 1
fi

echo ""
echo "Calibration complete. Stats saved to: $STATS_FILE"
echo ""

# =============================================================================
# Phase 2: DDP Training
# =============================================================================
echo "[Phase 2/2] Launching DDP Training on $NPROC GPU(s)..."
echo "  Log file: $LOG_FILE"
echo ""

# Random port for distributed training
RANDOM_PORT=$((29500 + RANDOM % 1000))

{
    torchrun \
        --standalone \
        --nproc_per_node="$NPROC" \
        --master_port "$RANDOM_PORT" \
        telepathy/train_telepathy.py \
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
        --recon_weight "$RECON_WEIGHT" \
        --bf16 \
        --save_every 500 \
        --save_path "${OUTPUT_DIR}/bridge_final.pt"
} 2>&1 | tee "$LOG_FILE"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=========================================================================="
echo " Training Complete!"
echo "=========================================================================="
echo " Artifacts saved to: $OUTPUT_DIR"
echo ""
echo " Files:"
echo "   - config.json:        Run configuration"
echo "   - calibration.log:    Phase 1 output"
echo "   - stats.pt:           Calibration statistics"
echo "   - train.log:          Phase 2 output"
echo "   - bridge_final.pt:    Final model checkpoint"
echo "   - bridge_step*.pt:    Intermediate checkpoints"
echo "=========================================================================="
