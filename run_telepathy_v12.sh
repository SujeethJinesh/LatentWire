#!/usr/bin/env bash
# run_telepathy_v12.sh
# Phase 12: Diffusion Bridge with DiT + Rectified Flow
#
# THE FIX: Stop predicting (regression) → start generating (diffusion).
#
# Why regression fails (The Blur Problem):
# - Given input, many valid Mistral vectors could decode it
# - Regression outputs AVERAGE of all valid outputs → lies OFF manifold
# - Off-manifold vectors decode as garbage
#
# Why diffusion works:
# - Learns to move TOWARD the data manifold from any starting point
# - Output guaranteed to lie ON the Mistral embedding manifold
# - Sharp, valid vectors instead of blurry averages

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

# Architecture
SOURCE_LAYER="${SOURCE_LAYER:-16}"
SOFT_TOKENS="${SOFT_TOKENS:-128}"
DEPTH="${DEPTH:-6}"
HEADS="${HEADS:-8}"

# Training
STEPS="${STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-3e-4}"

# Diffusion
DIFFUSION_STEPS="${DIFFUSION_STEPS:-10}"

# Output
RUN_ID="telepathy_v12_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="runs/${RUN_ID}"
LOG_FILE="${OUTPUT_DIR}/train.log"

mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Banner
# =============================================================================
echo "=========================================================================="
echo " Latent Telepathy Phase 12: Diffusion Bridge (DiT + Rectified Flow)"
echo "=========================================================================="
echo " Run ID:             $RUN_ID"
echo " Output Dir:         $OUTPUT_DIR"
echo " GPUs:               $NPROC"
echo ""
echo " THE FIX: Stop predicting → start generating"
echo ""
echo " Why regression fails (Blur Problem):"
echo "   - Outputs average of valid vectors → lies OFF the manifold"
echo "   - Off-manifold vectors decode as garbage"
echo ""
echo " Why diffusion works:"
echo "   - Learns to move TOWARD the data manifold"
echo "   - Output is ON the manifold = valid Mistral embedding"
echo ""
echo " Architecture: DiT with ${DEPTH} layers, ${HEADS} heads"
echo " Training: Rectified Flow (predict velocity v = target - noise)"
echo " Sampling: Euler integration with ${DIFFUSION_STEPS} steps"
echo "=========================================================================="
echo ""

# Save configuration
cat > "${OUTPUT_DIR}/config.json" << EOF
{
    "run_id": "$RUN_ID",
    "phase": 12,
    "key_fix": "Diffusion Bridge: Generate vectors ON the manifold instead of regressing averages",
    "source_model": "$SOURCE_MODEL",
    "target_model": "$TARGET_MODEL",
    "source_layer": $SOURCE_LAYER,
    "soft_tokens": $SOFT_TOKENS,
    "depth": $DEPTH,
    "heads": $HEADS,
    "steps": $STEPS,
    "batch_size": $BATCH_SIZE,
    "lr": "$LR",
    "diffusion_steps": $DIFFUSION_STEPS,
    "num_gpus": $NPROC
}
EOF

# =============================================================================
# Phase 1: DDP Training with Rectified Flow
# =============================================================================
echo "[Phase 1/2] Training Diffusion Bridge on $NPROC GPU(s)..."
echo "  Log file: $LOG_FILE"
echo ""

RANDOM_PORT=$((29500 + RANDOM % 1000))

{
    torchrun \
        --standalone \
        --nproc_per_node="$NPROC" \
        --master_port "$RANDOM_PORT" \
        telepathy/train_telepathy_v12.py \
        --source_model "$SOURCE_MODEL" \
        --target_model "$TARGET_MODEL" \
        --source_layer "$SOURCE_LAYER" \
        --soft_tokens "$SOFT_TOKENS" \
        --depth "$DEPTH" \
        --heads "$HEADS" \
        --steps "$STEPS" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --bf16 \
        --save_every 500 \
        --save_path "${OUTPUT_DIR}/bridge_v12_final.pt"
} 2>&1 | tee "$LOG_FILE"

# =============================================================================
# Phase 2: Evaluation
# =============================================================================
CHECKPOINT="${OUTPUT_DIR}/bridge_v12_final.pt"

if [[ -f "$CHECKPOINT" ]]; then
    echo ""
    echo "[Phase 2/2] Evaluating Diffusion Bridge..."

    EVAL_LOG="${OUTPUT_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"

    {
        python telepathy/eval_telepathy_v12.py \
            --checkpoint "$CHECKPOINT" \
            --source_layer "$SOURCE_LAYER" \
            --soft_tokens "$SOFT_TOKENS" \
            --depth "$DEPTH" \
            --heads "$HEADS" \
            --num_samples 20 \
            --max_new_tokens 200 \
            --diffusion_steps "$DIFFUSION_STEPS" \
            --output_dir "$OUTPUT_DIR"
    } 2>&1 | tee "$EVAL_LOG"

    echo ""
    echo "=========================================================================="
    echo " Phase 12 Complete!"
    echo "=========================================================================="
    echo " Results: ${OUTPUT_DIR}/eval_v12_results.json"
    echo " Log:     $EVAL_LOG"
    echo ""
    echo " SUCCESS CRITERIA:"
    echo "   - Outputs should VARY per input (not all identical)"
    echo "   - Outputs should contain question entities"
    echo "   - Entity transfer rate > 30% = Working!"
    echo "   - Output diversity > 50% = Diffusion is generating, not memorizing"
    echo ""
    echo " If still failing:"
    echo "   - Increase training steps (10000+)"
    echo "   - Increase diffusion steps (20+)"
    echo "   - Try deeper DiT (8 layers)"
    echo "=========================================================================="
else
    echo ""
    echo "WARNING: Checkpoint not found, skipping evaluation"
    echo "Expected: $CHECKPOINT"
fi
