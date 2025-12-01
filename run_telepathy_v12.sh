#!/usr/bin/env bash
# run_telepathy_v12.sh
# Phase 12: Diffusion Bridge with DiT + Rectified Flow
#
# THE FIX: Stop predicting (regression) → start generating (diffusion).
#
# Optimizations for stable DiT training:
# 1. Cosine LR Scheduler with Warmup (500 steps)
# 2. EMA (Exponential Moving Average, decay=0.999)
# 3. Gradient Clipping (max_norm=1.0)
#
# Flow Loss will NOT drop to zero - this is normal.
# Look for stable loss around 0.3-1.0 (not spikes/NaN).

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
DEPTH="${DEPTH:-4}"  # Reduced from 6 - OOM with 6 layers
HEADS="${HEADS:-8}"

# Training (optimized for DiT)
STEPS="${STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-4}"  # Reduced from 16 - OOM with both LLMs loaded
LR="${LR:-3e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
EMA_DECAY="${EMA_DECAY:-0.999}"

# Diffusion inference
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
echo " Optimizations for stable DiT training:"
echo "   - Cosine LR Scheduler (warmup=${WARMUP_STEPS})"
echo "   - EMA (decay=${EMA_DECAY})"
echo "   - Batch size: ${BATCH_SIZE}"
echo ""
echo " Architecture: DiT with ${DEPTH} layers, ${HEADS} heads"
echo " Training: Rectified Flow (predict velocity v = target - noise)"
echo " Sampling: Euler integration with ${DIFFUSION_STEPS} steps"
echo ""
echo " NOTE: Flow Loss will NOT drop to zero - this is normal!"
echo " Look for stable loss around 0.3-1.0"
echo "=========================================================================="
echo ""

# Save configuration
cat > "${OUTPUT_DIR}/config.json" << EOF
{
    "run_id": "$RUN_ID",
    "phase": 12,
    "key_fix": "Diffusion Bridge with EMA + Cosine Annealing",
    "source_model": "$SOURCE_MODEL",
    "target_model": "$TARGET_MODEL",
    "source_layer": $SOURCE_LAYER,
    "soft_tokens": $SOFT_TOKENS,
    "depth": $DEPTH,
    "heads": $HEADS,
    "steps": $STEPS,
    "batch_size": $BATCH_SIZE,
    "lr": "$LR",
    "warmup_steps": $WARMUP_STEPS,
    "ema_decay": $EMA_DECAY,
    "diffusion_steps": $DIFFUSION_STEPS,
    "num_gpus": $NPROC
}
EOF

# =============================================================================
# Phase 1: DDP Training with Rectified Flow + EMA + Cosine Annealing
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
        --warmup_steps "$WARMUP_STEPS" \
        --ema_decay "$EMA_DECAY" \
        --bf16 \
        --save_every 500 \
        --save_path "${OUTPUT_DIR}/bridge_v12_final.pt"
} 2>&1 | tee "$LOG_FILE"

# =============================================================================
# Phase 2: Evaluation (Using EMA checkpoint + ODE Solver)
# =============================================================================
# Use EMA checkpoint (higher quality) if available, else use standard
if [[ -f "${OUTPUT_DIR}/bridge_v12_final_ema.pt" ]]; then
    CHECKPOINT="${OUTPUT_DIR}/bridge_v12_final_ema.pt"
    echo "Using EMA checkpoint (recommended)"
elif [[ -f "${OUTPUT_DIR}/bridge_v12_final.pt" ]]; then
    CHECKPOINT="${OUTPUT_DIR}/bridge_v12_final.pt"
    echo "Using standard checkpoint"
else
    CHECKPOINT=""
fi

if [[ -n "$CHECKPOINT" ]]; then
    echo ""
    echo "[Phase 2/2] Evaluating Diffusion Bridge..."
    echo "  Checkpoint: $CHECKPOINT"
    echo "  CRITICAL: Using bridge.generate() (ODE solver)"

    EVAL_LOG="${OUTPUT_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"

    {
        python telepathy/eval_telepathy_v12.py \
            --checkpoint "$CHECKPOINT" \
            --source_layer "$SOURCE_LAYER" \
            --soft_tokens "$SOFT_TOKENS" \
            --depth "$DEPTH" \
            --heads "$HEADS" \
            --num_samples 20 \
            --max_new_tokens 128 \
            --diffusion_steps "$DIFFUSION_STEPS" \
            --output_dir "$OUTPUT_DIR" \
            --bf16
    } 2>&1 | tee "$EVAL_LOG"

    echo ""
    echo "=========================================================================="
    echo " Phase 12 Complete!"
    echo "=========================================================================="
    echo " Results: ${OUTPUT_DIR}/eval_v12_results.json"
    echo " Log:     $EVAL_LOG"
    echo ""
    echo " SUCCESS CRITERIA:"
    echo "   - Outputs should VARY per input (diversity > 50%)"
    echo "   - Outputs should contain question entities"
    echo "   - Entity transfer rate > 30% = Working!"
    echo ""
    echo " If still failing:"
    echo "   - Increase training steps (10000+)"
    echo "   - Increase diffusion steps (20+)"
    echo "   - Try deeper DiT (8 layers)"
    echo "   - Check if loss was stable during training"
    echo "=========================================================================="
else
    echo ""
    echo "WARNING: No checkpoint found, skipping evaluation"
    echo "Expected: ${OUTPUT_DIR}/bridge_v12_final.pt or bridge_v12_final_ema.pt"
fi
