#!/usr/bin/env bash
# run_telepathy_v14.sh
# Phase 14: Hybrid Conditioning Diffusion
#
# THE FIX FOR V13's CONDITIONING COLLAPSE:
# - Global Pooling (V12 style guide rail) + Cross-Attention (V13 style details)
# - Mirrors Stable Diffusion XL architecture
# - Increased capacity (1024 dim, 16 heads, 10k steps)

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

# Architecture - V14 increased capacity
SOURCE_LAYER="${SOURCE_LAYER:-16}"
SOFT_TOKENS="${SOFT_TOKENS:-128}"
DEPTH="${DEPTH:-6}"  # Increased from V13's 4

# Training - V14 longer training
STEPS="${STEPS:-10000}"  # Increased from 5000
BATCH_SIZE="${BATCH_SIZE:-4}"  # Reduced due to larger model
LR="${LR:-3e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"  # Increased warmup
EMA_DECAY="${EMA_DECAY:-0.999}"

# Diffusion inference
DIFFUSION_STEPS="${DIFFUSION_STEPS:-10}"

# Output
RUN_ID="telepathy_v14_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="runs/${RUN_ID}"
LOG_FILE="${OUTPUT_DIR}/train.log"

mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Banner
# =============================================================================
echo "=========================================================================="
echo " Latent Telepathy Phase 14: Hybrid Conditioning Diffusion"
echo "=========================================================================="
echo " Run ID:             $RUN_ID"
echo " Output Dir:         $OUTPUT_DIR"
echo " GPUs:               $NPROC"
echo ""
echo " THE FIX FOR V13's CONDITIONING COLLAPSE:"
echo ""
echo " V13 Problem:"
echo "   - Pure cross-attention was too weak"
echo "   - Model collapsed to repetitive outputs ('I I I I...')"
echo "   - Flow Loss plateaued at 1.58 (never converged)"
echo ""
echo " V14 Solution: HYBRID CONDITIONING"
echo "   1. Global Pooling (guide rail) - Attention-based pooling for 'gist'"
echo "   2. Cross-Attention (details) - Sequence-level entity retrieval"
echo ""
echo " Architecture Improvements:"
echo "   - Internal dim: 1024 (V13 was 512)"
echo "   - Heads: 16 (V13 was 8)"
echo "   - Depth: $DEPTH"
echo "   - Training steps: $STEPS (V13 was 5000)"
echo ""
echo " This mirrors Stable Diffusion XL:"
echo "   Pooled embeddings (global) + Sequence embeddings (local)"
echo "=========================================================================="
echo ""

# Save configuration
cat > "${OUTPUT_DIR}/config.json" << EOF
{
    "run_id": "$RUN_ID",
    "phase": 14,
    "key_fix": "Hybrid Conditioning (Global Pooling + Cross-Attention)",
    "source_model": "$SOURCE_MODEL",
    "target_model": "$TARGET_MODEL",
    "source_layer": $SOURCE_LAYER,
    "soft_tokens": $SOFT_TOKENS,
    "depth": $DEPTH,
    "internal_dim": 1024,
    "heads": 16,
    "steps": $STEPS,
    "batch_size": $BATCH_SIZE,
    "lr": "$LR",
    "warmup_steps": $WARMUP_STEPS,
    "ema_decay": $EMA_DECAY,
    "diffusion_steps": $DIFFUSION_STEPS,
    "num_gpus": $NPROC,
    "v13_problem": "Pure cross-attention too weak -> conditioning collapse",
    "v14_solution": [
        "Attention-based global pooling (guide rail)",
        "Cross-attention (entity details)",
        "Larger dim (1024 vs 512)",
        "More heads (16 vs 8)",
        "Longer training (10k steps)"
    ]
}
EOF

# =============================================================================
# Phase 1: DDP Training with Hybrid Conditioning
# =============================================================================
echo "[Phase 1/2] Training Hybrid Conditioning Diffusion Bridge..."
echo "  Log file: $LOG_FILE"
echo ""

RANDOM_PORT=$((29500 + RANDOM % 1000))

{
    torchrun \
        --standalone \
        --nproc_per_node="$NPROC" \
        --master_port "$RANDOM_PORT" \
        telepathy/train_telepathy_v14.py \
        --source_model "$SOURCE_MODEL" \
        --target_model "$TARGET_MODEL" \
        --source_layer "$SOURCE_LAYER" \
        --soft_tokens "$SOFT_TOKENS" \
        --depth "$DEPTH" \
        --steps "$STEPS" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --warmup_steps "$WARMUP_STEPS" \
        --ema_decay "$EMA_DECAY" \
        --bf16 \
        --save_every 1000 \
        --save_path "${OUTPUT_DIR}/bridge_v14_final.pt"
} 2>&1 | tee "$LOG_FILE"

# =============================================================================
# Phase 2: Evaluation
# =============================================================================
if [[ -f "${OUTPUT_DIR}/bridge_v14_final_ema.pt" ]]; then
    CHECKPOINT="${OUTPUT_DIR}/bridge_v14_final_ema.pt"
    echo "Using EMA checkpoint (recommended)"
elif [[ -f "${OUTPUT_DIR}/bridge_v14_final.pt" ]]; then
    CHECKPOINT="${OUTPUT_DIR}/bridge_v14_final.pt"
    echo "Using standard checkpoint"
else
    CHECKPOINT=""
fi

if [[ -n "$CHECKPOINT" ]]; then
    echo ""
    echo "[Phase 2/2] Evaluating Hybrid Conditioning Bridge..."
    echo "  Checkpoint: $CHECKPOINT"
    echo "  Testing: Does hybrid conditioning fix collapse + preserve entities?"

    EVAL_LOG="${OUTPUT_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"

    {
        python telepathy/eval_telepathy_v14.py \
            --checkpoint "$CHECKPOINT" \
            --source_layer "$SOURCE_LAYER" \
            --soft_tokens "$SOFT_TOKENS" \
            --depth "$DEPTH" \
            --num_samples 20 \
            --max_new_tokens 128 \
            --diffusion_steps "$DIFFUSION_STEPS" \
            --output_dir "$OUTPUT_DIR" \
            --bf16
    } 2>&1 | tee "$EVAL_LOG"

    echo ""
    echo "=========================================================================="
    echo " Phase 14 Complete!"
    echo "=========================================================================="
    echo " Results: ${OUTPUT_DIR}/eval_v14_results.json"
    echo " Log:     $EVAL_LOG"
    echo ""
    echo " SUCCESS CRITERIA:"
    echo "   1. Flow Loss should CONVERGE (V13 plateaued at 1.58)"
    echo "   2. Outputs should NOT be repetitive"
    echo "   3. Entity transfer rate > 30%"
    echo ""
    echo " If all criteria met:"
    echo "   => Hybrid conditioning fixed the collapse!"
    echo "   => Next: Add LM loss for accuracy (Phase 15)"
    echo ""
    echo " If training converges but entity transfer < 30%:"
    echo "   => Hybrid helps but need more capacity or different target"
    echo ""
    echo " If training still fails to converge:"
    echo "   => Fundamental architecture issue, try different approach"
    echo "=========================================================================="
else
    echo ""
    echo "WARNING: No checkpoint found, skipping evaluation"
    echo "Expected: ${OUTPUT_DIR}/bridge_v14_final.pt or bridge_v14_final_ema.pt"
fi
