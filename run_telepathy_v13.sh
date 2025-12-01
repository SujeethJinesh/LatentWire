#!/usr/bin/env bash
# run_telepathy_v13.sh
# Phase 13: High-Fidelity Cross-Attention Diffusion
#
# THE TWO FIXES:
# 1. Full cross-attention to Llama sequence (no pooling bottleneck)
# 2. Target = Question embeddings (not Answer)
#
# If bridge can reconstruct Q in Mistral's space, Mistral will solve it.

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
DEPTH="${DEPTH:-4}"

# Training
STEPS="${STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-3e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
EMA_DECAY="${EMA_DECAY:-0.999}"

# Diffusion inference
DIFFUSION_STEPS="${DIFFUSION_STEPS:-10}"

# Output
RUN_ID="telepathy_v13_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="runs/${RUN_ID}"
LOG_FILE="${OUTPUT_DIR}/train.log"

mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Banner
# =============================================================================
echo "=========================================================================="
echo " Latent Telepathy Phase 13: High-Fidelity Cross-Attention Diffusion"
echo "=========================================================================="
echo " Run ID:             $RUN_ID"
echo " Output Dir:         $OUTPUT_DIR"
echo " GPUs:               $NPROC"
echo ""
echo " THE TWO FIXES:"
echo "   1. Full cross-attention to Llama sequence (no pooling)"
echo "   2. Target = Question embeddings (not Answer)"
echo ""
echo " V12 Problem: Global pooling destroyed entity info"
echo " V13 Solution: Each DiT block cross-attends to full Llama sequence"
echo ""
echo " V12 Problem: Supervised on Answer embeddings (wrong target)"
echo " V13 Solution: Supervised on Question embeddings (translate Q->Q)"
echo ""
echo " If bridge can reconstruct Q in Mistral's space,"
echo " Mistral will solve it naturally (7B params do reasoning)."
echo "=========================================================================="
echo ""

# Save configuration
cat > "${OUTPUT_DIR}/config.json" << EOF
{
    "run_id": "$RUN_ID",
    "phase": 13,
    "key_fix": "Cross-attention + Question reconstruction",
    "source_model": "$SOURCE_MODEL",
    "target_model": "$TARGET_MODEL",
    "source_layer": $SOURCE_LAYER,
    "soft_tokens": $SOFT_TOKENS,
    "depth": $DEPTH,
    "internal_dim": 512,
    "steps": $STEPS,
    "batch_size": $BATCH_SIZE,
    "lr": "$LR",
    "warmup_steps": $WARMUP_STEPS,
    "ema_decay": $EMA_DECAY,
    "diffusion_steps": $DIFFUSION_STEPS,
    "num_gpus": $NPROC,
    "changes_from_v12": [
        "Full cross-attention to Llama sequence (no pooling)",
        "Target = Question embeddings (not Answer)",
        "Smaller internal dim (512) for memory"
    ]
}
EOF

# =============================================================================
# Phase 1: DDP Training with Cross-Attention + Question Reconstruction
# =============================================================================
echo "[Phase 1/2] Training High-Fidelity Cross-Attention Diffusion Bridge..."
echo "  Log file: $LOG_FILE"
echo ""

RANDOM_PORT=$((29500 + RANDOM % 1000))

{
    torchrun \
        --standalone \
        --nproc_per_node="$NPROC" \
        --master_port "$RANDOM_PORT" \
        telepathy/train_telepathy_v13.py \
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
        --save_every 500 \
        --save_path "${OUTPUT_DIR}/bridge_v13_final.pt"
} 2>&1 | tee "$LOG_FILE"

# =============================================================================
# Phase 2: Evaluation
# =============================================================================
if [[ -f "${OUTPUT_DIR}/bridge_v13_final_ema.pt" ]]; then
    CHECKPOINT="${OUTPUT_DIR}/bridge_v13_final_ema.pt"
    echo "Using EMA checkpoint (recommended)"
elif [[ -f "${OUTPUT_DIR}/bridge_v13_final.pt" ]]; then
    CHECKPOINT="${OUTPUT_DIR}/bridge_v13_final.pt"
    echo "Using standard checkpoint"
else
    CHECKPOINT=""
fi

if [[ -n "$CHECKPOINT" ]]; then
    echo ""
    echo "[Phase 2/2] Evaluating High-Fidelity Cross-Attention Bridge..."
    echo "  Checkpoint: $CHECKPOINT"
    echo "  Testing: Does cross-attention preserve entities?"

    EVAL_LOG="${OUTPUT_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"

    {
        python telepathy/eval_telepathy_v13.py \
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
    echo " Phase 13 Complete!"
    echo "=========================================================================="
    echo " Results: ${OUTPUT_DIR}/eval_v13_results.json"
    echo " Log:     $EVAL_LOG"
    echo ""
    echo " SUCCESS CRITERIA:"
    echo "   - Entity transfer rate > 30% = Cross-attention works!"
    echo "   - Outputs contain 'Janet', 'ducks', question numbers"
    echo "   - Diverse outputs (not mode collapsed)"
    echo ""
    echo " If entity transfer > 30%:"
    echo "   => The cross-attention fix worked!"
    echo "   => Increase training for better accuracy"
    echo ""
    echo " If entity transfer < 10%:"
    echo "   => May need different cross-attention design"
    echo "   => Or question embeddings may not be the right target"
    echo "=========================================================================="
else
    echo ""
    echo "WARNING: No checkpoint found, skipping evaluation"
    echo "Expected: ${OUTPUT_DIR}/bridge_v13_final.pt or bridge_v13_final_ema.pt"
fi
