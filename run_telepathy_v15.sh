#!/usr/bin/env bash
# run_telepathy_v15.sh
# Phase 15: FSQ-Telepathy (Finite Scalar Quantization)
#
# WHY FSQ INSTEAD OF VQ:
# - VQ collapsed to 1 code (perplexity=1) despite:
#   * Cosine similarity, entropy bonus, LayerNorm removal
# - FSQ has NO codebook = NO collapse possible
#
# FSQ ARCHITECTURE:
# - 8 dimensions × 8 levels each = 16,777,216 effective codes
# - Project: 4096 -> 8 -> quantize -> 8 -> 4096
# - Pure LM loss training (no auxiliary loss)

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
HEADS="${HEADS:-8}"

# Training
STEPS="${STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-2}"          # Reduced for OOM - 2 models per GPU
GRAD_ACCUM="${GRAD_ACCUM:-4}"          # Effective batch = 2 * 4 = 8
LR="${LR:-2e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"

# Output
RUN_ID="telepathy_v15_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="runs/${RUN_ID}"
LOG_FILE="${OUTPUT_DIR}/train.log"

mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Banner
# =============================================================================
echo "=========================================================================="
echo " Latent Telepathy Phase 15: FSQ-Telepathy (Finite Scalar Quantization)"
echo "=========================================================================="
echo " Run ID:             $RUN_ID"
echo " Output Dir:         $OUTPUT_DIR"
echo " GPUs:               $NPROC"
echo ""
echo " WHY FSQ INSTEAD OF VQ:"
echo "   - VQ collapsed to 1 code (perplexity=1) despite:"
echo "     * Cosine similarity, entropy bonus, LayerNorm removal"
echo "   - FSQ has NO codebook = NO collapse possible"
echo ""
echo " FSQ ARCHITECTURE:"
echo "   - 8 dimensions × 8 levels = 16,777,216 effective codes"
echo "   - Project: 4096 -> 8 -> quantize -> 8 -> 4096"
echo "   - Pure LM loss (no auxiliary loss needed)"
echo "   - Soft tokens: $SOFT_TOKENS"
echo "   - Depth: $DEPTH, Heads: $HEADS"
echo "=========================================================================="
echo ""

# Save configuration
cat > "${OUTPUT_DIR}/config.json" << EOF
{
    "run_id": "$RUN_ID",
    "phase": 15,
    "approach": "FSQ-Telepathy (Finite Scalar Quantization)",
    "source_model": "$SOURCE_MODEL",
    "target_model": "$TARGET_MODEL",
    "source_layer": $SOURCE_LAYER,
    "soft_tokens": $SOFT_TOKENS,
    "depth": $DEPTH,
    "heads": $HEADS,
    "fsq_levels": [8, 8, 8, 8, 8, 8, 8, 8],
    "effective_codebook": 16777216,
    "steps": $STEPS,
    "batch_size": $BATCH_SIZE,
    "lr": "$LR",
    "warmup_steps": $WARMUP_STEPS,
    "num_gpus": $NPROC,
    "key_changes": [
        "FSQ replaces VQ (no codebook collapse)",
        "8^8 = 16M effective codes",
        "Pure LM loss training",
        "1-step inference (no diffusion)"
    ]
}
EOF

# =============================================================================
# Phase 1: Calibration (Optional - compute Llama/Mistral stats)
# =============================================================================
STATS_FILE="${OUTPUT_DIR}/stats.pt"

if [[ -f "telepathy/phase1_calibration.py" ]]; then
    echo "[Phase 1/3] Computing calibration statistics..."
    python telepathy/phase1_calibration.py \
        --source_model "$SOURCE_MODEL" \
        --target_model "$TARGET_MODEL" \
        --source_layer "$SOURCE_LAYER" \
        --num_samples 200 \
        --batch_size 16 \
        --output_file "$STATS_FILE" \
        2>&1 | tee "${OUTPUT_DIR}/calibration.log"
    STATS_ARG="--stats_path $STATS_FILE"
else
    echo "[Phase 1/3] Skipping calibration (no phase1_calibration.py)"
    STATS_ARG=""
fi

# =============================================================================
# Phase 2: DDP Training with FSQ
# =============================================================================
echo ""
echo "[Phase 2/3] Training FSQ-Telepathy Bridge..."
echo "  Log file: $LOG_FILE"
echo ""

RANDOM_PORT=$((29500 + RANDOM % 1000))

{
    torchrun \
        --standalone \
        --nproc_per_node="$NPROC" \
        --master_port "$RANDOM_PORT" \
        telepathy/train_telepathy_v15.py \
        --source_model "$SOURCE_MODEL" \
        --target_model "$TARGET_MODEL" \
        $STATS_ARG \
        --source_layer "$SOURCE_LAYER" \
        --soft_tokens "$SOFT_TOKENS" \
        --depth "$DEPTH" \
        --heads "$HEADS" \
        --steps "$STEPS" \
        --batch_size "$BATCH_SIZE" \
        --grad_accum "$GRAD_ACCUM" \
        --lr "$LR" \
        --warmup_steps "$WARMUP_STEPS" \
        --bf16 \
        --save_every 500 \
        --save_path "${OUTPUT_DIR}/bridge_v15.pt"
} 2>&1 | tee "$LOG_FILE"

# =============================================================================
# Phase 3: Evaluation
# =============================================================================
if [[ -f "${OUTPUT_DIR}/bridge_v15.pt" ]]; then
    CHECKPOINT="${OUTPUT_DIR}/bridge_v15.pt"
    echo ""
    echo "[Phase 3/3] Evaluating FSQ-Telepathy Bridge..."
    echo "  Checkpoint: $CHECKPOINT"

    EVAL_LOG="${OUTPUT_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"

    {
        python telepathy/eval_telepathy_v15.py \
            --checkpoint "$CHECKPOINT" \
            $STATS_ARG \
            --source_layer "$SOURCE_LAYER" \
            --soft_tokens "$SOFT_TOKENS" \
            --depth "$DEPTH" \
            --heads "$HEADS" \
            --num_samples 20 \
            --max_new_tokens 128 \
            --output_dir "$OUTPUT_DIR" \
            --bf16
    } 2>&1 | tee "$EVAL_LOG"

    echo ""
    echo "=========================================================================="
    echo " Phase 15 FSQ Complete!"
    echo "=========================================================================="
    echo " Results: ${OUTPUT_DIR}/eval_v15_results.json"
    echo " Log:     $EVAL_LOG"
    echo ""
    echo " SUCCESS CRITERIA:"
    echo "   1. LM Loss should decrease during training"
    echo "   2. Diversity should stay high (0.5-1.0) - NO COLLAPSE"
    echo "   3. Outputs should be coherent (not repetitive garbage)"
    echo "   4. Entity transfer rate > 30%"
    echo ""
    echo " KEY DIFFERENCE FROM VQ:"
    echo "   - VQ collapsed to perplexity=1 (1 code used)"
    echo "   - FSQ cannot collapse (no learned codebook)"
    echo "=========================================================================="
else
    echo ""
    echo "WARNING: No checkpoint found, skipping evaluation"
    echo "Expected: ${OUTPUT_DIR}/bridge_v15.pt"
fi
