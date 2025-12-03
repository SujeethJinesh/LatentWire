#!/usr/bin/env bash
# run_telepathy_v15.sh
# Phase 15: VQ-Telepathy (Discrete Bottleneck)
#
# THE FIX FOR MANIFOLD MISMATCH:
# - Regression (V7): Blurry averages
# - Diffusion Global (V12): Lost details
# - Diffusion Cross-Attn (V13-14): Failed to converge
#
# VQ SOLUTION:
# - Discrete bottleneck prevents blur/drift
# - 4096 codebook entries for rich concepts
# - 1-step inference (no diffusion iteration)

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
VQ_WEIGHT="${VQ_WEIGHT:-1.0}"

# Output
RUN_ID="telepathy_v15_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="runs/${RUN_ID}"
LOG_FILE="${OUTPUT_DIR}/train.log"

mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Banner
# =============================================================================
echo "=========================================================================="
echo " Latent Telepathy Phase 15: VQ-Telepathy (Discrete Bottleneck)"
echo "=========================================================================="
echo " Run ID:             $RUN_ID"
echo " Output Dir:         $OUTPUT_DIR"
echo " GPUs:               $NPROC"
echo ""
echo " THE FIX FOR MANIFOLD MISMATCH:"
echo ""
echo " Previous Failures:"
echo "   - Regression (V7): Blurry averages (semantic drift)"
echo "   - Diffusion Global (V12): Converged but lost details"
echo "   - Diffusion Cross-Attn (V13-14): Failed to converge"
echo ""
echo " VQ SOLUTION:"
echo "   - Discrete bottleneck prevents blur/drift"
echo "   - 4096 codebook entries for rich concepts"
echo "   - 1-step inference (no diffusion iteration)"
echo "   - LM Loss ensures functional correctness"
echo ""
echo " Architecture:"
echo "   - Perceiver Resampler -> VQ Bottleneck -> Output Scale"
echo "   - Soft tokens: $SOFT_TOKENS"
echo "   - Depth: $DEPTH, Heads: $HEADS"
echo "   - VQ weight: $VQ_WEIGHT"
echo "=========================================================================="
echo ""

# Save configuration
cat > "${OUTPUT_DIR}/config.json" << EOF
{
    "run_id": "$RUN_ID",
    "phase": 15,
    "approach": "VQ-Telepathy (Discrete Bottleneck)",
    "source_model": "$SOURCE_MODEL",
    "target_model": "$TARGET_MODEL",
    "source_layer": $SOURCE_LAYER,
    "soft_tokens": $SOFT_TOKENS,
    "depth": $DEPTH,
    "heads": $HEADS,
    "codebook_size": 4096,
    "steps": $STEPS,
    "batch_size": $BATCH_SIZE,
    "lr": "$LR",
    "warmup_steps": $WARMUP_STEPS,
    "vq_weight": $VQ_WEIGHT,
    "num_gpus": $NPROC,
    "key_changes": [
        "Vector Quantization bottleneck",
        "4096 discrete codes",
        "LM loss on answer generation",
        "1-step inference (no diffusion)"
    ]
}
EOF

# =============================================================================
# Phase 1: Calibration (Optional - compute Llama/Mistral stats)
# =============================================================================
STATS_FILE="${OUTPUT_DIR}/stats.pt"

# Skip calibration - bridge uses identity normalization fallback
# Calibration was hanging on model load; not critical for VQ approach
echo "[Phase 1/3] Skipping calibration (using identity normalization)"
STATS_ARG=""

# =============================================================================
# Phase 2: DDP Training with VQ
# =============================================================================
echo ""
echo "[Phase 2/3] Training VQ-Telepathy Bridge..."
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
        --vq_weight "$VQ_WEIGHT" \
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
    echo "[Phase 3/3] Evaluating VQ-Telepathy Bridge..."
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
    echo " Phase 15 Complete!"
    echo "=========================================================================="
    echo " Results: ${OUTPUT_DIR}/eval_v15_results.json"
    echo " Log:     $EVAL_LOG"
    echo ""
    echo " SUCCESS CRITERIA:"
    echo "   1. LM Loss should decrease during training"
    echo "   2. Perplexity should be high (many codes used)"
    echo "   3. Outputs should be coherent (not repetitive garbage)"
    echo "   4. Entity transfer rate > 30%"
    echo ""
    echo " If successful:"
    echo "   => VQ-Telepathy fixed the manifold mismatch!"
    echo "   => Continue tuning for better accuracy"
    echo ""
    echo " If failed:"
    echo "   => Task may be fundamentally too hard"
    echo "   => Consider: larger codebook, more training, different arch"
    echo "=========================================================================="
else
    echo ""
    echo "WARNING: No checkpoint found, skipping evaluation"
    echo "Expected: ${OUTPUT_DIR}/bridge_v15.pt"
fi
