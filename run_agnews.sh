#!/usr/bin/env bash
# run_agnews.sh
# Phase 17: AG News 4-Class Classification
#
# GOAL: Test bridge on harder multi-class task after SST-2 success.
# Uses optimal config from comprehensive ablation study.
#
# Success Criteria:
#   - Random baseline: 25%
#   - Accuracy > 50%: Bridge works for multi-class
#   - Accuracy > 70%: Bridge is excellent
#   - Accuracy matches Mistral text: Perfect transfer

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
# Configuration (Optimal from SST-2 ablation)
# =============================================================================
SOURCE_MODEL="${SOURCE_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
TARGET_MODEL="${TARGET_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"

# OPTIMAL CONFIG from SST-2 comprehensive ablation:
# - Layer 31: 94.5% (final layer has task-specific info)
# - 8 tokens: 96.5% (information bottleneck principle)
SOURCE_LAYER="${SOURCE_LAYER:-31}"
SOFT_TOKENS="${SOFT_TOKENS:-8}"
DEPTH="${DEPTH:-2}"
HEADS="${HEADS:-8}"

# Training (more steps for 4-class)
STEPS="${STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LR="${LR:-2e-4}"
DIVERSITY_WEIGHT="${DIVERSITY_WEIGHT:-0.1}"

# Output
RUN_ID="agnews_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="runs/${RUN_ID}"
LOG_FILE="${OUTPUT_DIR}/train.log"

mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Banner
# =============================================================================
echo "=========================================================================="
echo " Phase 17: AG News 4-Class Classification"
echo "=========================================================================="
echo " Run ID:        $RUN_ID"
echo " Output Dir:    $OUTPUT_DIR"
echo " GPUs:          $NPROC"
echo ""
echo " GOAL: Test bridge on harder multi-class task"
echo ""
echo " AG NEWS CLASSES:"
echo "   - World (international news)"
echo "   - Sports (athletics, games)"
echo "   - Business (finance, economy)"
echo "   - Sci/Tech (science, technology)"
echo ""
echo " SUCCESS CRITERIA:"
echo "   - Random baseline: 25%"
echo "   - Accuracy > 50%: Bridge works for multi-class"
echo "   - Accuracy > 70%: Bridge is excellent"
echo ""
echo " OPTIMAL CONFIG (from SST-2 ablation):"
echo "   - Source layer: $SOURCE_LAYER (final layer)"
echo "   - Soft tokens: $SOFT_TOKENS (information bottleneck)"
echo "   - Diversity weight: $DIVERSITY_WEIGHT"
echo "=========================================================================="
echo ""

# Save configuration
cat > "${OUTPUT_DIR}/config.json" << EOF
{
    "run_id": "$RUN_ID",
    "phase": "17-agnews",
    "task": "AG News 4-Class Classification",
    "source_model": "$SOURCE_MODEL",
    "target_model": "$TARGET_MODEL",
    "source_layer": $SOURCE_LAYER,
    "soft_tokens": $SOFT_TOKENS,
    "depth": $DEPTH,
    "heads": $HEADS,
    "mode": "continuous",
    "steps": $STEPS,
    "batch_size": $BATCH_SIZE,
    "lr": "$LR",
    "diversity_weight": $DIVERSITY_WEIGHT,
    "num_gpus": $NPROC
}
EOF

# =============================================================================
# Phase 1: Baselines (establish upper/lower bounds)
# =============================================================================
echo "[Phase 1/3] Running AG News Baselines..."
BASELINE_LOG="${OUTPUT_DIR}/baselines_$(date +%Y%m%d_%H%M%S).log"

{
    python telepathy/eval_agnews_baselines.py \
        --num_samples 200 \
        --output_dir "$OUTPUT_DIR" \
        --bf16
} 2>&1 | tee "$BASELINE_LOG"

echo ""
echo "Baselines complete. See: ${OUTPUT_DIR}/agnews_baselines.json"
echo ""

# =============================================================================
# Phase 2: Training
# =============================================================================
echo "[Phase 2/3] Training AG News Bridge..."
echo "  Log file: $LOG_FILE"
echo ""

RANDOM_PORT=$((29500 + RANDOM % 1000))

{
    torchrun \
        --standalone \
        --nproc_per_node="$NPROC" \
        --master_port "$RANDOM_PORT" \
        telepathy/train_telepathy_agnews.py \
        --source_model "$SOURCE_MODEL" \
        --target_model "$TARGET_MODEL" \
        --source_layer "$SOURCE_LAYER" \
        --soft_tokens "$SOFT_TOKENS" \
        --depth "$DEPTH" \
        --heads "$HEADS" \
        --steps "$STEPS" \
        --batch_size "$BATCH_SIZE" \
        --grad_accum "$GRAD_ACCUM" \
        --lr "$LR" \
        --diversity_weight "$DIVERSITY_WEIGHT" \
        --eval_every 200 \
        --save_every 500 \
        --bf16 \
        --save_path "${OUTPUT_DIR}/bridge_agnews.pt"
} 2>&1 | tee "$LOG_FILE"

# =============================================================================
# Phase 3: Evaluation
# =============================================================================
if [[ -f "${OUTPUT_DIR}/bridge_agnews.pt" ]]; then
    CHECKPOINT="${OUTPUT_DIR}/bridge_agnews.pt"
    echo ""
    echo "[Phase 3/3] Evaluating AG News Bridge..."
    echo "  Checkpoint: $CHECKPOINT"

    EVAL_LOG="${OUTPUT_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"

    {
        python telepathy/eval_telepathy_agnews.py \
            --checkpoint "$CHECKPOINT" \
            --source_layer "$SOURCE_LAYER" \
            --soft_tokens "$SOFT_TOKENS" \
            --depth "$DEPTH" \
            --heads "$HEADS" \
            --num_samples 1000 \
            --output_dir "$OUTPUT_DIR" \
            --bf16
    } 2>&1 | tee "$EVAL_LOG"

    echo ""
    echo "=========================================================================="
    echo " Phase 17 AG News Complete!"
    echo "=========================================================================="
    echo " Results:"
    echo "   - Baselines: ${OUTPUT_DIR}/agnews_baselines.json"
    echo "   - Bridge: ${OUTPUT_DIR}/eval_agnews_results.json"
    echo " Logs:"
    echo "   - Baselines: $BASELINE_LOG"
    echo "   - Training: $LOG_FILE"
    echo "   - Evaluation: $EVAL_LOG"
    echo ""
    echo " INTERPRETATION:"
    echo "   - Compare bridge accuracy to Mistral text baseline"
    echo "   - If bridge > 50%: Bridge works for multi-class!"
    echo "   - If bridge ~ 25%: Bridge failed on harder task."
    echo "=========================================================================="
else
    echo ""
    echo "WARNING: No checkpoint found, skipping evaluation"
    echo "Expected: ${OUTPUT_DIR}/bridge_agnews.pt"
fi
