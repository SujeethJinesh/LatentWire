#!/usr/bin/env bash
# run_sst2_signal_check.sh
# Phase 16: SST-2 Signal Check
#
# GOAL: Validate the bridge can transmit ANY semantic information
# before attempting complex tasks like GSM8K.
#
# Success Criteria:
#   - Accuracy > 50%: Bridge transmits info
#   - Accuracy > 70%: Bridge is working
#   - Accuracy > 85%: Bridge is excellent
#   - Accuracy ~ 50%: Bridge is broken

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

# Lighter architecture for simple task
SOFT_TOKENS="${SOFT_TOKENS:-32}"
DEPTH="${DEPTH:-2}"
HEADS="${HEADS:-8}"

# Training
STEPS="${STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LR="${LR:-2e-4}"
DIVERSITY_WEIGHT="${DIVERSITY_WEIGHT:-0.1}"  # Batch diversity loss weight

# Output
RUN_ID="sst2_signal_check_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="runs/${RUN_ID}"
LOG_FILE="${OUTPUT_DIR}/train.log"

mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Banner
# =============================================================================
echo "=========================================================================="
echo " Phase 16: SST-2 Signal Check (CONTINUOUS VERSION)"
echo "=========================================================================="
echo " Run ID:        $RUN_ID"
echo " Output Dir:    $OUTPUT_DIR"
echo " GPUs:          $NPROC"
echo ""
echo " GOAL: Validate bridge can transmit semantic information"
echo ""
echo " WHY SST-2 (not GSM8K):"
echo "   - 67,000 training examples (10x more data)"
echo "   - Binary classification (much simpler task)"
echo "   - 'Blurriness' acceptable (no exact numbers needed)"
echo ""
echo " WHY CONTINUOUS (not VQ):"
echo "   - VQ collapsed in 7 attempts (perplexity → 1)"
echo "   - Continuous soft tokens + batch diversity loss"
echo "   - RMS normalization prevents saturation"
echo ""
echo " SUCCESS CRITERIA:"
echo "   - Accuracy > 50%: Bridge transmits info"
echo "   - Accuracy > 70%: Bridge is working"
echo "   - Accuracy > 85%: Bridge is excellent"
echo "   - Accuracy ~ 50%: Bridge is broken"
echo ""
echo " Architecture:"
echo "   - Soft tokens: $SOFT_TOKENS"
echo "   - Mode: CONTINUOUS (no VQ)"
echo "   - Depth: $DEPTH"
echo "   - Diversity weight: $DIVERSITY_WEIGHT"
echo "=========================================================================="
echo ""

# Save configuration
cat > "${OUTPUT_DIR}/config.json" << EOF
{
    "run_id": "$RUN_ID",
    "phase": 16,
    "task": "SST-2 Signal Check (CONTINUOUS)",
    "source_model": "$SOURCE_MODEL",
    "target_model": "$TARGET_MODEL",
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
# Training
# =============================================================================
echo "[Phase 1/2] Training SST-2 Bridge..."
echo "  Log file: $LOG_FILE"
echo ""

RANDOM_PORT=$((29500 + RANDOM % 1000))

{
    torchrun \
        --standalone \
        --nproc_per_node="$NPROC" \
        --master_port "$RANDOM_PORT" \
        telepathy/train_telepathy_sst2.py \
        --source_model "$SOURCE_MODEL" \
        --target_model "$TARGET_MODEL" \
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
        --save_path "${OUTPUT_DIR}/bridge_sst2.pt"
} 2>&1 | tee "$LOG_FILE"

# =============================================================================
# Evaluation
# =============================================================================
if [[ -f "${OUTPUT_DIR}/bridge_sst2.pt" ]]; then
    CHECKPOINT="${OUTPUT_DIR}/bridge_sst2.pt"
    echo ""
    echo "[Phase 2/2] Evaluating SST-2 Bridge..."
    echo "  Checkpoint: $CHECKPOINT"

    EVAL_LOG="${OUTPUT_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"

    {
        python telepathy/eval_telepathy_sst2.py \
            --checkpoint "$CHECKPOINT" \
            --soft_tokens "$SOFT_TOKENS" \
            --depth "$DEPTH" \
            --heads "$HEADS" \
            --num_samples 872 \
            --output_dir "$OUTPUT_DIR" \
            --bf16
    } 2>&1 | tee "$EVAL_LOG"

    echo ""
    echo "=========================================================================="
    echo " Phase 16 SST-2 Signal Check Complete! (CONTINUOUS VERSION)"
    echo "=========================================================================="
    echo " Results: ${OUTPUT_DIR}/eval_sst2_results.json"
    echo " Train Log: $LOG_FILE"
    echo " Eval Log: $EVAL_LOG"
    echo ""
    echo " INTERPRETATION:"
    echo "   - If accuracy > 70%: Bridge works! Try harder tasks."
    echo "   - If accuracy ~ 50%: Bridge is broken. Check diversity loss."
    echo "=========================================================================="
else
    echo ""
    echo "WARNING: No checkpoint found, skipping evaluation"
    echo "Expected: ${OUTPUT_DIR}/bridge_sst2.pt"
fi
