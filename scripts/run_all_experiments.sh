#!/usr/bin/env bash
#
# Master Script - Run All Experiments
#
# Runs complete experiment suite in order:
#   1. Text baseline (upper bound)
#   2. Token budget baseline (fair comparison)
#   3. PCA baseline (linear compression)
#   4. LatentWire training (learned compression)
#   5. LatentWire evaluation
#   6. Comprehensive analysis comparing all results
#
# Usage: PYTHONPATH=. bash scripts/run_all_experiments.sh
#
# Estimated time: ~4-6 hours on 4x H100
#

set -e

echo "========================================================================"
echo "MASTER EXPERIMENT SUITE - Compressed Interlingua"
echo "========================================================================"
echo ""
echo "This runs the complete experimental pipeline:"
echo "  1. Text Baseline (upper bound)"
echo "  2. Token Budget Baseline (fair comparison)"
echo "  3. PCA Baseline (linear compression)"
echo "  4. LatentWire Training (learned compression)"
echo "  5. LatentWire Evaluation"
echo "  6. Comprehensive Analysis"
echo ""
echo "Estimated time: 4-6 hours on 4x H100"
echo "========================================================================"
echo ""

# Configuration
LLAMA_ID="${LLAMA_ID:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
QWEN_ID="${QWEN_ID:-Qwen/Qwen2.5-7B-Instruct}"
DATASET="${DATASET:-squad}"
EVAL_SAMPLES="${EVAL_SAMPLES:-10000}"     # Full validation set for baselines
TRAIN_SAMPLES="${TRAIN_SAMPLES:-87599}"   # Full training set for LatentWire
TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-32}"            # Training batch size
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}" # Eval batch size (256=safe ~50-60GB, 384=aggressive ~65-70GB, 512+=OOM risk)
LATENT_LEN="${LATENT_LEN:-32}"
D_Z="${D_Z:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-12}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-runs/full_suite}"

echo "Configuration:"
echo "  LLAMA_ID: $LLAMA_ID"
echo "  QWEN_ID: $QWEN_ID"
echo "  DATASET: $DATASET"
echo "  EVAL_SAMPLES: $EVAL_SAMPLES"
echo "  TRAIN_SAMPLES: $TRAIN_SAMPLES"
echo "  TRAIN_EPOCHS: $TRAIN_EPOCHS"
echo "  BATCH_SIZE (training): $BATCH_SIZE"
echo "  EVAL_BATCH_SIZE (baselines): $EVAL_BATCH_SIZE"
echo "  LATENT_LEN (M): $LATENT_LEN"
echo "  D_Z: $D_Z"
echo "  BASE_OUTPUT_DIR: $BASE_OUTPUT_DIR"
echo ""

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory and master log
mkdir -p "$BASE_OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG="$BASE_OUTPUT_DIR/master_${TIMESTAMP}.log"

echo "Master log: $MASTER_LOG"
echo ""
echo "========================================================================"
echo ""

# Track timing
SUITE_START=$(date +%s)

{
echo "Starting experiment suite at $(date)"
echo ""

# =============================================================================
# PHASE 1: TEXT BASELINE (Upper Bound)
# =============================================================================
echo "========================================================================"
echo "PHASE 1/5: TEXT BASELINE - Upper Bound"
echo "========================================================================"
echo ""
echo "Evaluating both LLMs with full text prompts..."
echo "This establishes the best possible performance."
echo ""

PHASE_START=$(date +%s)

# Llama text baseline (QWEN SKIPPED FOR SPEED)
echo "[1a] Running Llama text baseline (batch_size=$EVAL_BATCH_SIZE)..."
python scripts/baselines/evaluate_text_baseline.py \
    --model_id "$LLAMA_ID" \
    --dataset "$DATASET" \
    --samples "$EVAL_SAMPLES" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --save_dir "$BASE_OUTPUT_DIR/baselines/text_llama"

echo ""
echo "[1b] Skipping Qwen text baseline for speed..."

PHASE_END=$(date +%s)
PHASE_TIME=$((PHASE_END - PHASE_START))
echo ""
echo "✓ Phase 1 complete in ${PHASE_TIME}s ($((PHASE_TIME / 60))m)"
echo ""

# =============================================================================
# PHASE 2: TOKEN BUDGET BASELINE (Fair Comparison)
# =============================================================================
echo "========================================================================"
echo "PHASE 2/5: TOKEN BUDGET BASELINE - Fair Comparison"
echo "========================================================================"
echo ""
echo "Evaluating with text truncated to M=$LATENT_LEN tokens..."
echo "This is the critical fairness baseline."
echo ""

PHASE_START=$(date +%s)

# Llama token budget (QWEN SKIPPED FOR SPEED)
echo "[2a] Running Llama token budget (M=$LATENT_LEN, batch_size=$EVAL_BATCH_SIZE)..."
python scripts/baselines/evaluate_token_budget.py \
    --model_id "$LLAMA_ID" \
    --dataset "$DATASET" \
    --samples "$EVAL_SAMPLES" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --token_budget "$LATENT_LEN" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --save_dir "$BASE_OUTPUT_DIR/baselines/token_budget_llama_m${LATENT_LEN}"

echo ""
echo "[2b] Skipping Qwen token budget for speed..."

PHASE_END=$(date +%s)
PHASE_TIME=$((PHASE_END - PHASE_START))
echo ""
echo "✓ Phase 2 complete in ${PHASE_TIME}s ($((PHASE_TIME / 60))m)"
echo ""

# =============================================================================
# PHASE 3: PCA BASELINE (Linear Compression)
# =============================================================================
echo "========================================================================"
echo "PHASE 3/5: PCA BASELINE - Linear Compression"
echo "========================================================================"
echo ""
echo "Testing if PCA (linear projection) is sufficient..."
echo ""

PHASE_START=$(date +%s)

echo "[3] Running PCA baseline (M=$LATENT_LEN, samples=$EVAL_SAMPLES)..."
python scripts/baselines/pca_baseline.py \
    --llama_id "$LLAMA_ID" \
    --dataset "$DATASET" \
    --samples "$EVAL_SAMPLES" \
    --latent_len "$LATENT_LEN" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --save_dir "$BASE_OUTPUT_DIR/baselines/pca_m${LATENT_LEN}"

PHASE_END=$(date +%s)
PHASE_TIME=$((PHASE_END - PHASE_START))
echo ""
echo "✓ Phase 3 complete in ${PHASE_TIME}s ($((PHASE_TIME / 60))m)"
echo ""

# =============================================================================
# PHASE 4: LATENTWIRE TRAINING (Learned Compression)
# =============================================================================
echo "========================================================================"
echo "PHASE 4/5: LATENTWIRE TRAINING - Learned Compression"
echo "========================================================================"
echo ""
echo "Training encoder + adapters for compressed interlingua..."
echo "  Samples: $TRAIN_SAMPLES"
echo "  Epochs: $TRAIN_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Latent: M=$LATENT_LEN, d_z=$D_Z"
echo ""

PHASE_START=$(date +%s)

python latentwire/train.py \
    --llama_id "$LLAMA_ID" \
    --qwen_id "$QWEN_ID" \
    --samples "$TRAIN_SAMPLES" \
    --epochs "$TRAIN_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --latent_len "$LATENT_LEN" \
    --d_z "$D_Z" \
    --encoder_type byte \
    --dataset "$DATASET" \
    --sequential_models \
    --lr 1e-4 \
    --warm_anchor_text "Answer: " \
    --first_token_ce_weight 0.5 \
    --K 4 \
    --kd_first_k_weight 0.1 \
    --save_dir "$BASE_OUTPUT_DIR/latentwire" \
    --save_every 1000

PHASE_END=$(date +%s)
PHASE_TIME=$((PHASE_END - PHASE_START))
echo ""
echo "✓ Phase 4 complete in ${PHASE_TIME}s ($((PHASE_TIME / 60))m)"
echo ""

# =============================================================================
# PHASE 5: LATENTWIRE EVALUATION
# =============================================================================
echo "========================================================================"
echo "PHASE 5/5: LATENTWIRE EVALUATION"
echo "========================================================================"
echo ""
echo "Evaluating trained LatentWire system..."
echo ""

PHASE_START=$(date +%s)

# Find best checkpoint
BEST_CKPT="$BASE_OUTPUT_DIR/latentwire/checkpoint_best"
if [ ! -d "$BEST_CKPT" ]; then
    echo "WARNING: Best checkpoint not found, using latest"
    BEST_CKPT="$BASE_OUTPUT_DIR/latentwire"
fi

echo "Using checkpoint: $BEST_CKPT"
echo ""

# Evaluate on both models
echo "[5a] Evaluating LatentWire on Llama..."
python latentwire/eval.py \
    --ckpt "$BEST_CKPT" \
    --llama_id "$LLAMA_ID" \
    --dataset "$DATASET" \
    --samples "$EVAL_SAMPLES" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --sequential_eval \
    --fresh_eval \
    --calibration embed_rms \
    --latent_anchor_mode text \
    --latent_anchor_text "Answer: " \
    --append_bos_after_prefix yes \
    --out_dir "$BASE_OUTPUT_DIR/latentwire/eval_llama" \
    --models llama

echo ""
echo "[5b] Skipping Qwen evaluation for speed..."

PHASE_END=$(date +%s)
PHASE_TIME=$((PHASE_END - PHASE_START))
echo ""
echo "✓ Phase 5 complete in ${PHASE_TIME}s ($((PHASE_TIME / 60))m)"
echo ""

# =============================================================================
# COMPREHENSIVE ANALYSIS
# =============================================================================
echo "========================================================================"
echo "COMPREHENSIVE ANALYSIS"
echo "========================================================================"
echo ""
echo "Analyzing all results..."
echo ""

python scripts/analyze_all_results.py --results_dir "$BASE_OUTPUT_DIR"

SUITE_END=$(date +%s)
SUITE_TIME=$((SUITE_END - SUITE_START))

echo ""
echo "========================================================================"
echo "EXPERIMENT SUITE COMPLETE!"
echo "========================================================================"
echo ""
echo "Total time: ${SUITE_TIME}s ($((SUITE_TIME / 60))m / $((SUITE_TIME / 3600))h)"
echo ""
echo "Results saved to: $BASE_OUTPUT_DIR"
echo "  - baselines/text_llama/"
echo "  - baselines/text_qwen/"
echo "  - baselines/token_budget_llama_m${LATENT_LEN}/"
echo "  - baselines/token_budget_qwen_m${LATENT_LEN}/"
echo "  - baselines/pca_m${LATENT_LEN}/"
echo "  - latentwire/ (checkpoints)"
echo "  - latentwire/eval_llama/"
echo "  - latentwire/eval_qwen/"
echo "  - comparison_report.json"
echo "  - comparison_report.txt"
echo ""
echo "View report:"
echo "  cat $BASE_OUTPUT_DIR/comparison_report.txt"
echo ""

} 2>&1 | tee "$MASTER_LOG"

echo "Master log saved to: $MASTER_LOG"
echo ""
