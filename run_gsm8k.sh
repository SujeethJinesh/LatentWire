#!/usr/bin/env bash
# run_gsm8k.sh - Phase 18: GSM8K with Latent Chain-of-Thought
#
# This experiment tests if the bridge can transfer mathematical reasoning
# from Llama to Mistral via latent chain-of-thought tokens.
#
# Architecture:
# - Llama reads question -> Bridge produces N reasoning steps
# - Each step: K latent tokens, attending to question + previous steps
# - Mistral receives all latent tokens and predicts final answer
#
# Success criteria:
# - > 5%: Some reasoning signal
# - > 15%: Bridge works for reasoning
# - > 30%: Excellent transfer
set -e

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/gsm8k_$(date +"%Y%m%d_%H%M%S")}"

# Architecture (from optimal SST-2/AG News config + CoT extension)
SOURCE_LAYER=31
SOFT_TOKENS=8      # Per reasoning step
COT_STEPS=4        # Number of reasoning steps
DEPTH=2
HEADS=8

# Training (more steps for reasoning task)
STEPS=5000
BATCH_SIZE=8
GRAD_ACCUM=4
LR=1e-4
WARMUP=200
EVAL_EVERY=250
SAVE_EVERY=500
DIVERSITY_WEIGHT=0.1

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/gsm8k_${TIMESTAMP}.log"

echo "============================================================"
echo "Phase 18: GSM8K with Latent Chain-of-Thought"
echo "============================================================"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""
echo "Architecture:"
echo "  Source layer: $SOURCE_LAYER"
echo "  Soft tokens per step: $SOFT_TOKENS"
echo "  CoT steps: $COT_STEPS"
echo "  Total latent tokens: $((SOFT_TOKENS * COT_STEPS))"
echo ""
echo "Training:"
echo "  Steps: $STEPS"
echo "  Batch size: $BATCH_SIZE x $GRAD_ACCUM (grad accum)"
echo "  Learning rate: $LR"
echo "============================================================"
echo ""

{
    # ============================================================
    # PHASE 1: Baselines
    # ============================================================
    echo "============================================================"
    echo "PHASE 1: GSM8K BASELINES"
    echo "============================================================"
    echo ""

    python telepathy/eval_gsm8k_baselines.py \
        --num_samples 200 \
        --output_dir "$OUTPUT_DIR"

    echo ""
    echo "Baselines complete!"
    echo ""

    # ============================================================
    # PHASE 2: Training
    # ============================================================
    echo "============================================================"
    echo "PHASE 2: TRAINING (Latent CoT Bridge)"
    echo "============================================================"
    echo ""

    python telepathy/train_telepathy_gsm8k.py \
        --source_layer $SOURCE_LAYER \
        --soft_tokens $SOFT_TOKENS \
        --cot_steps $COT_STEPS \
        --depth $DEPTH \
        --heads $HEADS \
        --lr $LR \
        --batch_size $BATCH_SIZE \
        --grad_accum $GRAD_ACCUM \
        --steps $STEPS \
        --warmup_steps $WARMUP \
        --eval_every $EVAL_EVERY \
        --save_every $SAVE_EVERY \
        --diversity_weight $DIVERSITY_WEIGHT \
        --save_path "$OUTPUT_DIR/bridge_gsm8k.pt" \
        --bf16

    echo ""
    echo "Training complete!"
    echo ""

    # ============================================================
    # PHASE 3: Final Evaluation
    # ============================================================
    echo "============================================================"
    echo "PHASE 3: FINAL EVALUATION"
    echo "============================================================"
    echo ""

    python telepathy/eval_telepathy_gsm8k.py \
        --checkpoint "$OUTPUT_DIR/bridge_gsm8k.pt" \
        --source_layer $SOURCE_LAYER \
        --soft_tokens $SOFT_TOKENS \
        --cot_steps $COT_STEPS \
        --depth $DEPTH \
        --heads $HEADS \
        --num_samples 500 \
        --output_dir "$OUTPUT_DIR" \
        --bf16

    echo ""
    echo "============================================================"
    echo "PHASE 18 COMPLETE!"
    echo "============================================================"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""

} 2>&1 | tee "$LOG_FILE"

# Git commit results
echo ""
echo "Committing results..."
git add "$OUTPUT_DIR"/*.json "$OUTPUT_DIR"/*.log 2>/dev/null || true
git commit -m "Phase 18 GSM8K Results: $(date +%Y%m%d)" 2>/dev/null || echo "No changes to commit"
git push 2>/dev/null || echo "Push failed - please push manually"

echo ""
echo "============================================================"
echo "All done! Check results in: $OUTPUT_DIR"
echo "============================================================"
