#!/usr/bin/env bash
#
# LatentWire Training - Optimized for 4x H100
#
# Trains the full LatentWire system (encoder + adapters) for compressed interlingua.
# Optimized for 4 H100 GPUs with high batch size and proper parallelism.
#
# Usage: PYTHONPATH=. bash scripts/experiments/train_latentwire.sh
#

set -e

echo "========================================================================"
echo "LATENTWIRE TRAINING - Compressed Interlingua"
echo "========================================================================"
echo ""
echo "Training encoder + adapters to create compressed representations that"
echo "condition both Llama and Qwen with minimal information loss."
echo ""
echo "System architecture:"
echo "  Text → Encoder → Z (M × d_z) → Adapter → Soft Embeds → LLM → Answer"
echo ""
echo "========================================================================"
echo ""

# Configuration
LLAMA_ID="${LLAMA_ID:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
QWEN_ID="${QWEN_ID:-Qwen/Qwen2.5-7B-Instruct}"
DATASET="${DATASET:-squad}"
SAMPLES="${SAMPLES:-87599}"  # Full SQuAD training set
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-32}"  # Increased from 8 (20% GPU usage)
LATENT_LEN="${LATENT_LEN:-32}"
D_Z="${D_Z:-256}"
ENCODER_TYPE="${ENCODER_TYPE:-byte}"
LR="${LR:-1e-4}"
FIRST_TOKEN_CE_WEIGHT="${FIRST_TOKEN_CE_WEIGHT:-0.5}"
K="${K:-4}"  # K-token teacher forcing
KD_WEIGHT="${KD_WEIGHT:-0.1}"  # Knowledge distillation weight
OUTPUT_DIR="${OUTPUT_DIR:-runs/experiments/latentwire}"

echo "Configuration:"
echo "  LLAMA_ID: $LLAMA_ID"
echo "  QWEN_ID: $QWEN_ID"
echo "  DATASET: $DATASET"
echo "  SAMPLES: $SAMPLES"
echo "  EPOCHS: $EPOCHS"
echo "  BATCH_SIZE: $BATCH_SIZE (optimized for 4x H100)"
echo "  LATENT_LEN (M): $LATENT_LEN"
echo "  D_Z: $D_Z"
echo "  ENCODER_TYPE: $ENCODER_TYPE"
echo "  LR: $LR"
echo "  FIRST_TOKEN_CE_WEIGHT: $FIRST_TOKEN_CE_WEIGHT"
echo "  K_TOKENS: $K"
echo "  KD_WEIGHT: $KD_WEIGHT"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo ""

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/train_${TIMESTAMP}.log"

echo "========================================================================"
echo "Starting training..."
echo "Log file: $LOG_FILE"
echo "========================================================================"
echo ""

# Run training
{
python latentwire/train.py \
    --llama_id "$LLAMA_ID" \
    --qwen_id "$QWEN_ID" \
    --samples "$SAMPLES" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --latent_len "$LATENT_LEN" \
    --d_z "$D_Z" \
    --encoder_type "$ENCODER_TYPE" \
    --dataset "$DATASET" \
    --sequential_models \
    --lr "$LR" \
    --warm_anchor_text "Answer: " \
    --first_token_ce_weight "$FIRST_TOKEN_CE_WEIGHT" \
    --K "$K" \
    --kd_first_k_weight "$KD_WEIGHT" \
    --save_dir "$OUTPUT_DIR" \
    --save_every 1000

echo ""
echo "========================================================================"
echo "Training complete!"
echo "========================================================================"
echo ""
} 2>&1 | tee "$LOG_FILE"

echo "Results saved to:"
echo "  - $OUTPUT_DIR/ (checkpoints)"
echo "  - $LOG_FILE"
echo ""

# Find best checkpoint
BEST_CKPT=$(ls -d "$OUTPUT_DIR"/checkpoint_best 2>/dev/null | head -1)
if [ -n "$BEST_CKPT" ]; then
    echo "Best checkpoint: $BEST_CKPT"
    echo ""
    echo "To evaluate:"
    echo "  python latentwire/eval.py --ckpt $BEST_CKPT --samples 1000"
fi
echo ""
