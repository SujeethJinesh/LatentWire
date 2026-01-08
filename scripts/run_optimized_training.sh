#!/usr/bin/env bash
set -e

# =============================================================================
# Optimized Training Configuration for 1-2 GPU Setup
# =============================================================================
# This script is optimized to run efficiently on 1-2 GPUs within 8-12 hours
# using gradient accumulation and reduced samples while maintaining quality.
#
# Optimizations applied:
# 1. Gradient accumulation to simulate larger batch sizes
# 2. Reduced sample count (5000 -> 2000) with focused learning
# 3. Elastic GPU configuration for automatic hardware adaptation
# 4. Mixed precision training for faster computation
# 5. Efficient checkpoint saving (only best and final)
#
# Expected runtime:
#   - 1 GPU: ~10-12 hours
#   - 2 GPUs: ~6-8 hours
# =============================================================================

# Configuration - OPTIMIZED VALUES
EXPERIMENT_NAME="${EXPERIMENT_NAME:-optimized_training}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/$EXPERIMENT_NAME}"

# Reduced samples for faster training while maintaining quality
SAMPLES=2000  # Reduced from 5000 (60% reduction)
EPOCHS=6      # Reduced from 8 (25% reduction)
EVAL_SAMPLES=200  # Quick validation

# Model configuration
LLAMA_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"
QWEN_ID="Qwen/Qwen2.5-7B-Instruct"
DATASET="squad"

# Optimized hyperparameters
BATCH_SIZE=16  # Small per-GPU batch size to fit in memory
GRAD_ACCUM_STEPS=4  # Accumulate to effective batch of 64
LATENT_LEN=32
D_Z=256
LR=5e-4  # Slightly higher LR for fewer steps
FIRST_TOKEN_CE_WEIGHT=0.5
K_TOKENS=4

# Hardware optimization flags
ENABLE_ELASTIC="${ENABLE_ELASTIC:-yes}"  # Auto-adapt to GPU count
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"  # bf16 for stability
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-yes}"  # Save memory

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Auto-detect GPU count for configuration
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
echo "Detected $GPU_COUNT GPU(s)"

# Adjust configuration based on GPU count
if [ "$GPU_COUNT" -eq "0" ]; then
    echo "ERROR: No GPUs detected. This training requires CUDA GPUs."
    exit 1
elif [ "$GPU_COUNT" -eq "1" ]; then
    # Single GPU - maximize gradient accumulation
    GRAD_ACCUM_STEPS=8  # Effective batch = 128
    BATCH_SIZE=16
    ENABLED_MODELS="llama"  # Train only Llama to save memory
    echo "Single GPU mode: Using gradient accumulation (8 steps) and Llama only"
elif [ "$GPU_COUNT" -eq "2" ]; then
    # Dual GPU - balanced approach
    GRAD_ACCUM_STEPS=2  # Effective batch = 64
    BATCH_SIZE=16
    ENABLED_MODELS="llama,qwen"  # Can handle both models
    LLAMA_DEVICES="0"
    QWEN_DEVICES="1"
    echo "Dual GPU mode: Models split across GPUs"
else
    # 3+ GPUs - use elastic configuration
    GRAD_ACCUM_STEPS=1
    BATCH_SIZE=32
    ENABLED_MODELS="llama,qwen"
    echo "Multi-GPU mode: Using elastic configuration"
fi

# Create output directory and log file
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/optimized_training_${TIMESTAMP}.log"

echo "=============================================================="
echo "OPTIMIZED TRAINING CONFIGURATION"
echo "=============================================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo ""
echo "Training Configuration:"
echo "  Samples: $SAMPLES (reduced for efficiency)"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE per GPU"
echo "  Gradient accumulation: $GRAD_ACCUM_STEPS steps"
echo "  Effective batch size: $((BATCH_SIZE * GRAD_ACCUM_STEPS * GPU_COUNT))"
echo "  Models: $ENABLED_MODELS"
echo ""
echo "Optimizations:"
echo "  - Mixed precision: $MIXED_PRECISION"
echo "  - Gradient checkpointing: $GRADIENT_CHECKPOINTING"
echo "  - Elastic GPU: $ENABLE_ELASTIC"
echo ""
echo "Expected runtime: 8-12 hours"
echo "=============================================================="
echo ""

# Build training command
TRAIN_CMD="python latentwire/train.py \
    --llama_id '$LLAMA_ID' \
    --qwen_id '$QWEN_ID' \
    --samples $SAMPLES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --grad_accum_steps $GRAD_ACCUM_STEPS \
    --latent_len $LATENT_LEN \
    --d_z $D_Z \
    --encoder_type byte \
    --dataset $DATASET \
    --output_dir '$OUTPUT_DIR/checkpoints' \
    --warm_anchor_text 'Answer: ' \
    --first_token_ce_weight $FIRST_TOKEN_CE_WEIGHT \
    --k $K_TOKENS \
    --lr $LR \
    --enabled_models $ENABLED_MODELS \
    --save_every_n_epochs 2 \
    --diagnostic_log '$OUTPUT_DIR/diagnostics.jsonl'"

# Add hardware-specific flags
if [ "$ENABLE_ELASTIC" = "yes" ] && [ "$GPU_COUNT" -gt "1" ]; then
    TRAIN_CMD="$TRAIN_CMD --elastic_gpu --elastic_base_batch 64 --elastic_target_util 0.75"
fi

if [ "$MIXED_PRECISION" = "bf16" ]; then
    TRAIN_CMD="$TRAIN_CMD --mixed_precision bf16"
elif [ "$MIXED_PRECISION" = "fp16" ]; then
    TRAIN_CMD="$TRAIN_CMD --mixed_precision fp16 --grad_scaler_init 32768"
fi

if [ "$GRADIENT_CHECKPOINTING" = "yes" ]; then
    TRAIN_CMD="$TRAIN_CMD --grad_ckpt"
fi

if [ -n "$LLAMA_DEVICES" ]; then
    TRAIN_CMD="$TRAIN_CMD --llama_devices '$LLAMA_DEVICES'"
fi

if [ -n "$QWEN_DEVICES" ]; then
    TRAIN_CMD="$TRAIN_CMD --qwen_devices '$QWEN_DEVICES'"
fi

# Function to monitor training progress
monitor_progress() {
    echo ""
    echo "Monitoring training progress..."

    # Wait for diagnostics file to be created
    while [ ! -f "$OUTPUT_DIR/diagnostics.jsonl" ]; do
        sleep 10
    done

    # Periodically report progress
    while true; do
        if [ -f "$OUTPUT_DIR/diagnostics.jsonl" ]; then
            # Get latest metrics
            LAST_LINE=$(tail -1 "$OUTPUT_DIR/diagnostics.jsonl" 2>/dev/null || echo "{}")
            if [ "$LAST_LINE" != "{}" ]; then
                python -c "
import json
data = json.loads('$LAST_LINE')
if 'epoch' in data:
    print(f\"Progress: Epoch {data['epoch']}/{$EPOCHS}, Step {data.get('global_step', 0)}\")
    if 'loss' in data:
        print(f\"  Loss: {data['loss']:.4f}\")
    if 'first_tok_acc' in data:
        print(f\"  First Token Acc: {data['first_tok_acc']:.2%}\")
"
            fi
        fi
        sleep 60  # Update every minute
    done
}

# Start background monitor (optional)
if [ "${ENABLE_MONITOR:-no}" = "yes" ]; then
    monitor_progress &
    MONITOR_PID=$!
fi

# Run training with output capture
echo "Starting optimized training..."
echo "Command: $TRAIN_CMD"
echo ""

{
    eval $TRAIN_CMD
} 2>&1 | tee "$LOG_FILE"

# Kill monitor if running
if [ -n "$MONITOR_PID" ]; then
    kill $MONITOR_PID 2>/dev/null || true
fi

echo ""
echo "=============================================================="
echo "Training Complete!"
echo "=============================================================="

# Quick evaluation on final checkpoint
if [ -d "$OUTPUT_DIR/checkpoints/epoch$EPOCHS" ]; then
    echo "Running final evaluation..."

    EVAL_CMD="python latentwire/eval.py \
        --ckpt '$OUTPUT_DIR/checkpoints/epoch$EPOCHS' \
        --samples $EVAL_SAMPLES \
        --max_new_tokens 12 \
        --dataset $DATASET \
        --sequential_eval \
        --fresh_eval \
        --calibration embed_rms \
        --latent_anchor_mode text \
        --latent_anchor_text 'Answer: ' \
        --append_bos_after_prefix yes \
        --output_file '$OUTPUT_DIR/final_eval.json'"

    {
        eval $EVAL_CMD
    } 2>&1 | tee -a "$LOG_FILE"

    # Report final metrics
    if [ -f "$OUTPUT_DIR/final_eval.json" ]; then
        echo ""
        echo "Final Metrics:"
        python -c "
import json
with open('$OUTPUT_DIR/final_eval.json', 'r') as f:
    data = json.load(f)
    if 'aggregate_stats' in data:
        stats = data['aggregate_stats']
        print(f'  Latent F1: {stats.get(\"latent_f1_mean\", 0):.4f}')
        print(f'  Text F1: {stats.get(\"text_f1_mean\", 0):.4f}')
        print(f'  First Token Accuracy: {stats.get(\"first_tok_acc\", 0):.2%}')
        print(f'  Compression Ratio: {stats.get(\"compression_ratio\", 0):.2f}x')
"
    fi
fi

echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Training log: $LOG_FILE"
echo ""

# Memory usage summary
if [ "$GPU_COUNT" -gt "0" ]; then
    echo "GPU Memory Usage:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F',' '{printf "  GPU %s (%s): %s MB / %s MB (%.1f%%)\n", $1, $2, $3, $4, ($3/$4)*100}'
fi

echo ""
echo "=============================================================="
echo "Optimization Summary:"
echo "=============================================================="
echo "This training used the following optimizations:"
echo "1. Gradient accumulation: $GRAD_ACCUM_STEPS steps (effective batch: $((BATCH_SIZE * GRAD_ACCUM_STEPS * GPU_COUNT)))"
echo "2. Reduced samples: $SAMPLES (vs typical 5000-10000)"
echo "3. Mixed precision: $MIXED_PRECISION"
echo "4. Smart model placement: $ENABLED_MODELS"
if [ "$GPU_COUNT" -eq "1" ]; then
    echo "5. Single model training (Llama only) to fit in memory"
fi
echo ""
echo "These optimizations allow training on limited hardware (1-2 GPUs)"
echo "while maintaining reasonable quality for research purposes."
echo "=============================================================="