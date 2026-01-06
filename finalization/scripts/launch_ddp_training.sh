#!/bin/bash
# Launch script for DDP (Distributed Data Parallel) training
# Supports 1-4 GPUs elastically

set -e

# Configuration
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l 2>/dev/null || echo 1)}
MASTER_PORT=${MASTER_PORT:-29500}
OUTPUT_DIR=${OUTPUT_DIR:-"runs/ddp_experiment"}

echo "============================================================"
echo "DDP Training Launcher"
echo "============================================================"
echo "Detected GPUs: $NUM_GPUS"
echo "Master port: $MASTER_PORT"
echo "Output directory: $OUTPUT_DIR"
echo "============================================================"

# Check if GPUs are available
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. Running in CPU mode."
    NUM_GPUS=1
    USE_CPU="--require_cuda no"
else
    nvidia-smi
    USE_CPU=""
fi

# Set environment variables for better GPU utilization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export TORCH_DTYPE=bfloat16

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/train_ddp_${TIMESTAMP}.log"

echo ""
echo "Starting DDP training with $NUM_GPUS GPU(s)..."
echo "Log file: $LOG_FILE"
echo ""

# Training arguments
TRAIN_ARGS="
    --llama_id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --qwen_id Qwen/Qwen2.5-7B-Instruct \
    --samples 1000 \
    --epochs 2 \
    --batch_size 16 \
    --grad_accum_steps 2 \
    --latent_len 32 \
    --d_z 256 \
    --lr 1e-4 \
    --encoder_type byte \
    --dataset squad \
    --elastic_gpu \
    --save_dir $OUTPUT_DIR \
    --diagnostic_log $OUTPUT_DIR/diagnostics.jsonl \
    $USE_CPU
"

# Choose launch method based on number of GPUs
if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Running in single GPU mode (no DDP)..."
    {
        python -m latentwire.train $TRAIN_ARGS
    } 2>&1 | tee "$LOG_FILE"

elif [ "$NUM_GPUS" -gt 1 ]; then
    echo "Running in multi-GPU DDP mode with $NUM_GPUS GPUs..."

    # Check if torchrun is available
    if command -v torchrun &> /dev/null; then
        echo "Using torchrun for DDP launch..."
        {
            torchrun \
                --nproc_per_node=$NUM_GPUS \
                --master_port=$MASTER_PORT \
                --nnodes=1 \
                --node_rank=0 \
                -m latentwire.train \
                $TRAIN_ARGS
        } 2>&1 | tee "$LOG_FILE"
    else
        echo "torchrun not found, using torch.distributed.launch..."
        {
            python -m torch.distributed.launch \
                --nproc_per_node=$NUM_GPUS \
                --master_port=$MASTER_PORT \
                --use_env \
                -m latentwire.train \
                $TRAIN_ARGS
        } 2>&1 | tee "$LOG_FILE"
    fi
else
    echo "ERROR: Invalid number of GPUs: $NUM_GPUS"
    exit 1
fi

echo ""
echo "============================================================"
echo "Training complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "============================================================"

# Print summary of results if they exist
if [ -f "$OUTPUT_DIR/config.json" ]; then
    echo ""
    echo "Configuration summary:"
    python -c "import json; print(json.dumps(json.load(open('$OUTPUT_DIR/config.json')), indent=2))" 2>/dev/null || true
fi

if [ -f "$OUTPUT_DIR/training_stats.json" ]; then
    echo ""
    echo "Training statistics:"
    python -c "import json; print(json.dumps(json.load(open('$OUTPUT_DIR/training_stats.json')), indent=2))" 2>/dev/null || true
fi