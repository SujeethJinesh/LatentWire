#!/usr/bin/env bash
set -e

# Activate ARM64 venv
source /Users/sujeethjinesh/Desktop/LatentWire/venv_arm64/bin/activate

# Set up paths
cd /Users/sujeethjinesh/Desktop/LatentWire/experimental/learning/reproduce_coconut/coconut

# Create output directory
mkdir -p ../runs/stage0

# Set log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="../runs/stage0/training_${TIMESTAMP}.log"

echo "Starting Stage 0 (CoT baseline) training..."
echo "Log file: $LOG_FILE"
echo ""

# Run training with torchrun (single GPU mode)
# Using gloo backend instead of nccl for CPU/MPS
torchrun --nproc_per_node=1 --master_port=29500 run.py args/gsm_cot_test.yaml 2>&1 | tee "$LOG_FILE"

echo ""
echo "Training complete! Log saved to: $LOG_FILE"
