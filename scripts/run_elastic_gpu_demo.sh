#!/usr/bin/env bash
set -e

# =============================================================================
# Elastic GPU Configuration Demo Script
# =============================================================================
# This script demonstrates the elastic GPU configuration system that
# automatically adapts to 1-4 GPUs.
#
# Usage:
#   bash scripts/run_elastic_gpu_demo.sh
#
# The script will:
# 1. Detect available GPUs
# 2. Show optimal configuration for your hardware
# 3. Run a quick training test with elastic settings
# =============================================================================

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-runs/elastic_gpu_demo}"
DATASET="${DATASET:-squad}"
SAMPLES="${SAMPLES:-500}"
MAX_STEPS="${MAX_STEPS:-10}"

# Set up environment
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=============================================================="
echo "ELASTIC GPU CONFIGURATION DEMO"
echo "=============================================================="
echo ""

# Step 1: Test configuration detection
echo "Step 1: Detecting GPU configuration"
echo "--------------------------------------------------------------"
python scripts/test_elastic_gpu.py | tee "$OUTPUT_DIR/gpu_detection.log"

echo ""
echo "=============================================================="
echo "Step 2: Running Quick Training Test with Elastic Config"
echo "=============================================================="
echo ""

# Set up log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/elastic_training_${TIMESTAMP}.log"

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Samples: $SAMPLES"
echo "  Max steps: $MAX_STEPS (for quick test)"
echo "  Output: $OUTPUT_DIR"
echo "  Log: $LOG_FILE"
echo ""

# Run training with elastic GPU configuration
{
    python latentwire/train.py \
        --elastic_gpu \
        --elastic_base_batch 64 \
        --elastic_target_util 0.75 \
        --llama_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --qwen_id "Qwen/Qwen2.5-7B-Instruct" \
        --samples "$SAMPLES" \
        --epochs 1 \
        --dataset "$DATASET" \
        --output_dir "$OUTPUT_DIR/checkpoint" \
        --latent_len 32 \
        --d_z 256 \
        --enabled_models llama \
        --max_steps "$MAX_STEPS" \
        --warm_anchor_text "Answer: " \
        --first_token_ce_weight 0.5
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "=============================================================="
echo "Step 3: Configuration Summary"
echo "=============================================================="
echo ""

# Extract key metrics from log
python -c "
import re

log_file = '$LOG_FILE'
try:
    with open(log_file, 'r') as f:
        content = f.read()

    # Extract configuration details
    gpu_match = re.search(r'Detected (\d+) GPU', content)
    strategy_match = re.search(r'Strategy: (\S+)', content)
    batch_match = re.search(r'Batch size per step: (\d+)', content)
    accum_match = re.search(r'Gradient accumulation: (\d+)', content)
    effective_match = re.search(r'Effective batch size: (\d+)', content)

    print('ELASTIC GPU CONFIGURATION APPLIED:')
    print('-' * 40)

    if gpu_match:
        print(f'GPUs detected: {gpu_match.group(1)}')
    if strategy_match:
        print(f'Strategy used: {strategy_match.group(1)}')
    if batch_match:
        print(f'Batch size: {batch_match.group(1)}')
    if accum_match:
        print(f'Gradient accumulation: {accum_match.group(1)}')
    if effective_match:
        print(f'Effective batch size: {effective_match.group(1)}')

    # Look for throughput
    throughput_matches = re.findall(r'(\d+\.?\d*) samples/sec', content)
    if throughput_matches:
        print(f'Throughput: {throughput_matches[-1]} samples/sec')

    # Memory utilization
    mem_matches = re.findall(r'Peak allocated: (\d+\.?\d*) GB', content)
    if mem_matches:
        print(f'Peak memory: {mem_matches[-1]} GB')

except Exception as e:
    print(f'Error parsing log: {e}')
"

echo ""
echo "=============================================================="
echo "Demo Complete!"
echo "=============================================================="
echo ""
echo "The elastic GPU configuration has automatically:"
echo "1. Detected your available GPUs"
echo "2. Calculated optimal batch size and accumulation"
echo "3. Configured model parallelism if beneficial"
echo "4. Maximized throughput for your hardware"
echo ""
echo "To use in your own training:"
echo "  python latentwire/train.py --elastic_gpu [other args]"
echo ""
echo "Results saved to: $OUTPUT_DIR"