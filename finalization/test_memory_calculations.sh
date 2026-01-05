#!/usr/bin/env bash
# Test memory calculations for various GPU configurations

set -e

echo "=============================================================="
echo "Memory-Safe Batch Size Calculations for LatentWire"
echo "=============================================================="
echo ""
echo "Testing memory calculator with various GPU configurations..."
echo ""

# Test different GPU configurations
echo "1. A100 40GB (single GPU):"
echo "----------------------------"
python utils/memory_calculator.py \
    --model_size_gb 14 \
    --gpu_memory_gb 40 \
    --num_gpus 1 \
    --target_batch 32
echo ""

echo "2. A100 80GB (single GPU):"
echo "----------------------------"
python utils/memory_calculator.py \
    --model_size_gb 14 \
    --gpu_memory_gb 80 \
    --num_gpus 1 \
    --target_batch 32
echo ""

echo "3. H100 80GB (single GPU):"
echo "----------------------------"
python utils/memory_calculator.py \
    --model_size_gb 14 \
    --gpu_memory_gb 80 \
    --num_gpus 1 \
    --target_batch 32
echo ""

echo "4. H100 80GB (4 GPUs):"
echo "------------------------"
python utils/memory_calculator.py \
    --model_size_gb 14 \
    --gpu_memory_gb 80 \
    --num_gpus 4 \
    --target_batch 64
echo ""

echo "5. V100 32GB (single GPU) - Should fail:"
echo "------------------------------------------"
python utils/memory_calculator.py \
    --model_size_gb 14 \
    --gpu_memory_gb 32 \
    --num_gpus 1 \
    --target_batch 32
echo ""

echo "=============================================================="
echo "Key Insights:"
echo "=============================================================="
echo "• Adam optimizer state adds ~28GB for an 8B model (2x params)"
echo "• Total fixed overhead is ~60GB (model + optimizer + gradients)"
echo "• A100 40GB cannot fit the full model with Adam optimizer"
echo "• H100/A100 80GB can handle batch size ~20 safely"
echo "• Gradient checkpointing or model parallelism needed for smaller GPUs"
echo ""
echo "Recommendations:"
echo "• Use gradient accumulation for smaller GPUs"
echo "• Consider Adafactor optimizer for memory efficiency"
echo "• Enable gradient checkpointing to reduce activation memory"
echo "=============================================================="