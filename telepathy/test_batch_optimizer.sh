#!/usr/bin/env bash
#
# Test script for batch size optimizer
# Shows safe batch sizes for various model combinations
#

set -e

# Set up environment
export PYTHONPATH=.

echo "=============================================================="
echo "Testing Batch Size Optimizer for Various Model Combinations"
echo "=============================================================="
echo ""

# Test small models (Llama-1B + Qwen-1.5B)
echo "1. Small Models: Llama-1B + Qwen-1.5B"
echo "--------------------------------------------------------------"
python telepathy/optimize_batch_size.py \
    --source-model "meta-llama/Llama-3.2-1B-Instruct" \
    --target-model "Qwen/Qwen2.5-1.5B-Instruct" \
    --verbose
echo ""

# Test medium models (Llama-3B + Mistral-7B)
echo "2. Medium Models: Llama-3B + Mistral-7B"
echo "--------------------------------------------------------------"
python telepathy/optimize_batch_size.py \
    --source-model "meta-llama/Llama-3.2-3B-Instruct" \
    --target-model "mistralai/Mistral-7B-Instruct-v0.3" \
    --verbose
echo ""

# Test large models (Llama-8B + Mistral-7B)
echo "3. Large Models: Llama-8B + Mistral-7B"
echo "--------------------------------------------------------------"
python telepathy/optimize_batch_size.py \
    --source-model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --target-model "mistralai/Mistral-7B-Instruct-v0.3" \
    --verbose
echo ""

# Test very large combination (Llama-8B + Qwen-7B)
echo "4. Very Large: Llama-8B + Qwen-7B"
echo "--------------------------------------------------------------"
python telepathy/optimize_batch_size.py \
    --source-model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --target-model "Qwen/Qwen2.5-7B-Instruct" \
    --verbose
echo ""

# Test with different safety margins
echo "5. Testing Different Safety Margins for Llama-8B + Qwen-7B"
echo "--------------------------------------------------------------"
echo "With 10% safety margin:"
python telepathy/optimize_batch_size.py \
    --source-model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --target-model "Qwen/Qwen2.5-7B-Instruct" \
    --safety-margin 0.1

echo ""
echo "With 30% safety margin (conservative):"
python telepathy/optimize_batch_size.py \
    --source-model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --target-model "Qwen/Qwen2.5-7B-Instruct" \
    --safety-margin 0.3
echo ""

# JSON output example
echo "6. JSON Output Example (for integration)"
echo "--------------------------------------------------------------"
python telepathy/optimize_batch_size.py \
    --source-model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --target-model "Qwen/Qwen2.5-7B-Instruct" \
    --json
echo ""

echo "=============================================================="
echo "Test Complete!"
echo "=============================================================="