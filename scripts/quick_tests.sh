#!/bin/bash
# Quick tests to understand the problem
set -euo pipefail

echo "==================================="
echo "QUICK DIAGNOSTIC TESTS"
echo "==================================="
echo ""

# Test 1: Check if your existing model works at all
echo "TEST 1: Analyze existing checkpoint"
echo "-----------------------------------"
if [ -d "runs/optimized/lora_20ep_best" ]; then
    echo "Found checkpoint at runs/optimized/lora_20ep_best"
    python scripts/progressive_ablation.sh runs/optimized/lora_20ep_best
else
    echo "No checkpoint found. Train one first with:"
    echo "  bash scripts/run_optimized_h100.sh"
fi
echo ""

# Test 2: Verify embeddings work
echo "TEST 2: Verify embedding baseline"
echo "-----------------------------------"
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
print(f'Loading {model_id}...')
model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Test
text = 'The capital of France is'
inputs = tokenizer(text, return_tensors='pt')
embeds = model.get_input_embeddings()(inputs.input_ids.to(model.device))

with torch.no_grad():
    outputs = model.generate(inputs_embeds=embeds, max_new_tokens=2, do_sample=False)
    result = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):])

print(f'Input: {text}')
print(f'Output: {result}')
print(f'✅ SUCCESS: Embeddings work!' if 'Paris' in result else '❌ FAILURE: Even embeddings dont work!')
"
echo ""

# Test 3: Show why we need compression
echo "TEST 3: Why compression is needed"
echo "-----------------------------------"
python -c "
import torch
import torch.nn as nn

print('Without compression:')
print('  Input: [B, L, 4096] → Adapter → [B, L, 4096]')
print('  This is just learning y = x (identity)')
print('  No meaningful task!')
print('')
print('With compression:')
print('  Input: [B, L, 4096] → PCA → [B, L, 512] → Adapter → [B, L, 4096]')
print('  Adapter must learn to reconstruct from 8x less information')
print('  This tests if adapter can handle compressed representations')
print('')
print('The compression is what makes it a meaningful test!')
"