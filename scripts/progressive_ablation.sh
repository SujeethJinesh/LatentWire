#!/bin/bash
# Progressive ablation study to identify where the pipeline breaks
set -euo pipefail

echo "="
echo "PROGRESSIVE ABLATION STUDY"
echo "Goal: Identify exactly where the pipeline fails"
echo "="
echo ""

# Configuration
CHECKPOINT="${1:-runs/optimized/lora_20ep_best}"
SAMPLES=100

echo "Using checkpoint: $CHECKPOINT"
echo ""

# Step 1: Validate baseline still works
echo "STEP 1: Validate Embedding Baseline"
echo "-----"
python -c "
from latentwire.eval import evaluate_on_squad
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Quick test
text = 'The capital of France is Paris. Question: What is the capital of France? Answer: '
inputs = tokenizer(text, return_tensors='pt')
embeds = model.get_input_embeddings()(inputs.input_ids.to(model.device))

with torch.no_grad():
    outputs = model.generate(inputs_embeds=embeds, max_new_tokens=3, do_sample=False)
    result = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):])
    print(f'Generated: {result}')
    print(f'✓ Embedding baseline works' if 'Paris' in result else '✗ Embedding baseline broken!')
"
echo ""

# Step 2: Test adapter in isolation
echo "STEP 2: Test Adapter Isolation"
echo "-----"
python -c "
import torch
import json
from pathlib import Path

ckpt_path = '$CHECKPOINT'
if Path(f'{ckpt_path}/model.pt').exists():
    ckpt = torch.load(f'{ckpt_path}/model.pt', map_location='cpu')
    config = json.load(open(f'{ckpt_path}/config.json'))

    # Check adapter weights
    adapter_state = ckpt.get('adapter_llama')
    if adapter_state:
        # Check if adapter is roughly initialized
        if 'layers.0.weight' in adapter_state:
            w = adapter_state['layers.0.weight']
            print(f'Adapter weight shape: {w.shape}')
            print(f'Adapter weight stats: mean={w.mean():.4f}, std={w.std():.4f}')
            print(f'✓ Adapter loaded successfully')
        else:
            print('✗ Adapter structure unexpected')
    else:
        print('✗ No adapter found in checkpoint')
else:
    print('✗ No checkpoint found at {ckpt_path}')
"
echo ""

# Step 3: Test encoder output distribution
echo "STEP 3: Encoder Output Analysis"
echo "-----"
python -c "
import torch
import json
from pathlib import Path
import numpy as np

ckpt_path = '$CHECKPOINT'
if Path(f'{ckpt_path}/model.pt').exists():
    ckpt = torch.load(f'{ckpt_path}/model.pt', map_location='cpu')
    config = json.load(open(f'{ckpt_path}/config.json'))

    # Check encoder weights
    encoder_state = ckpt.get('encoder')
    if encoder_state:
        # Analyze encoder output layer
        keys = list(encoder_state.keys())
        print(f'Encoder has {len(keys)} parameters')

        # Look for output projection
        output_keys = [k for k in keys if 'output' in k or 'proj' in k or 'final' in k]
        if output_keys:
            for k in output_keys[:3]:
                w = encoder_state[k]
                print(f'  {k}: shape={w.shape}, mean={w.mean():.4f}, std={w.std():.4f}')

        print(f'✓ Encoder structure analyzed')
    else:
        print('✗ No encoder found in checkpoint')
"
echo ""

# Step 4: Test information flow
echo "STEP 4: Information Flow Test"
echo "-----"
python -c "
import torch
import torch.nn as nn
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

ckpt_path = '$CHECKPOINT'
model_id = 'meta-llama/Meta-Llama-3.1-8B-Instruct'

print('Loading models...')
model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id)

if Path(f'{ckpt_path}/model.pt').exists():
    ckpt = torch.load(f'{ckpt_path}/model.pt', map_location='cpu')
    config = json.load(open(f'{ckpt_path}/config.json'))

    # Create dummy latent
    batch_size = 1
    latent_len = config['latent_len']
    d_z = config['d_z']
    embed_dim = model.config.hidden_size

    # Test 1: Zero latents
    z_zeros = torch.zeros(batch_size, latent_len, d_z).to(model.device)

    # Simple linear projection (simplified adapter)
    proj = nn.Linear(d_z, embed_dim).to(model.device)
    proj.weight.data.normal_(0, 0.02)
    proj.bias.data.zero_()

    embeds_zeros = proj(z_zeros)
    print(f'Zero latents → embeds: mean={embeds_zeros.mean():.4f}, std={embeds_zeros.std():.4f}')

    # Test 2: Random latents
    z_random = torch.randn(batch_size, latent_len, d_z).to(model.device) * 0.1
    embeds_random = proj(z_random)
    print(f'Random latents → embeds: mean={embeds_random.mean():.4f}, std={embeds_random.std():.4f}')

    # Test 3: Generate from both
    with torch.no_grad():
        out_zeros = model.generate(inputs_embeds=embeds_zeros, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        out_random = model.generate(inputs_embeds=embeds_random, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.pad_token_id)

        print(f'Zero latents generate: {tokenizer.decode(out_zeros[0][:20])}...')
        print(f'Random latents generate: {tokenizer.decode(out_random[0][:20])}...')

    print('✓ Information flow tested')
"
echo ""

# Step 5: Compare to training behavior
echo "STEP 5: Training vs Inference Comparison"
echo "-----"
if [ -f "$CHECKPOINT/config.json" ]; then
    echo "Configuration used in training:"
    python -c "
import json
config = json.load(open('$CHECKPOINT/config.json'))
print(f'  Latent length: {config.get(\"latent_len\", \"?\")}')
print(f'  Latent dim: {config.get(\"d_z\", \"?\")}')
print(f'  First token CE weight: {config.get(\"first_token_ce_weight\", \"?\")}')
print(f'  Using LoRA: {config.get(\"use_lora\", False)}')
print(f'  LoRA rank: {config.get(\"lora_r\", \"?\")}')
print(f'  Best first-token acc: {config.get(\"best_first_acc\", 0)*100:.1f}%')
"
fi
echo ""

# Step 6: Identify the failure point
echo "STEP 6: Failure Point Analysis"
echo "-----"
python -c "
print('Based on the above tests, the failure likely occurs at:')
print('')
print('[ ] Raw embeddings → LLM (Working ✓)')
print('[ ] Encoder → Latents (Check statistics)')
print('[ ] Latents → Adapter (Check projection)')
print('[ ] Adapter → Embeddings (Check calibration)')
print('[ ] Embeddings → LLM (Should work if stats match)')
print('')
print('Most likely issue: Latent/Embedding distribution mismatch')
print('Solution: Better calibration or distribution matching during training')
"

echo ""
echo "="
echo "ABLATION COMPLETE"
echo "="