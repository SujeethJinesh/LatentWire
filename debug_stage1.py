#!/usr/bin/env python3
"""
Debug Stage 1 Phase 1: Understand why F1 is 0% despite improving reconstruction.

Hypothesis: Cosine similarity of 0.17 is too low for coherent generation.
We need >0.9 for the model to actually understand the reconstructed embeddings.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from latentwire.data import load_squad_subset
from latentwire.models import Adapter
import numpy as np

# Load model (small for testing)
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
device = next(model.parameters()).device

# Load one example
dataset = load_squad_subset("train", 5, seed=42)
item = dataset[0]
text = item['source'] + "Answer: "
answer = item['answer']

print(f"\nQuestion: {text[:200]}...")
print(f"Expected answer: {answer}")

# Tokenize
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
input_ids = inputs.input_ids.to(device)

# Get original embeddings
with torch.no_grad():
    orig_embeds = model.get_input_embeddings()(input_ids)

    # Generate from ORIGINAL embeddings (should work)
    print("\n" + "="*60)
    print("1. BASELINE: Generate from ORIGINAL embeddings")
    print("="*60)

    attention_mask = torch.ones(
        orig_embeds.shape[0], orig_embeds.shape[1],
        dtype=torch.long, device=device
    )

    outputs = model.generate(
        inputs_embeds=orig_embeds,
        attention_mask=attention_mask,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )

    baseline_gen = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
    print(f"Generated: {baseline_gen}")

    # Test random projection + untrained adapter
    print("\n" + "="*60)
    print("2. Random Projection + Untrained Adapter")
    print("="*60)

    # Random projection
    compress_dim = 512
    projection = torch.randn(4096, compress_dim, device=device, dtype=torch.bfloat16)
    projection = projection / np.sqrt(compress_dim)

    compressed = orig_embeds @ projection
    print(f"Compression: {orig_embeds.shape} -> {compressed.shape}")

    # Untrained adapter
    adapter = Adapter(
        d_z=compress_dim,
        d_model=4096,
        latent_length=32,
        hidden_mult=4,
        dropout=0.0,
        enable_metadata=False,
        colorize=False
    ).to(device).to(torch.bfloat16)

    reconstructed = adapter(compressed)

    # Check reconstruction quality
    mse = F.mse_loss(reconstructed, orig_embeds).item()
    cos_sim = F.cosine_similarity(reconstructed.flatten(0, 1), orig_embeds.flatten(0, 1), dim=-1).mean().item()

    print(f"MSE: {mse:.4f}")
    print(f"Cosine similarity: {cos_sim:.4f}")

    # Generate
    outputs = model.generate(
        inputs_embeds=reconstructed,
        attention_mask=attention_mask,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )

    untrained_gen = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
    print(f"Generated: {untrained_gen}")

    # Test with NOISE added to original embeddings
    print("\n" + "="*60)
    print("3. NOISE EXPERIMENT: Add controlled noise to original")
    print("="*60)

    for noise_level in [0.1, 0.5, 1.0, 2.0]:
        noise = torch.randn_like(orig_embeds) * noise_level
        noisy_embeds = orig_embeds + noise

        cos_sim_noisy = F.cosine_similarity(
            noisy_embeds.flatten(0, 1),
            orig_embeds.flatten(0, 1),
            dim=-1
        ).mean().item()

        outputs = model.generate(
            inputs_embeds=noisy_embeds,
            attention_mask=attention_mask,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

        noisy_gen = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        print(f"\nNoise level: {noise_level:.1f}")
        print(f"  Cosine sim: {cos_sim_noisy:.4f}")
        print(f"  Generated: {noisy_gen[:100]}...")

    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    print(f"Expected answer: {answer}")
    print(f"Baseline (original): {baseline_gen}")
    print(f"Untrained adapter: {untrained_gen}")
    print(f"\nCurrent training achieves ~0.17 cosine similarity")
    print(f"This experiment shows what quality is needed for generation.")
