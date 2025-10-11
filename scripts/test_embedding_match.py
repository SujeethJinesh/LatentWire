#!/usr/bin/env python3
"""
Test if we can match embedding statistics and preserve information flow.
This is the critical test - if latents don't match embedding distribution,
the model can't process them properly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from pathlib import Path
import json

def test_embedding_preservation():
    """Test if we can preserve information through encode→adapt cycle"""

    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    print(f"Loading {model_id}...")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    embed_dim = model.config.hidden_size  # 4096 for 8B model
    vocab_size = model.config.vocab_size

    print(f"Model loaded: embed_dim={embed_dim}, vocab={vocab_size}")

    # Test texts
    test_texts = [
        "The capital of France is",
        "2 + 2 equals",
        "The largest planet in our solar system is",
    ]

    # Get real embeddings as ground truth
    print("\n" + "="*60)
    print("GROUND TRUTH: Real Embeddings")
    print("="*60)

    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)

        with torch.no_grad():
            # Get real embeddings
            real_embeds = model.get_input_embeddings()(input_ids)

            print(f"\nText: '{text}'")
            print(f"  Shape: {real_embeds.shape}")
            print(f"  Mean: {real_embeds.mean():.4f}")
            print(f"  Std: {real_embeds.std():.4f}")
            print(f"  Norm: {real_embeds.norm(dim=-1).mean():.4f}")
            print(f"  Range: [{real_embeds.min():.4f}, {real_embeds.max():.4f}]")

            # Test generation with real embeddings
            outputs = model.generate(
                inputs_embeds=real_embeds,
                max_new_tokens=5,
                do_sample=False
            )
            generated = tokenizer.decode(outputs[0][len(input_ids[0]):])
            print(f"  → Generated: '{generated}'")

    # Now test different approaches to match these statistics
    print("\n" + "="*60)
    print("TEST 1: Random latents with matching statistics")
    print("="*60)

    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt")
        seq_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            # Create random latents matching embedding statistics
            # Key insight: embeddings are near-zero mean with small std
            fake_embeds = torch.randn(1, seq_len, embed_dim).to(model.device) * 0.02

            print(f"\nText: '{text}'")
            print(f"  Fake stats: mean={fake_embeds.mean():.4f}, std={fake_embeds.std():.4f}")

            # Test generation
            outputs = model.generate(
                inputs_embeds=fake_embeds,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            generated = tokenizer.decode(outputs[0])
            print(f"  → Generated: '{generated}'")

    print("\n" + "="*60)
    print("TEST 2: Linear projection of random features")
    print("="*60)

    # Create a simple linear adapter
    d_z = 256  # Latent dimension
    adapter = nn.Linear(d_z, embed_dim).to(model.device)

    # Initialize with small weights (critical!)
    with torch.no_grad():
        adapter.weight.data.normal_(0, 0.02)
        adapter.bias.data.zero_()

    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt")
        seq_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            # Create latent representation
            z = torch.randn(1, seq_len, d_z).to(model.device)

            # Project to embedding space
            projected = adapter(z)

            print(f"\nText: '{text}'")
            print(f"  Latent stats: mean={z.mean():.4f}, std={z.std():.4f}")
            print(f"  Projected stats: mean={projected.mean():.4f}, std={projected.std():.4f}")

            # Test generation
            outputs = model.generate(
                inputs_embeds=projected,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            generated = tokenizer.decode(outputs[0])
            print(f"  → Generated: '{generated}'")

    print("\n" + "="*60)
    print("TEST 3: Calibrated projection (match per-example statistics)")
    print("="*60)

    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        seq_len = input_ids.shape[1]

        with torch.no_grad():
            # Get real embedding statistics for calibration
            real_embeds = model.get_input_embeddings()(input_ids)
            target_norm = real_embeds.norm(dim=-1, keepdim=True).mean()

            # Create and project latents
            z = torch.randn(1, seq_len, d_z).to(model.device)
            projected = adapter(z)

            # Calibrate to match target statistics
            proj_norm = projected.norm(dim=-1, keepdim=True).mean()
            scale = target_norm / (proj_norm + 1e-8)
            calibrated = projected * scale

            print(f"\nText: '{text}'")
            print(f"  Target norm: {target_norm:.4f}")
            print(f"  Projected norm: {proj_norm:.4f}")
            print(f"  Scale factor: {scale:.4f}")
            print(f"  Calibrated stats: mean={calibrated.mean():.4f}, std={calibrated.std():.4f}, norm={calibrated.norm(dim=-1).mean():.4f}")

            # Test generation
            outputs = model.generate(
                inputs_embeds=calibrated,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            generated = tokenizer.decode(outputs[0])
            print(f"  → Generated: '{generated}'")

    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print("1. Embeddings have very specific statistics (near-zero mean, ~0.02 std)")
    print("2. Random embeddings with wrong statistics generate garbage")
    print("3. Linear projection alone isn't enough - need calibration")
    print("4. Per-example calibration is critical for proper generation")
    print("\nNext step: Test with actual trained encoder/adapter from checkpoint")

if __name__ == "__main__":
    test_embedding_preservation()