#!/usr/bin/env python3
"""
Simple demonstration of how the text pipeline works vs Stage 1 compression.
Run this to understand the difference!
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
import numpy as np

def demo_pipelines():
    print("="*60)
    print("DEMONSTRATING TEXT PIPELINE VS STAGE 1")
    print("="*60)

    # Load model (small one for demo)
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    print(f"\nLoading {model_id}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Test text
    text = "The capital of France is"
    print(f"\nInput text: '{text}'")

    # ============================================================
    # NORMAL PIPELINE (What works at 82% F1)
    # ============================================================
    print("\n" + "="*60)
    print("NORMAL TEXT PIPELINE")
    print("="*60)

    # Step 1: Tokenize
    tokens = tokenizer(text, return_tensors="pt")
    token_ids = tokens.input_ids
    print(f"\n1. Tokenization:")
    print(f"   Token IDs: {token_ids[0].tolist()}")
    print(f"   Shape: {token_ids.shape}")

    # Step 2: Get embeddings
    with torch.no_grad():
        embeddings = model.get_input_embeddings()(token_ids.to(model.device))

    print(f"\n2. Embeddings:")
    print(f"   Shape: {embeddings.shape} (seq_len={embeddings.shape[1]}, dim={embeddings.shape[2]})")
    print(f"   Stats: mean={embeddings.mean():.4f}, std={embeddings.std():.4f}")
    print(f"   Norm: {embeddings.norm(dim=-1).mean():.4f}")

    # Step 3: Generate
    with torch.no_grad():
        # Using inputs_embeds instead of input_ids
        outputs = model.generate(
            inputs_embeds=embeddings,
            max_new_tokens=3,
            do_sample=False
        )

    generated = tokenizer.decode(outputs[0][len(token_ids[0]):])
    print(f"\n3. Generated: '{generated}'")
    print(f"   ✅ This works at 82% F1!")

    # ============================================================
    # STAGE 1 PIPELINE (What we're testing)
    # ============================================================
    print("\n" + "="*60)
    print("STAGE 1: COMPRESSED PIPELINE")
    print("="*60)

    # Get embeddings again
    with torch.no_grad():
        original_embeddings = model.get_input_embeddings()(token_ids.to(model.device))

    embed_dim = original_embeddings.shape[-1]  # 4096
    compressed_dim = 512  # Much smaller!

    print(f"\n1. Original embeddings: {original_embeddings.shape}")

    # Step 2: Compress with PCA (simulation)
    print(f"\n2. Compressing {embed_dim} → {compressed_dim} dimensions")

    # Flatten for PCA
    flat_embeds = original_embeddings.reshape(-1, embed_dim).cpu().numpy()

    # Simulate PCA compression (in real training, we'd fit on more data)
    pca = PCA(n_components=compressed_dim)
    pca.fit(flat_embeds)  # In reality, fit on many examples

    # Compress
    compressed = pca.transform(flat_embeds)
    print(f"   Compressed shape: {compressed.shape}")
    print(f"   Compression ratio: {embed_dim/compressed_dim:.1f}x")

    # Step 3: Reconstruct (this is what the adapter learns)
    reconstructed = pca.inverse_transform(compressed)
    reconstructed_tensor = torch.tensor(
        reconstructed.reshape(original_embeddings.shape),
        dtype=original_embeddings.dtype,
        device=original_embeddings.device
    )

    print(f"\n3. Reconstructed embeddings: {reconstructed_tensor.shape}")

    # Compare statistics
    print(f"\n4. Comparing statistics:")
    print(f"   Original:      mean={original_embeddings.mean():.4f}, std={original_embeddings.std():.4f}")
    print(f"   Reconstructed: mean={reconstructed_tensor.mean():.4f}, std={reconstructed_tensor.std():.4f}")

    # Calculate reconstruction error
    mse = torch.nn.functional.mse_loss(reconstructed_tensor, original_embeddings)
    print(f"   Reconstruction MSE: {mse:.6f}")

    # Step 4: Generate with reconstructed embeddings
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=reconstructed_tensor,
            max_new_tokens=3,
            do_sample=False
        )

    generated = tokenizer.decode(outputs[0][len(token_ids[0]):])
    print(f"\n5. Generated with reconstructed: '{generated}'")

    # ============================================================
    # WHY STAGE 1 SHOULD WORK
    # ============================================================
    print("\n" + "="*60)
    print("WHY STAGE 1 SHOULD WORK")
    print("="*60)
    print("""
1. We start with embeddings that ALREADY WORK (82% F1)
2. PCA preserves the most important information
3. The adapter only learns: compressed → original
4. Much easier than learning: bytes → embeddings (full LatentWire)

The adapter is learning the INVERSE of PCA, which is a well-defined
linear transformation. This is MUCH easier than learning to create
embeddings from scratch!

Think of it like:
- Full LatentWire = Learning to paint from descriptions (hard!)
- Stage 1 = Learning to uncompress JPEG images (easier!)
    """)

if __name__ == "__main__":
    demo_pipelines()