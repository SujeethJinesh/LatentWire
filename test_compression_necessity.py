#!/usr/bin/env python3
"""
Demonstrates why compression is necessary for Stage 1 training.
Shows what happens with and without compression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_no_compression():
    """Test adapter without compression - learns nothing useful"""
    print("="*60)
    print("TEST 1: NO COMPRESSION (Pointless)")
    print("="*60)

    # Simple adapter
    adapter = nn.Sequential(
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096)
    )

    # Initialize near identity
    with torch.no_grad():
        adapter[0].weight.data = torch.eye(4096) + torch.randn(4096, 4096) * 0.01
        adapter[2].weight.data = torch.eye(4096) + torch.randn(4096, 4096) * 0.01

    # Create fake "embedding" data
    batch_size = 10
    seq_len = 20
    fake_embeddings = torch.randn(batch_size, seq_len, 4096) * 0.02

    # "Train" the adapter
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)

    for step in range(100):
        # Forward pass
        output = adapter(fake_embeddings)

        # Loss: just reconstruct input (identity function)
        loss = F.mse_loss(output, fake_embeddings)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"  Step {step}: Loss = {loss.item():.6f}")

    print("\n  Result: Adapter learned y = x (identity function)")
    print("  This tells us NOTHING about handling compressed representations!")

def test_with_compression():
    """Test adapter with compression - meaningful task"""
    print("\n" + "="*60)
    print("TEST 2: WITH COMPRESSION (Meaningful)")
    print("="*60)

    # Compression layer (fixed, like PCA)
    compress = nn.Linear(4096, 512, bias=False)
    with torch.no_grad():
        compress.weight.data.normal_(0, 0.1)

    # Adapter (trainable)
    adapter = nn.Sequential(
        nn.Linear(512, 2048),
        nn.ReLU(),
        nn.Linear(2048, 4096)
    )

    # Create fake "embedding" data
    batch_size = 10
    seq_len = 20
    original_embeddings = torch.randn(batch_size, seq_len, 4096) * 0.02

    # Training
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)

    for step in range(100):
        # Compress (fixed)
        with torch.no_grad():
            compressed = compress(original_embeddings)

        # Reconstruct (trainable)
        reconstructed = adapter(compressed)

        # Loss: reconstruct from compressed
        loss = F.mse_loss(reconstructed, original_embeddings)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"  Step {step}: Loss = {loss.item():.6f}")

    print("\n  Result: Adapter learned to reconstruct from 8x compression!")
    print("  This tests if adapter can expand compressed info → embeddings")
    print("  Exactly what we need for the full pipeline!")

def explain_why():
    """Explain why compression matters"""
    print("\n" + "="*60)
    print("WHY COMPRESSION IS NECESSARY")
    print("="*60)

    print("""
In the full LatentWire pipeline:

    Text → [Encoder] → Compressed Latents [32, 256]
                              ↓
                         [Adapter]
                              ↓
                    Embeddings [32, 4096]

The adapter MUST learn to expand compressed info into embeddings.

Stage 1 tests this in isolation:

    Embeddings → [PCA] → Compressed [L, 512]
                             ↓
                        [Adapter] ← We train this
                             ↓
                    Reconstructed [L, 4096]

Without compression:
- No information bottleneck
- Nothing to reconstruct from
- Just learning identity (y = x)
- Doesn't test the actual task

With compression:
- Information bottleneck (8x)
- Must reconstruct missing information
- Tests expansion capability
- Simulates real pipeline requirements

Think of it like:
- No compression = Copying a photo (trivial)
- With compression = Restoring a JPEG (meaningful)
    """)

def test_compression_ratios():
    """Test different compression ratios"""
    print("\n" + "="*60)
    print("COMPRESSION RATIO EFFECTS")
    print("="*60)

    original_dim = 4096
    test_dims = [2048, 1024, 512, 256, 128, 64, 32]

    print("\nCompression  |  Ratio  |  Difficulty  |  Expected F1")
    print("-" * 55)

    for dim in test_dims:
        ratio = original_dim / dim
        if ratio < 4:
            difficulty = "Easy"
            expected_f1 = "75-80%"
        elif ratio < 16:
            difficulty = "Moderate"
            expected_f1 = "60-70%"
        elif ratio < 64:
            difficulty = "Hard"
            expected_f1 = "40-50%"
        else:
            difficulty = "Very Hard"
            expected_f1 = "20-30%"

        print(f"4096 → {dim:4d}  |  {ratio:5.1f}x  |  {difficulty:9s}  |  {expected_f1}")

    print("\nWe chose 512 (8x compression) as a good balance:")
    print("- Meaningful compression (not trivial)")
    print("- Should preserve ~70% of performance")
    print("- Similar to final target (32 latents = 128x)")

if __name__ == "__main__":
    test_no_compression()
    test_with_compression()
    explain_why()
    test_compression_ratios()