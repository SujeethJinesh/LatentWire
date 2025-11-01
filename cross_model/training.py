"""Training functions for learned projections."""

import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

from .models import LearnedProjection


def train_learned_projection(
    model_a,
    model_b,
    tokenizer_a,
    tokenizer_b,
    dim_a: int,
    dim_b: int,
    layer_idx: int = 26,
    num_samples: int = 3072,
    learning_rate: float = 1e-3,
    num_epochs: int = 10,
    batch_size: int = 32,
    device: str = "cuda",
    seed: int = 42
):
    """
    Train projection matrix W to minimize MSE between model activations.

    Following Ramesh & Li (ICML 2025):
    - Uses 3072 sentences from C4 dataset
    - Minimizes MSE: ℒ = (1/N) Σ ||z^(i) - W y^(i)||²₂
    - y^(i): Model A's layer-26 final-token activation
    - z^(i): Model B's layer-26 final-token activation

    Args:
        model_a: Source model (provides y activations)
        model_b: Target model (provides z activations)
        tokenizer_a: Tokenizer for model A
        tokenizer_b: Tokenizer for model B
        dim_a: Hidden dimension of model A
        dim_b: Hidden dimension of model B
        layer_idx: Layer to extract activations from (default: 26)
        num_samples: Number of C4 sentences (default: 3072, per paper)
        learning_rate: Learning rate for Adam optimizer
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        device: Device to train on
        seed: Random seed

    Returns:
        Trained LearnedProjection module
    """
    print("\n" + "="*80)
    print("TRAINING LEARNED PROJECTION (Ramesh & Li 2025)")
    print("="*80)
    print(f"Loading {num_samples} sentences from C4 dataset...")

    # Load C4 dataset
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load C4 validation split (smaller, faster)
    c4_dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)

    # Sample sentences
    sentences = []
    for i, example in enumerate(c4_dataset):
        if i >= num_samples:
            break
        text = example["text"].strip()
        if len(text) > 50:  # Filter very short texts
            sentences.append(text[:512])  # Truncate long texts

    print(f"Loaded {len(sentences)} sentences from C4")

    # Set models to eval mode
    model_a.eval()
    model_b.eval()

    # Handle DDP/DataParallel wrapping
    base_model_a = model_a.module if isinstance(model_a, (nn.DataParallel, DDP)) else model_a
    base_model_b = model_b.module if isinstance(model_b, (nn.DataParallel, DDP)) else model_b

    # Extract activations for all sentences - BATCHED for GPU efficiency
    print(f"\nExtracting layer-{layer_idx} final-token activations...")
    print(f"Using batch size {batch_size} for activation extraction")
    activations_a = []
    activations_b = []

    # Process in batches for GPU efficiency
    num_batches = (len(sentences) + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Extracting activations"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(sentences))
            batch_sentences = sentences[batch_start:batch_end]

            # Tokenize batch for model A
            inputs_a = tokenizer_a(
                batch_sentences,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(device)

            # Get model A activations
            outputs_a = base_model_a(
                **inputs_a,
                output_hidden_states=True,
                use_cache=False
            )
            # Final token of layer layer_idx for each sequence
            h_a = outputs_a.hidden_states[layer_idx][:, -1, :]  # [B, dim_a]
            activations_a.append(h_a.cpu())

            # Tokenize batch for model B
            inputs_b = tokenizer_b(
                batch_sentences,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(device)

            # Get model B activations
            outputs_b = base_model_b(
                **inputs_b,
                output_hidden_states=True,
                use_cache=False
            )
            # Final token of layer layer_idx for each sequence
            h_b = outputs_b.hidden_states[layer_idx][:, -1, :]  # [B, dim_b]
            activations_b.append(h_b.cpu())

    # Stack activations into tensors
    activations_a = torch.cat(activations_a, dim=0)  # [N, dim_a]
    activations_b = torch.cat(activations_b, dim=0)  # [N, dim_b]

    print(f"Activations A shape: {activations_a.shape}")
    print(f"Activations B shape: {activations_b.shape}")

    # Detect dtype from activations (they may be bfloat16 from H100 models)
    activation_dtype = activations_a.dtype
    print(f"Activation dtype: {activation_dtype}")

    # Initialize projection matrix with matching dtype
    projection = LearnedProjection(dim_a, dim_b).to(device=device, dtype=activation_dtype)
    optimizer = optim.Adam(projection.parameters(), lr=learning_rate)

    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(activations_a, activations_b)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Training loop
    print(f"\nTraining projection matrix for {num_epochs} epochs...")
    projection.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for y_batch, z_batch in dataloader:
            y_batch = y_batch.to(device)  # [B, dim_a]
            z_batch = z_batch.to(device)  # [B, dim_b]

            # Forward pass: project y to z's space
            z_pred = projection(y_batch)  # [B, dim_b]

            # MSE loss
            loss = torch.nn.functional.mse_loss(z_pred, z_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}: MSE Loss = {avg_loss:.6f}")

    projection.eval()

    # Compute final MSE
    with torch.no_grad():
        all_y = activations_a.to(device)
        all_z = activations_b.to(device)
        z_pred = projection(all_y)
        final_mse = torch.nn.functional.mse_loss(z_pred, all_z).item()
        print(f"\nFinal MSE on all {len(sentences)} samples: {final_mse:.6f}")

    print("="*80)
    print("PROJECTION TRAINING COMPLETE")
    print("="*80 + "\n")

    return projection
