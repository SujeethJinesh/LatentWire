#!/usr/bin/env python3
"""
Linear Probe Baseline for LatentWire Project

This implements a standard linear probe baseline that extracts hidden states from
a frozen LLM and trains a simple linear classifier on top. This is a CRITICAL
baseline for comparing against the Bridge method.

Key differences from Bridge:
- Bridge: Sender → Perceiver → Receiver (cross-model transfer)
- Linear Probe: Sender → Mean Pool → Linear Classifier (single model)

The linear probe tests whether the sender model's hidden states already contain
sufficient information for the task, without needing cross-model transfer.

Integration with run_unified_comparison.py:
- Follows same interface as UnifiedBridge and SoftPromptTuning
- Can be trained with train_linear_probe() and evaluated with eval_linear_probe()
- Supports multi-seed evaluation
- Works with batch_size=2-4 to avoid OOM

Usage:
    from telepathy.linear_probe_baseline import LinearProbeBaseline, train_linear_probe, eval_linear_probe

    # Create probe
    probe = LinearProbeBaseline(hidden_dim=4096, num_classes=2)

    # Train
    train_linear_probe(probe, sender, sender_tok, train_ds, dataset_name, device)

    # Evaluate
    results = eval_linear_probe(probe, sender, sender_tok, eval_ds, dataset_name, device)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import random
from typing import Dict, List, Any, Optional


class LinearProbeBaseline(nn.Module):
    """
    Linear probe baseline: Extract hidden states → Linear classifier.

    This is a standard baseline used in representation learning to test whether
    the frozen model's hidden states contain task-relevant information.

    Architecture:
        1. Extract hidden states from specific layer (default: layer 24)
        2. Mean pool across sequence dimension [B, seq_len, hidden_dim] → [B, hidden_dim]
        3. Linear classifier: [B, hidden_dim] → [B, num_classes]

    Args:
        hidden_dim: Dimension of hidden states (e.g., 4096 for Llama-8B)
        num_classes: Number of output classes (2 for binary, 4 for AG News, etc.)
        layer_idx: Which layer to extract from (0-indexed, 24 = layer 24 of 32)
        pooling: How to pool hidden states ("mean", "last_token", "first_token")
        dropout: Dropout probability for regularization (0.0 = no dropout)
        normalize: Whether to L2-normalize hidden states before classifier
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        layer_idx: int = 24,
        pooling: str = "mean",
        dropout: float = 0.1,
        normalize: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.layer_idx = layer_idx
        self.pooling = pooling
        self.normalize = normalize

        # Linear classifier with optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # Initialize weights (Xavier initialization for stability)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def extract_and_pool_hidden_states(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract hidden states from model and pool across sequence dimension.

        Args:
            model: HuggingFace model with output_hidden_states support
            input_ids: [B, seq_len] token IDs
            attention_mask: [B, seq_len] attention mask

        Returns:
            pooled: [B, hidden_dim] pooled hidden states
        """
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        # Extract from specific layer
        # outputs.hidden_states is tuple: (embeddings, layer1, ..., layerN)
        # layer_idx=0 is embeddings, layer_idx=1 is first transformer layer
        hidden = outputs.hidden_states[self.layer_idx]  # [B, seq_len, hidden_dim]

        # Pool across sequence dimension
        if self.pooling == "mean":
            # Mean pooling over non-padding tokens
            mask = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
            masked_hidden = hidden * mask
            pooled = masked_hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        elif self.pooling == "last_token":
            # Use last non-padding token
            seq_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
            batch_indices = torch.arange(hidden.size(0), device=hidden.device)
            pooled = hidden[batch_indices, seq_lengths]
        elif self.pooling == "first_token":
            # Use first token (CLS-style)
            pooled = hidden[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        return pooled  # [B, hidden_dim]

    def forward(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: Extract hidden states → Pool → Classify.

        Args:
            model: Frozen LLM to extract hidden states from
            input_ids: [B, seq_len] input token IDs
            attention_mask: [B, seq_len] attention mask

        Returns:
            logits: [B, num_classes] classification logits
        """
        # Extract and pool hidden states
        pooled = self.extract_and_pool_hidden_states(model, input_ids, attention_mask)

        # Optional L2 normalization
        if self.normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)

        # Apply dropout and classifier
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        return logits


# =============================================================================
# Training Function
# =============================================================================

def train_linear_probe(
    probe: LinearProbeBaseline,
    model,
    tokenizer,
    train_ds,
    dataset_name: str,
    device: str,
    dataset_config: Dict[str, Any],
    steps: int = 2000,
    batch_size: int = 4,  # Small batch size to avoid OOM
    lr: float = 1e-3,  # Higher LR for linear classifier
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
) -> Dict[str, Any]:
    """
    Train linear probe on classification task.

    Args:
        probe: LinearProbeBaseline module
        model: Frozen LLM to extract hidden states from
        tokenizer: Tokenizer for the model
        train_ds: Training dataset
        dataset_name: Name of dataset (for logging)
        device: Device to train on
        dataset_config: Dataset configuration dict with text_field, label_field, etc.
        steps: Number of training steps
        batch_size: Batch size (use small values like 2-4 to avoid OOM)
        lr: Learning rate (higher for linear probe)
        weight_decay: Weight decay for regularization
        grad_clip: Gradient clipping threshold

    Returns:
        Dictionary with training info (final_loss, final_acc)
    """
    print(f"  [train_linear_probe] Training on {dataset_name}")
    print(f"  [train_linear_probe] Steps: {steps}, Batch size: {batch_size}, LR: {lr}")
    print(f"  [train_linear_probe] Layer: {probe.layer_idx}, Pooling: {probe.pooling}")

    # Optimizer (only train probe, model is frozen)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    probe.train()
    model.eval()  # Keep model frozen

    # DataLoader with proper batching
    def collate_fn(batch):
        texts = [item[dataset_config["text_field"]] for item in batch]
        labels = [item[dataset_config["label_field"]] for item in batch]
        return texts, labels

    # CLASS-BALANCED SAMPLING for binary classification (same as Bridge)
    if dataset_config["num_classes"] == 2:
        from torch.utils.data import WeightedRandomSampler
        # Count labels
        label_counts = {}
        for item in train_ds:
            lbl = item[dataset_config["label_field"]]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        # Compute weights (inverse frequency)
        weights = [1.0 / label_counts[item[dataset_config["label_field"]]] for item in train_ds]
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
        dataloader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                               collate_fn=collate_fn, drop_last=True)
        print(f"  [train_linear_probe] Using class-balanced sampling for binary classification")
    else:
        dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               collate_fn=collate_fn, drop_last=True)

    data_iter = iter(dataloader)

    losses = []
    accuracies = []
    step = 0
    pbar = tqdm(total=steps, desc=f"Training Linear Probe on {dataset_name}")

    while step < steps:
        try:
            texts, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            texts, labels = next(data_iter)

        B = len(texts)

        # Tokenize inputs
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass through probe
        logits = probe(
            model=model,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )  # [B, num_classes]

        # Compute loss
        targets = torch.tensor(labels, device=device)
        loss = F.cross_entropy(logits, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(probe.parameters(), grad_clip)

        optimizer.step()

        # Track metrics
        losses.append(loss.item())
        preds = logits.argmax(dim=-1)
        acc = (preds == targets).float().mean().item()
        accuracies.append(acc)

        step += 1
        pbar.update(1)

        if step % 100 == 0:
            pbar.set_postfix({
                "loss": np.mean(losses[-100:]),
                "acc": np.mean(accuracies[-100:]) * 100,
            })

    pbar.close()

    return {
        "final_loss": np.mean(losses[-100:]),
        "final_acc": np.mean(accuracies[-100:]) * 100,
    }


# =============================================================================
# Evaluation Function
# =============================================================================

def eval_linear_probe(
    probe: LinearProbeBaseline,
    model,
    tokenizer,
    eval_ds,
    dataset_name: str,
    device: str,
    dataset_config: Dict[str, Any],
    batch_size: int = 4,  # Small batch size to avoid OOM
) -> Dict[str, Any]:
    """
    Evaluate linear probe on classification task.

    Args:
        probe: Trained LinearProbeBaseline module
        model: Frozen LLM to extract hidden states from
        tokenizer: Tokenizer for the model
        eval_ds: Evaluation dataset
        dataset_name: Name of dataset (for logging)
        device: Device to evaluate on
        dataset_config: Dataset configuration dict
        batch_size: Batch size for evaluation

    Returns:
        Dictionary with evaluation metrics (accuracy, correct, total)
    """
    probe.eval()
    model.eval()

    correct = 0
    total = 0

    # Process in batches to avoid OOM
    texts = [item[dataset_config["text_field"]] for item in eval_ds]
    labels = [item[dataset_config["label_field"]] for item in eval_ds]

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Eval Linear Probe on {dataset_name}"):
            batch_texts = texts[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]

            # Tokenize
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            logits = probe(
                model=model,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

            # Predictions
            preds = logits.argmax(dim=-1).cpu().numpy()
            batch_labels = np.array(batch_labels)

            correct += (preds == batch_labels).sum()
            total += len(batch_labels)

    return {
        "accuracy": 100.0 * correct / total,
        "correct": int(correct),
        "total": int(total),
    }


# =============================================================================
# Example Integration with run_unified_comparison.py
# =============================================================================

def example_integration():
    """
    Example showing how to integrate LinearProbeBaseline into run_unified_comparison.py.

    Add this to the main() function in run_unified_comparison.py:
    """
    example_code = '''
    # In main() function, after loading sender model:

    from telepathy.linear_probe_baseline import LinearProbeBaseline, train_linear_probe, eval_linear_probe

    # For each dataset:
    for dataset_name in args.datasets:
        config = DATASET_CONFIGS[dataset_name]

        # ... existing code for other baselines ...

        # LINEAR PROBE BASELINE
        print(f"\\n[8/8] Training LINEAR PROBE (sender only, seed={seed})...")
        linear_probe = LinearProbeBaseline(
            hidden_dim=sender_dim,  # e.g., 4096 for Llama-8B
            num_classes=config["num_classes"],
            layer_idx=24,  # Use layer 24 for sentiment/classification
            pooling="mean",
            dropout=0.1,
            normalize=True,
        ).to(device=device, dtype=torch.bfloat16)

        train_info = train_linear_probe(
            probe=linear_probe,
            model=sender,
            tokenizer=sender_tok,
            train_ds=train_ds,
            dataset_name=dataset_name,
            device=device,
            dataset_config=config,
            steps=args.train_steps,
            batch_size=4,  # Small batch size to avoid OOM
            lr=1e-3,
        )

        linear_probe_results = eval_linear_probe(
            probe=linear_probe,
            model=sender,
            tokenizer=sender_tok,
            eval_ds=eval_ds,
            dataset_name=dataset_name,
            device=device,
            dataset_config=config,
            batch_size=4,
        )

        linear_probe_results["train_info"] = train_info
        dataset_results["linear_probe"] = linear_probe_results
        seed_results_list["linear_probe"].append(linear_probe_results)
        print(f"  Linear probe accuracy: {linear_probe_results['accuracy']:.1f}%")

        # Save checkpoint
        torch.save(linear_probe.state_dict(), f"{args.output_dir}/linear_probe_{dataset_name}_seed{seed}.pt")
    '''
    return example_code


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*80)
    print("Example Integration Code:")
    print("="*80)
    print(example_integration())
    print("\n" + "="*80)
    print("To use this baseline in run_unified_comparison.py:")
    print("="*80)
    print("""
    1. Import the baseline:
       from telepathy.linear_probe_baseline import LinearProbeBaseline, train_linear_probe, eval_linear_probe

    2. Add to seed_results_list initialization:
       seed_results_list = {
           "bridge": [],
           "prompt_tuning": [],
           "linear_probe": [],  # <-- ADD THIS
           ...
       }

    3. Add training and evaluation in the main loop (see example_integration() above)

    4. Update aggregation and comparison table to include linear_probe results

    This will provide a critical baseline showing whether cross-model transfer (Bridge)
    provides value beyond simply using the sender model's hidden states directly.
    """)
