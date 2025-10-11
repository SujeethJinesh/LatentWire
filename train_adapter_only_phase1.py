#!/usr/bin/env python3
"""
Stage 1 Phase 1: Pure Reconstruction Training

CORRECTED IMPLEMENTATION:
- Tests hypothesis: Good reconstruction → Good generation
- Training: Pure MSE reconstruction loss (no teacher forcing)
- Evaluation: Autoregressive generation (matches real use case)
- PCA fitted on full 80k training set for best generalization
- F1 score for all evaluations (no substring matching)

This validates whether compression+adapter can maintain QA performance
through reconstruction quality alone, without generation-aware training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
# sklearn PCA removed - using GPU-based torch implementation instead
import numpy as np
from tqdm import tqdm
import argparse
import json
from pathlib import Path
import time
import traceback
from typing import Dict, Any

from latentwire.data import load_squad_subset
from latentwire.models import Adapter


class EmbeddingCompressor:
    """PCA-based compression of embeddings"""

    def __init__(self, input_dim=4096, output_dim=512, method="pca"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.method = method
        self.projection = None
        self.mean = None
        self.explained_variance_ratio = None

    def fit(self, embeddings: torch.Tensor, device='cpu'):
        """
        Fit compression on embedding samples using PyTorch PCA.

        Args:
            embeddings: [N, input_dim] tensor of embedding vectors
            device: Device to run PCA on ('cuda' for GPU, 'cpu' for CPU)
        """
        if self.method == "pca":
            device_name = "GPU" if device == "cuda" else "CPU"
            print(f"  Fitting PCA on {device_name} with {embeddings.shape[0]:,} embedding vectors...")

            # Move to device and ensure float32
            embeddings = embeddings.to(device).float()

            # Center the data
            self.mean = embeddings.mean(dim=0)
            centered = embeddings - self.mean

            # Compute SVD (GPU accelerated if device='cuda')
            # U: [N, N], S: [min(N,D)], V: [D, D]
            # We want the top k components from V
            U, S, Vt = torch.linalg.svd(centered, full_matrices=False)

            # Take top output_dim components
            self.projection = Vt[:self.output_dim].T  # [input_dim, output_dim]

            # Compute explained variance ratio
            variance = S ** 2 / (embeddings.shape[0] - 1)
            total_variance = variance.sum()
            explained_variance = variance[:self.output_dim].sum()
            self.explained_variance_ratio = (explained_variance / total_variance).item()

            print(f"  PCA explained variance: {self.explained_variance_ratio:.1%}")

            # Keep on CPU for later use
            self.projection = self.projection.cpu()
            self.mean = self.mean.cpu()

        else:
            # Random projection fallback
            self.projection = torch.randn(self.input_dim, self.output_dim) / np.sqrt(self.output_dim)
            self.mean = torch.zeros(self.input_dim)

    def compress(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compress embeddings: [batch, seq, input_dim] -> [batch, seq, output_dim]"""
        device = embeddings.device
        dtype = embeddings.dtype
        shape = embeddings.shape[:-1]

        # Flatten sequence dimension
        flat = embeddings.reshape(-1, self.input_dim)

        # Project
        if self.projection is not None:
            proj = self.projection.to(device).to(dtype)
            mean = self.mean.to(device).to(dtype)
            compressed = (flat - mean) @ proj
            return compressed.reshape(*shape, self.output_dim)

        # Fallback: simple truncation
        return embeddings[..., :self.output_dim]


def log_diagnostics(diagnostic_log: str, step: int, epoch: int, metrics: Dict[str, Any]):
    """Log diagnostics in JSONL format"""
    if diagnostic_log:
        entry = {
            "step": step,
            "epoch": epoch,
            "timestamp": time.time(),
            **metrics
        }
        with open(diagnostic_log, 'a') as f:
            f.write(json.dumps(entry) + '\n')


def compute_reconstruction_metrics(reconstructed: torch.Tensor,
                                   original: torch.Tensor,
                                   mask: torch.Tensor) -> Dict[str, float]:
    """
    Compute detailed reconstruction quality metrics.

    Returns:
        - mse: Mean squared error
        - cosine_sim: Cosine similarity
        - relative_error: Relative L2 error
    """
    with torch.no_grad():
        # Select only valid (non-padded) positions
        rec_valid = reconstructed[mask.bool()]
        orig_valid = original[mask.bool()]

        # MSE
        mse = F.mse_loss(rec_valid.float(), orig_valid.float()).item()

        # Cosine similarity
        cos_sim = F.cosine_similarity(rec_valid, orig_valid, dim=-1).mean().item()

        # Relative error
        rel_error = (rec_valid - orig_valid).norm() / (orig_valid.norm() + 1e-8)
        rel_error = rel_error.item()

    return {
        "recon_mse": mse,
        "recon_cosine_sim": cos_sim,
        "recon_rel_error": rel_error
    }


def train_adapter_phase1(args):
    """
    Phase 1: Pure reconstruction training.

    Tests hypothesis: Good reconstruction → Good generation
    """

    print("="*60)
    print("STAGE 1 PHASE 1: PURE RECONSTRUCTION TRAINING")
    print(f"Model: {args.model_id}")
    print(f"Compression: {args.input_dim} → {args.compress_dim} ({args.input_dim/args.compress_dim:.1f}× compression)")
    print(f"Training samples: {args.samples}")
    print(f"PCA samples: {args.pca_samples}")
    print("="*60)

    # Require GPU
    if not torch.cuda.is_available():
        print("\nERROR: No CUDA GPUs detected!")
        print("This script requires GPU for training.")
        import sys
        sys.exit(1)

    # Print GPU info
    print(f"\nGPU Information:")
    print(f"  CUDA available: Yes")
    print(f"  GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

    # Setup diagnostics
    diagnostic_log = args.diagnostic_log
    if diagnostic_log:
        Path(diagnostic_log).parent.mkdir(parents=True, exist_ok=True)
        print(f"\nLogging diagnostics to: {diagnostic_log}")

    # Load model
    print(f"\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    embed_dim = model.config.hidden_size
    device = next(model.parameters()).device

    # Create adapter
    print(f"\nCreating adapter...")
    adapter = Adapter(
        d_z=args.compress_dim,
        d_model=embed_dim,
        latent_length=32,  # Dummy value, not used for simple reconstruction
        hidden_mult=args.adapter_hidden_mult,
        dropout=args.adapter_dropout,
        enable_metadata=False,
        colorize=False
    ).to(device).to(torch.bfloat16)

    print(f"Adapter parameters: {sum(p.numel() for p in adapter.parameters()):,}")

    # Create compressor
    compressor = EmbeddingCompressor(
        input_dim=embed_dim,
        output_dim=args.compress_dim,
        method=args.compress_method
    )

    # Load data
    print(f"\nLoading dataset...")
    dataset = load_squad_subset("train", args.samples)
    val_dataset = load_squad_subset("validation", 500)
    print(f"Training samples: {len(dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Fit compressor
    if args.compress_method == "pca":
        print(f"\nFitting PCA compressor on {args.pca_samples:,} samples...")
        print(f"Using CPU-based PyTorch PCA (avoids GPU OOM)...")

        pca_dataset = load_squad_subset("train", args.pca_samples, seed=42)
        BATCH_SIZE = 64

        # Collect embeddings (move to CPU immediately to avoid GPU OOM)
        all_embeddings = []
        total_vectors = 0

        print("Collecting embeddings...")
        for i in tqdm(range(0, len(pca_dataset), BATCH_SIZE), desc="Collecting embeddings"):
            batch_items = pca_dataset[i:i+BATCH_SIZE]
            texts = [item['source'] + "Answer: " for item in batch_items]

            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)

            with torch.no_grad():
                embeds = model.get_input_embeddings()(input_ids)
                # Get valid (non-padded) embeddings, move to CPU immediately
                valid_embeds = embeds[attention_mask.bool()].cpu()
                all_embeddings.append(valid_embeds)
                total_vectors += valid_embeds.shape[0]

            # Periodic cache cleanup
            if i % (BATCH_SIZE * 10) == 0:
                torch.cuda.empty_cache()

        # Concatenate on CPU (avoids GPU OOM)
        print(f"Concatenating {total_vectors:,} embedding vectors on CPU...")
        all_embeddings = torch.cat(all_embeddings, dim=0)

        # Fit PCA on CPU (fast enough with torch, avoids GPU OOM)
        compressor.fit(all_embeddings, device='cpu')
        print(f"✓ PCA fitted on {total_vectors:,} embedding vectors")

    elif args.compress_method == "random":
        # Random projection - instant initialization, no fitting needed
        print(f"\nInitializing random projection compressor (instant)...")
        # Random projection matrix already initialized in __init__
        print(f"✓ Random projection ready ({args.input_dim} → {args.compress_dim})")

    # Training setup
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.adapter_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(dataset) // args.batch_size
    )

    # Training loop
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}\n")

    best_f1 = 0
    step = 0

    for epoch in range(args.epochs):
        adapter.train()
        epoch_loss = 0
        epoch_metrics = {
            "mse": 0,
            "cosine_sim": 0,
            "rel_error": 0
        }

        # Random shuffle
        indices = torch.randperm(len(dataset))
        num_batches = len(dataset) // args.batch_size

        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch_idx in pbar:
            # Get batch
            batch_indices = indices[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
            batch_items = [dataset[i] for i in batch_indices]

            # Prepare prompts
            texts = [item['source'] + "Answer: " for item in batch_items]

            # Tokenize
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True
            )
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)

            # Get original embeddings
            with torch.no_grad():
                orig_embeds = model.get_input_embeddings()(input_ids)

            # Compress and reconstruct
            compressed = compressor.compress(orig_embeds)
            reconstructed = adapter(compressed)

            # Ensure dtype matches
            if reconstructed.dtype != orig_embeds.dtype:
                reconstructed = reconstructed.to(orig_embeds.dtype)

            # PHASE 1: Combined reconstruction loss (MSE + Cosine similarity)
            # MSE loss for magnitude
            rec_valid = reconstructed[attention_mask.bool()].to(torch.float32)
            orig_valid = orig_embeds[attention_mask.bool()].to(torch.float32)

            mse_loss = F.mse_loss(rec_valid, orig_valid)

            # Cosine similarity loss for direction (1 - cosine_sim as loss)
            cosine_sim = F.cosine_similarity(rec_valid, orig_valid, dim=-1).mean()
            cosine_loss = 1.0 - cosine_sim

            # Combined loss (weighted)
            loss = mse_loss + 0.1 * cosine_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Track metrics
            epoch_loss += loss.item()
            step += 1

            # Compute detailed reconstruction metrics
            metrics = compute_reconstruction_metrics(reconstructed, orig_embeds, attention_mask)
            epoch_metrics["mse"] += metrics["recon_mse"]
            epoch_metrics["cosine_sim"] += metrics["recon_cosine_sim"]
            epoch_metrics["rel_error"] += metrics["recon_rel_error"]

            # Update progress
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "cos_sim": f"{metrics['recon_cosine_sim']:.3f}"
            })

            # Log diagnostics
            if step % 10 == 0:
                gpu_info = {}
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    total_allocated = 0
                    for i in range(gpu_count):
                        allocated = torch.cuda.memory_allocated(i) / 1e9
                        gpu_info[f"gpu_{i}_allocated_gb"] = allocated
                        total_allocated += allocated
                    gpu_info["total_gpu_memory_gb"] = total_allocated
                    gpu_info["gpu_count"] = gpu_count

                log_diagnostics(diagnostic_log, step, epoch, {
                    "loss": loss.item(),
                    **metrics,
                    "lr": scheduler.get_last_lr()[0],
                    "compression_ratio": args.input_dim / args.compress_dim,
                    "batch_size": args.batch_size,
                    **gpu_info
                })

            # Periodic quick evaluation
            if step % 100 == 0:
                adapter.eval()
                quick_f1 = evaluate_quick(model, tokenizer, adapter, compressor, val_dataset[:10], device)
                print(f"\n  Step {step}: Quick F1 = {quick_f1:.1%}")

                log_diagnostics(diagnostic_log, step, epoch, {
                    "quick_f1": quick_f1,
                    "type": "quick_eval"
                })

                adapter.train()

        # End of epoch
        avg_loss = epoch_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}

        print(f"\nEpoch {epoch+1} complete:")
        print(f"  Avg loss: {avg_loss:.4f}")
        print(f"  Avg cosine similarity: {avg_metrics['cosine_sim']:.3f}")
        print(f"  Avg relative error: {avg_metrics['rel_error']:.3f}")

        # Full evaluation
        if (epoch + 1) % args.eval_every == 0:
            print(f"\nRunning full evaluation...")
            adapter.eval()

            results = evaluate_full(
                model, tokenizer, adapter, compressor,
                val_dataset[:args.eval_samples],
                device
            )

            f1 = results['f1']
            em = results['em']

            print(f"  F1: {f1:.3f}")
            print(f"  EM: {em:.3f}")

            log_diagnostics(diagnostic_log, step, epoch, {
                "f1": f1,
                "em": em,
                "type": "full_eval",
                "eval_samples": args.eval_samples
            })

            # Save best checkpoint
            if f1 > best_f1:
                best_f1 = f1
                save_dir = Path(args.save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)

                torch.save({
                    'adapter': adapter.state_dict(),
                    'compressor_projection': compressor.projection,
                    'compressor_mean': compressor.mean,
                    'pca_explained_variance': compressor.explained_variance_ratio,
                    'config': vars(args),
                    'best_f1': best_f1,
                    'epoch': epoch + 1
                }, save_dir / 'adapter_phase1_best.pt')

                print(f"  → New best F1! Saved checkpoint")

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best F1: {best_f1:.3f}")
    print(f"{'='*60}")

    return best_f1


def evaluate_quick(model, tokenizer, adapter, compressor, dataset, device):
    """Quick evaluation on small sample using F1 score"""
    from latentwire.core_utils import batch_metrics

    predictions = []
    references = []

    for item in dataset:
        text = item['source'] + "Answer: "
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        input_ids = inputs.input_ids.to(device)

        with torch.no_grad():
            orig_embeds = model.get_input_embeddings()(input_ids)
            compressed = compressor.compress(orig_embeds)
            adapted = adapter(compressed)

            if adapted.dtype != orig_embeds.dtype:
                adapted = adapted.to(orig_embeds.dtype)

            # Create attention mask (all 1s for embeddings we're passing)
            attention_mask = torch.ones(
                adapted.shape[0], adapted.shape[1],
                dtype=torch.long, device=adapted.device
            )

            outputs = model.generate(
                inputs_embeds=adapted,
                attention_mask=attention_mask,
                max_new_tokens=10,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id
            )

            generated = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)

        predictions.append(generated.strip())
        references.append(item['answer'])

    _, f1_score = batch_metrics(predictions, references)
    return f1_score


def evaluate_full(model, tokenizer, adapter, compressor, dataset, device):
    """Full evaluation with F1 and EM scores"""
    from latentwire.core_utils import batch_metrics

    predictions = []
    references = []

    for item in tqdm(dataset, desc="Evaluating"):
        text = item['source'] + "Answer: "
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        input_ids = inputs.input_ids.to(device)

        with torch.no_grad():
            orig_embeds = model.get_input_embeddings()(input_ids)
            compressed = compressor.compress(orig_embeds)
            adapted = adapter(compressed)

            if adapted.dtype != orig_embeds.dtype:
                adapted = adapted.to(orig_embeds.dtype)

            # Create attention mask (all 1s for embeddings we're passing)
            attention_mask = torch.ones(
                adapted.shape[0], adapted.shape[1],
                dtype=torch.long, device=adapted.device
            )

            outputs = model.generate(
                inputs_embeds=adapted,
                attention_mask=attention_mask,
                max_new_tokens=20,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id
            )

            generated = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)

        predictions.append(generated.strip())
        references.append(item['answer'])

    em_score, f1_score = batch_metrics(predictions, references)
    return {'em': em_score, 'f1': f1_score}


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Phase 1: Pure Reconstruction Training")

    # Model
    parser.add_argument("--model_id", default="meta-llama/Meta-Llama-3.1-8B-Instruct")

    # Compression
    parser.add_argument("--compress_dim", type=int, default=1024, help="Compressed dimension")
    parser.add_argument("--compress_method", default="pca", choices=["pca", "random"])
    parser.add_argument("--input_dim", type=int, default=4096, help="Input embedding dimension")
    parser.add_argument("--pca_samples", type=int, default=20000, help="Samples for fitting PCA (GPU-accelerated, fast)")

    # Adapter
    parser.add_argument("--adapter_hidden_mult", type=int, default=4)
    parser.add_argument("--adapter_dropout", type=float, default=0.1)
    parser.add_argument("--adapter_lr", type=float, default=5e-4)

    # Training
    parser.add_argument("--samples", type=int, default=10000, help="Training samples")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)

    # Evaluation
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_samples", type=int, default=500)

    # Output
    parser.add_argument("--save_dir", default="runs/stage1_phase1")
    parser.add_argument("--diagnostic_log", default=None, help="Path to diagnostic log (JSONL)")

    args = parser.parse_args()

    # Print configuration
    print("\nConfiguration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    try:
        best_f1 = train_adapter_phase1(args)

        # Print conclusion
        print(f"\n{'='*60}")
        print("PHASE 1 RESULTS")
        print(f"{'='*60}")
        print(f"Best F1: {best_f1:.1%}")

        if best_f1 >= 0.70:
            print("✅ SUCCESS: Reconstruction alone achieves target!")
            print("   Hypothesis validated: Good reconstruction → Good generation")
        elif best_f1 >= 0.50:
            print("⚠️ PARTIAL: Reconstruction helps but may need Phase 2")
            print("   Consider adding generation-aware training")
        else:
            print("❌ BELOW TARGET: Investigation needed")
            print("   Check: PCA quality, adapter capacity, or training setup")
        print(f"{'='*60}")

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Training failed!")
        print(f"{'='*60}")
        print(f"Error: {e}")
        traceback.print_exc()
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
