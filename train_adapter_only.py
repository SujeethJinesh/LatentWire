#!/usr/bin/env python3
"""
Stage 1: Adapter-only training with light compression.
Start from working embeddings, add light compression, train adapter to reconstruct.
This should maintain most of the 82% F1 baseline performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm
import argparse
import json
from pathlib import Path
import time
import traceback

from latentwire.data import load_squad_subset
from latentwire.models import Adapter


class EmbeddingCompressor:
    """Simple compression using PCA or random projection"""

    def __init__(self, input_dim=4096, output_dim=256, method="pca"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.method = method
        self.projection = None

    def fit(self, embeddings):
        """Fit compression on sample embeddings"""
        if self.method == "pca":
            # Use PCA for compression
            # Convert BFloat16 to Float32 for numpy compatibility
            embeddings_flat = embeddings.reshape(-1, self.input_dim).cpu().float().numpy()
            pca = PCA(n_components=self.output_dim)
            pca.fit(embeddings_flat[:10000])  # Fit on subset

            # Convert to torch
            self.projection = torch.tensor(pca.components_.T, dtype=torch.float32)
            self.mean = torch.tensor(pca.mean_, dtype=torch.float32)
        else:
            # Random projection
            self.projection = torch.randn(self.input_dim, self.output_dim) / np.sqrt(self.output_dim)
            self.mean = torch.zeros(self.input_dim)

    def compress(self, embeddings):
        """Compress embeddings"""
        device = embeddings.device
        dtype = embeddings.dtype  # Preserve input dtype (bfloat16)
        shape = embeddings.shape[:-1]

        # Flatten
        flat = embeddings.reshape(-1, self.input_dim)

        # Project
        if self.projection is not None:
            # Convert projection matrices to match dtype
            proj = self.projection.to(device).to(dtype)
            mean = self.mean.to(device).to(dtype)
            compressed = (flat - mean) @ proj
            return compressed.reshape(*shape, self.output_dim)

        return embeddings[..., :self.output_dim]  # Simple truncation fallback


def log_diagnostics(diagnostic_log, step, epoch, metrics):
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


def train_adapter_only(args):
    """Train only the adapter with pre-computed compressed embeddings"""

    print("="*60)
    print("ADAPTER-ONLY TRAINING")
    print(f"Model: {args.model_id}")
    print(f"Compression: {args.input_dim} → {args.compress_dim}")
    print(f"Samples: {args.samples}")
    print("="*60)

    # Require GPU for training
    if not torch.cuda.is_available():
        print("\n" + "="*60)
        print("ERROR: No CUDA GPUs detected!")
        print("="*60)
        print("This script requires GPU for training.")
        print("Please check:")
        print("  - CUDA is installed")
        print("  - GPUs are visible to PyTorch")
        print("  - Driver and runtime are compatible")
        print("="*60)
        import sys
        sys.exit(1)

    # Print GPU information
    print(f"\nGPU Information:")
    print(f"  CUDA available: Yes")
    print(f"  Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

    # Setup diagnostic logging
    diagnostic_log = args.diagnostic_log if hasattr(args, 'diagnostic_log') else None
    if diagnostic_log:
        Path(diagnostic_log).parent.mkdir(parents=True, exist_ok=True)
        print(f"Logging diagnostics to: {diagnostic_log}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    embed_dim = model.config.hidden_size
    # Get device from embedding layer when using device_map="auto"
    device = next(model.parameters()).device

    # Create adapter
    # Note: We use a dummy latent_length here as it's required but not used for our simple mapping
    adapter = Adapter(
        d_z=args.compress_dim,
        d_model=embed_dim,
        latent_length=32,  # Not used in our simple reconstruction task
        hidden_mult=args.adapter_hidden_mult,
        dropout=args.adapter_dropout,
        enable_metadata=False,
        colorize=False
    ).to(device).to(torch.bfloat16)  # Match model dtype

    print(f"\nAdapter params: {sum(p.numel() for p in adapter.parameters()):,}")

    # Create compressor
    compressor = EmbeddingCompressor(
        input_dim=embed_dim,
        output_dim=args.compress_dim,
        method=args.compress_method
    )

    # Load data
    dataset = load_squad_subset("train", args.samples)
    val_dataset = load_squad_subset("validation", 500)

    # Pre-compute embeddings for fitting compressor
    print("\nFitting compressor...")
    sample_embeds = []
    for item in tqdm(dataset[:100], desc="Collecting embeddings"):
        # The 'source' field already contains "Context: ... \nQuestion: ..."
        text = item['source'] + "Answer: "
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        input_ids = inputs.input_ids.to(device)

        with torch.no_grad():
            embeds = model.get_input_embeddings()(input_ids)
            # Flatten to get individual embedding vectors
            sample_embeds.append(embeds.squeeze(0))  # Remove batch dimension

    # Concatenate all embeddings along the sequence dimension (dim=0)
    # This gives us a tensor of shape [total_tokens, embed_dim]
    sample_embeds = torch.cat(sample_embeds, dim=0)
    compressor.fit(sample_embeds)
    print(f"Compressor fitted on {sample_embeds.shape[0]} embedding vectors")

    # Training setup
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.adapter_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(dataset) // args.batch_size
    )

    # Training loop
    print("\nTraining adapter...")
    best_f1 = 0
    step = 0

    for epoch in range(args.epochs):
        adapter.train()
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_ce_loss = 0

        # Create batches
        indices = torch.randperm(len(dataset))
        num_batches = len(dataset) // args.batch_size

        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch_idx in pbar:
            # Get batch
            batch_indices = indices[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
            batch_items = [dataset[i] for i in batch_indices]

            # Prepare texts
            texts = [item['source'] + "Answer: " for item in batch_items]
            answers = [item['answer'] for item in batch_items]

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

            # Tokenize answers for labels
            answer_inputs = tokenizer(
                answers,
                return_tensors="pt",
                truncation=True,
                max_length=20,
                padding=True
            )
            answer_ids = answer_inputs.input_ids.to(device)
            answer_attention_mask = answer_inputs.attention_mask.to(device)

            # Get original embeddings
            with torch.no_grad():
                orig_embeds = model.get_input_embeddings()(input_ids)

            # Compress (preserves bfloat16 dtype)
            compressed = compressor.compress(orig_embeds)

            # Adapt (ensure output matches model dtype)
            reconstructed = adapter(compressed)
            # Ensure adapter output matches model dtype
            if reconstructed.dtype != orig_embeds.dtype:
                reconstructed = reconstructed.to(orig_embeds.dtype)

            # Loss 1: Reconstruction
            # Convert to float32 for MSE loss (not supported for BFloat16 on CPU)
            # Keep computation on GPU by ensuring tensors stay on same device
            device = reconstructed.device
            recon_loss = F.mse_loss(
                reconstructed[attention_mask.bool()].to(torch.float32),
                orig_embeds[attention_mask.bool()].to(torch.float32)
            )

            # Loss 2: Generation quality
            # Concatenate prompt embeddings with answer tokens
            full_ids = torch.cat([input_ids, answer_ids], dim=1)
            full_mask = torch.cat([attention_mask, answer_attention_mask], dim=1)

            # Get answer embeddings
            with torch.no_grad():
                answer_embeds = model.get_input_embeddings()(answer_ids)
                # Ensure answer_embeds is on same device as reconstructed (critical for multi-GPU)
                if answer_embeds.device != reconstructed.device:
                    answer_embeds = answer_embeds.to(reconstructed.device)

            # Concatenate reconstructed prompt with answer
            full_embeds = torch.cat([reconstructed, answer_embeds], dim=1)

            # Forward through model
            outputs = model(
                inputs_embeds=full_embeds,
                attention_mask=full_mask,
                labels=full_ids
            )

            ce_loss = outputs.loss

            # Combined loss
            loss = args.recon_weight * recon_loss + args.ce_weight * ce_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Track
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_ce_loss += ce_loss.item()
            step += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{recon_loss.item():.4f}",
                "ce": f"{ce_loss.item():.4f}"
            })

            # Log diagnostics
            if step % 10 == 0:
                # Get GPU memory stats with detailed multi-GPU info
                gpu_info = {}
                if torch.cuda.is_available():
                    # Log info for all visible GPUs
                    gpu_count = torch.cuda.device_count()
                    total_mem_gb = 0
                    total_allocated_gb = 0

                    for i in range(gpu_count):
                        allocated = torch.cuda.memory_allocated(i) / 1e9
                        reserved = torch.cuda.memory_reserved(i) / 1e9
                        total_mem_gb += torch.cuda.get_device_properties(i).total_memory / 1e9
                        total_allocated_gb += allocated
                        gpu_info[f"gpu_{i}_allocated_gb"] = allocated
                        gpu_info[f"gpu_{i}_reserved_gb"] = reserved

                    gpu_info["total_gpu_memory_gb"] = total_allocated_gb
                    gpu_info["gpu_count"] = gpu_count
                    gpu_info["device_name"] = torch.cuda.get_device_name(0)
                else:
                    gpu_info["gpu_count"] = 0
                    gpu_info["total_gpu_memory_gb"] = 0

                log_diagnostics(diagnostic_log, step, epoch, {
                    "loss": loss.item(),
                    "recon_loss": recon_loss.item(),
                    "ce_loss": ce_loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "compression_ratio": args.input_dim / args.compress_dim,
                    "batch_size": args.batch_size,
                    **gpu_info
                })

            # Periodic evaluation
            if step % 100 == 0:
                adapter.eval()

                # Quick eval on 10 samples
                correct = 0
                for val_item in val_dataset[:10]:
                    text = val_item['source'] + "Answer: "
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
                    input_ids = inputs.input_ids.to(device)

                    with torch.no_grad():
                        # Original → compress → adapt pipeline
                        orig_embeds = model.get_input_embeddings()(input_ids)
                        compressed = compressor.compress(orig_embeds)
                        adapted = adapter(compressed)
                        # Ensure dtype matches for generation
                        if adapted.dtype != orig_embeds.dtype:
                            adapted = adapted.to(orig_embeds.dtype)

                        # Generate
                        outputs = model.generate(
                            inputs_embeds=adapted,
                            max_new_tokens=10,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id
                        )

                        generated = tokenizer.decode(outputs[0][len(input_ids[0]):])
                        if val_item['answer'].lower() in generated.lower():
                            correct += 1

                quick_acc = correct / 10
                print(f"\n  Step {step}: Quick accuracy = {quick_acc:.1%}")

                # Log quick eval
                log_diagnostics(diagnostic_log, step, epoch, {
                    "quick_eval_acc": quick_acc,
                    "type": "quick_eval"
                })

                adapter.train()

        # End of epoch evaluation
        print(f"\nEpoch {epoch+1} complete:")
        print(f"  Avg loss: {epoch_loss/num_batches:.4f}")
        print(f"  Avg recon: {epoch_recon_loss/num_batches:.4f}")
        print(f"  Avg CE: {epoch_ce_loss/num_batches:.4f}")

        # Full evaluation
        if (epoch + 1) % args.eval_every == 0:
            print("\nRunning full evaluation...")
            adapter.eval()

            # Evaluate with compressed embeddings
            results = evaluate_compressed_adapter(
                model, tokenizer, adapter, compressor,
                val_dataset[:args.eval_samples]
            )

            f1 = results['f1']
            em = results['em']

            print(f"  F1: {f1:.3f}")
            print(f"  EM: {em:.3f}")

            # Log full evaluation
            log_diagnostics(diagnostic_log, step, epoch, {
                "f1": f1,
                "em": em,
                "type": "full_eval",
                "eval_samples": args.eval_samples
            })

            if f1 > best_f1:
                best_f1 = f1
                # Save checkpoint
                save_dir = Path(args.save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)

                torch.save({
                    'adapter': adapter.state_dict(),
                    'compressor_projection': compressor.projection,
                    'compressor_mean': compressor.mean,
                    'config': vars(args),
                    'best_f1': best_f1,
                    'epoch': epoch + 1
                }, save_dir / 'adapter_only_best.pt')

                print(f"  → New best F1! Saved checkpoint")

    print("\n" + "="*60)
    print(f"Training complete! Best F1: {best_f1:.3f}")
    if diagnostic_log:
        print(f"Diagnostics saved to: {diagnostic_log}")
    save_dir = Path(args.save_dir)
    if save_dir.exists():
        print(f"Checkpoints saved to: {save_dir}")
    print("="*60)


def evaluate_compressed_adapter(model, tokenizer, adapter, compressor, dataset):
    """Evaluate adapter with compressed embeddings"""
    predictions = []
    references = []

    # Get device from model parameters when using device_map="auto"
    device = next(model.parameters()).device

    for item in tqdm(dataset, desc="Evaluating"):
        text = item['source'] + "Answer: "
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        input_ids = inputs.input_ids.to(device)

        with torch.no_grad():
            # Pipeline: embed → compress → adapt
            orig_embeds = model.get_input_embeddings()(input_ids)
            compressed = compressor.compress(orig_embeds)
            adapted = adapter(compressed)
            # Ensure dtype matches for generation
            if adapted.dtype != orig_embeds.dtype:
                adapted = adapted.to(orig_embeds.dtype)

            # Generate
            outputs = model.generate(
                inputs_embeds=adapted,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

            generated = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)

        predictions.append(generated.strip())
        references.append(item['answer'])

    # Calculate metrics
    from latentwire.core_utils import batch_metrics
    em_score, f1_score = batch_metrics(predictions, references)

    return {'em': em_score, 'f1': f1_score}


def main():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--model_id", default="meta-llama/Meta-Llama-3.1-8B-Instruct")

    # Compression
    parser.add_argument("--compress_dim", type=int, default=512, help="Compressed dimension")
    parser.add_argument("--compress_method", default="pca", choices=["pca", "random"])
    parser.add_argument("--input_dim", type=int, default=4096, help="Input embedding dimension")

    # Adapter
    parser.add_argument("--adapter_hidden_mult", type=int, default=4)
    parser.add_argument("--adapter_dropout", type=float, default=0.1)
    parser.add_argument("--adapter_lr", type=float, default=1e-3)

    # Training
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--ce_weight", type=float, default=1.0)

    # Evaluation
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_samples", type=int, default=500)

    # Output
    parser.add_argument("--save_dir", default="runs/adapter_only")
    parser.add_argument("--diagnostic_log", default=None, help="Path to diagnostic log file (JSONL)")

    args = parser.parse_args()

    # Print configuration
    print("\nConfiguration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    try:
        train_adapter_only(args)
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