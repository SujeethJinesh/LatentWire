#!/usr/bin/env python3
"""
Phase 1 + Sequence Pooling V2: Multiple compression strategies

Supports multiple pooling modes:
  - generation_loss: Train with generation loss directly (no reconstruction)
  - hierarchical: Multi-stage gradual compression
  - convolutional: Conv1D with strided downsampling
  - hybrid_expand: Pool, expand, reconstruct (test with compressed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from tqdm import tqdm
import argparse
import json
from pathlib import Path
import time
from typing import Dict, Any, Tuple

from latentwire.data import load_squad_subset
from latentwire.models import Adapter

import sys
sys.path.append('.')
from train_adapter_only_phase1 import (
    EmbeddingCompressor,
    log_diagnostics,
)


class SequencePooler(nn.Module):
    """Standard cross-attention pooling"""

    def __init__(self, M=75, d_model=1024, num_heads=8):
        super().__init__()
        self.M = M
        self.queries = nn.Parameter(torch.randn(M, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """[batch, seq, d] → [batch, M, d]"""
        q = self.queries.unsqueeze(0).expand(x.size(0), -1, -1)
        pooled, _ = self.cross_attn(q, x, x)
        return self.norm(pooled)


class HierarchicalPooler(nn.Module):
    """Multi-stage pooling: 300 → 225 → 150 → 75"""

    def __init__(self, M=75, d_model=1024):
        super().__init__()
        self.M = M

        # Calculate intermediate stages
        # For 300→75 (4×), use 3 stages of ~1.33× each
        self.stage1 = SequencePooler(M=225, d_model=d_model, num_heads=8)
        self.stage2 = SequencePooler(M=150, d_model=d_model, num_heads=8)
        self.stage3 = SequencePooler(M=75, d_model=d_model, num_heads=8)

        print(f"  HierarchicalPooler: 300 → 225 → 150 → {M}")

    def forward(self, x):
        """[batch, 300, d] → [batch, M, d]"""
        x = self.stage1(x)  # 300 → 225
        x = self.stage2(x)  # 225 → 150
        x = self.stage3(x)  # 150 → 75
        return x


class ConvolutionalPooler(nn.Module):
    """Strided 1D convolution for downsampling"""

    def __init__(self, M=75, d_model=1024):
        super().__init__()
        self.M = M

        # For 300→75, need 4× compression
        # Use stride=4 conv to downsample
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=4,
            stride=4,
            padding=0
        )
        self.norm = nn.LayerNorm(d_model)

        print(f"  ConvolutionalPooler: stride=4 conv, 300 → {M}")

    def forward(self, x):
        """[batch, seq, d] → [batch, M, d]"""
        # Conv1D expects [batch, d, seq]
        x = x.transpose(1, 2)  # [batch, d, seq]
        x = self.conv(x)        # [batch, d, M]
        x = x.transpose(1, 2)  # [batch, M, d]
        return self.norm(x)


def create_pooler(mode, M, d_model):
    """Factory for creating poolers"""
    if mode == "generation_loss":
        return SequencePooler(M=M, d_model=d_model)
    elif mode == "hierarchical":
        return HierarchicalPooler(M=M, d_model=d_model)
    elif mode == "convolutional":
        return ConvolutionalPooler(M=M, d_model=d_model)
    elif mode == "hybrid_expand":
        return SequencePooler(M=M, d_model=d_model)
    else:
        raise ValueError(f"Unknown pooling mode: {mode}")


def train_with_pooling_v2(args):
    """Training with various pooling strategies"""

    print("="*60)
    print(f"PHASE 1 + SEQUENCE POOLING ({args.pooling_mode.upper()})")
    print(f"Model: {args.model_id}")
    print(f"Compression: 300 → {args.sequence_pooling_target} tokens ({300/args.sequence_pooling_target:.1f}×)")
    print(f"Mode: {args.pooling_mode}")
    print("="*60)

    if not torch.cuda.is_available():
        print("\nERROR: No CUDA GPUs detected!")
        import sys
        sys.exit(1)

    diagnostic_log = args.diagnostic_log
    if diagnostic_log:
        Path(diagnostic_log).parent.mkdir(parents=True, exist_ok=True)

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

    # Create pooler
    print(f"\nCreating pooler ({args.pooling_mode})...")
    pooler = create_pooler(
        args.pooling_mode,
        M=args.sequence_pooling_target,
        d_model=args.compress_dim
    ).to(device).to(torch.bfloat16)

    # Create adapter
    print(f"\nCreating adapter...")
    adapter = Adapter(
        d_z=args.compress_dim,
        d_model=embed_dim,
        latent_length=args.sequence_pooling_target,
        hidden_mult=args.adapter_hidden_mult,
        dropout=args.adapter_dropout,
        enable_metadata=False,
        colorize=True
    ).to(device).to(torch.bfloat16)

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

    # Fit PCA
    if args.compress_method == "pca":
        print(f"\nFitting PCA...")
        gpu_count = torch.cuda.device_count()
        max_free = 0
        best_gpu = 0
        for i in range(gpu_count):
            free_mem = torch.cuda.mem_get_info(i)[0] / 1e9
            if free_mem > max_free:
                max_free = free_mem
                best_gpu = i
        pca_device = f"cuda:{best_gpu}"

        pca_dataset = load_squad_subset("train", args.pca_samples, seed=42)
        BATCH_SIZE = 64

        all_embeddings = []
        for i in tqdm(range(0, len(pca_dataset), BATCH_SIZE), desc="Collecting embeddings"):
            batch_items = pca_dataset[i:i+BATCH_SIZE]
            texts = [item['source'] + "Answer: " for item in batch_items]

            inputs = tokenizer(texts, return_tensors="pt", padding=True,
                             truncation=True, max_length=256)
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)

            with torch.no_grad():
                embeds = model.get_input_embeddings()(input_ids)
                valid_embeds = embeds[attention_mask.bool()].to(pca_device)
                all_embeddings.append(valid_embeds)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        compressor.fit(all_embeddings, device=pca_device)

    # Training setup
    trainable_params = list(pooler.parameters()) + list(adapter.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.adapter_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(dataset) // args.batch_size
    )

    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}\n")

    model.eval()  # Keep LLM frozen
    best_f1 = 0
    step = 0

    for epoch in range(args.epochs):
        pooler.train()
        adapter.train()

        epoch_loss = 0
        indices = torch.randperm(len(dataset))
        num_batches = len(dataset) // args.batch_size

        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch_idx in pbar:
            batch_indices = indices[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
            batch_items = [dataset[i] for i in batch_indices]

            # Prepare inputs
            texts = [item['source'] + "Answer: " for item in batch_items]
            answers = [item['answer'] for item in batch_items]

            inputs = tokenizer(texts, return_tensors="pt", truncation=True,
                             max_length=256, padding=True)
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)

            # Tokenize answers for generation loss
            answer_inputs = tokenizer(
                ["Answer: " + ans for ans in answers],
                return_tensors="pt",
                truncation=True,
                max_length=32,
                padding=True
            )
            answer_ids = answer_inputs.input_ids.to(device)
            answer_mask = answer_inputs.attention_mask.to(device)

            # Get embeddings
            with torch.no_grad():
                orig_embeds = model.get_input_embeddings()(input_ids)

            # Compress and pool
            compressed = compressor.compress(orig_embeds)  # [batch, seq, 1024]
            pooled = pooler(compressed)                     # [batch, M, 1024]

            # Choose loss based on mode
            if args.pooling_mode == "hybrid_expand":
                # Expand pooled back to sequence for reconstruction
                seq_len = orig_embeds.size(1)
                expand_factor = seq_len // args.sequence_pooling_target
                expanded = pooled.repeat_interleave(expand_factor, dim=1)  # [batch, seq, 1024]

                # Pad/trim to exact length
                if expanded.size(1) < seq_len:
                    pad = seq_len - expanded.size(1)
                    expanded = F.pad(expanded, (0, 0, 0, pad))
                elif expanded.size(1) > seq_len:
                    expanded = expanded[:, :seq_len, :]

                reconstructed = adapter(expanded)

                # Reconstruction loss
                if reconstructed.dtype != orig_embeds.dtype:
                    reconstructed = reconstructed.to(orig_embeds.dtype)

                # Magnitude matching
                orig_rms = orig_embeds.pow(2).mean(dim=-1, keepdim=True).sqrt()
                recon_rms = reconstructed.pow(2).mean(dim=-1, keepdim=True).sqrt()
                reconstructed = reconstructed * (orig_rms / (recon_rms + 1e-8))

                rec_valid = reconstructed[attention_mask.bool()].to(torch.float32)
                orig_valid = orig_embeds[attention_mask.bool()].to(torch.float32)

                mse_loss = F.mse_loss(rec_valid, orig_valid)
                cosine_sim = F.cosine_similarity(rec_valid, orig_valid, dim=-1).mean()
                cosine_loss = 1.0 - cosine_sim

                loss = 0.1 * mse_loss + cosine_loss

                metrics = {
                    "recon_mse": mse_loss.item(),
                    "recon_cosine_sim": cosine_sim.item()
                }

            else:  # generation_loss mode (and hierarchical, convolutional)
                # Adapt pooled embeddings
                adapted = adapter(pooled)  # [batch, M, 4096]

                if adapted.dtype != orig_embeds.dtype:
                    adapted = adapted.to(orig_embeds.dtype)

                # Create attention mask for pooled sequence
                pooled_attention_mask = torch.ones(
                    adapted.size(0), adapted.size(1),
                    dtype=attention_mask.dtype, device=device
                )

                # Prepare labels for generation loss
                # We'll use teacher forcing: feed adapted embeddings, predict answer
                # Combine: [adapted_embeds, answer_embeds]
                answer_embeds = model.get_input_embeddings()(answer_ids)

                # Concatenate
                combined_embeds = torch.cat([adapted, answer_embeds], dim=1)
                combined_mask = torch.cat([
                    pooled_attention_mask,
                    answer_mask
                ], dim=1)

                # Create labels: -100 for prefix (ignore), actual tokens for answer
                labels = torch.full_like(answer_ids, -100)
                labels[:, 1:] = answer_ids[:, 1:]  # Shift for next-token prediction
                labels[~answer_mask.bool()] = -100  # Mask padding

                # Prepend -100 for prefix tokens
                prefix_labels = torch.full(
                    (labels.size(0), adapted.size(1)),
                    -100,
                    dtype=labels.dtype,
                    device=device
                )
                full_labels = torch.cat([prefix_labels, labels], dim=1)

                # Forward pass through LLM
                outputs = model(
                    inputs_embeds=combined_embeds,
                    attention_mask=combined_mask,
                    labels=full_labels
                )

                loss = outputs.loss

                metrics = {
                    "gen_loss": loss.item()
                }

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Log diagnostics
            if step % 10 == 0:
                log_diagnostics(diagnostic_log, step, epoch, {
                    "loss": loss.item(),
                    **metrics,
                    "lr": scheduler.get_last_lr()[0],
                    "mode": args.pooling_mode
                })

        # Evaluation
        if (epoch + 1) % args.eval_every == 0:
            print(f"\nRunning evaluation...")
            model.eval()
            pooler.eval()
            adapter.eval()

            results = evaluate_with_pooling_v2(
                model, tokenizer, adapter, pooler, compressor,
                val_dataset[:args.eval_samples],
                device,
                args.pooling_mode,
                args.sequence_pooling_target
            )

            f1 = results['f1']
            em = results['em']

            print(f"  F1: {f1:.3f}")
            print(f"  EM: {em:.3f}")

            log_diagnostics(diagnostic_log, step, epoch, {
                "f1": f1,
                "em": em,
                "type": "full_eval"
            })

            # Save best
            if f1 > best_f1:
                best_f1 = f1
                save_dir = Path(args.save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)

                checkpoint = {
                    'pooler': pooler.state_dict(),
                    'adapter': adapter.state_dict(),
                    'compressor_projection': compressor.projection,
                    'compressor_mean': compressor.mean,
                    'config': vars(args),
                    'best_f1': best_f1
                }

                torch.save(checkpoint, save_dir / 'best_checkpoint.pt')
                print(f"  → New best F1! Saved checkpoint")

    print(f"\n{'='*60}")
    print(f"Training complete! Best F1: {best_f1:.3f}")
    print(f"{'='*60}")

    return best_f1


def evaluate_with_pooling_v2(model, tokenizer, adapter, pooler, compressor,
                             dataset, device, pooling_mode, M, batch_size=32):
    """Evaluate with pooling - always uses compressed at inference"""
    from latentwire.core_utils import batch_metrics

    predictions = []
    references = []

    num_batches = (len(dataset) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        batch_items = dataset[start_idx:end_idx]

        texts = [item['source'] + "Answer: " for item in batch_items]
        batch_references = [item['answer'] for item in batch_items]

        inputs = tokenizer(texts, return_tensors="pt", truncation=True,
                         max_length=256, padding=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        with torch.no_grad():
            # Forward: embed → compress → pool → adapt
            orig_embeds = model.get_input_embeddings()(input_ids)
            compressed = compressor.compress(orig_embeds)
            pooled = pooler(compressed)  # Always use pooled (compressed) at inference
            adapted = adapter(pooled)

            if adapted.dtype != orig_embeds.dtype:
                adapted = adapted.to(orig_embeds.dtype)

            # Attention mask for pooled sequence
            pooled_attention_mask = torch.ones(
                adapted.size(0), adapted.size(1),
                dtype=attention_mask.dtype, device=device
            )

            # Generate
            outputs = model.generate(
                inputs_embeds=adapted,
                attention_mask=pooled_attention_mask,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

            batch_generated = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

        predictions.extend([g.strip() for g in batch_generated])
        references.extend(batch_references)

    em_score, f1_score = batch_metrics(predictions, references)
    return {'em': em_score, 'f1': f1_score}


def main():
    parser = argparse.ArgumentParser(description="Phase 1 + Sequence Pooling V2")

    # Model
    parser.add_argument("--model_id", default="meta-llama/Meta-Llama-3.1-8B-Instruct")

    # Compression
    parser.add_argument("--compress_dim", type=int, default=1024)
    parser.add_argument("--compress_method", default="pca")
    parser.add_argument("--input_dim", type=int, default=4096)
    parser.add_argument("--pca_samples", type=int, default=5000)
    parser.add_argument("--sequence_pooling_target", type=int, default=75)

    # Pooling mode
    parser.add_argument("--pooling_mode", default="generation_loss",
                       choices=["generation_loss", "hierarchical", "convolutional", "hybrid_expand"],
                       help="Pooling strategy to use")

    # Adapter
    parser.add_argument("--adapter_hidden_mult", type=int, default=4)
    parser.add_argument("--adapter_dropout", type=float, default=0.1)
    parser.add_argument("--adapter_lr", type=float, default=5e-4)

    # Training
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)

    # Evaluation
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_samples", type=int, default=200)

    # Output
    parser.add_argument("--save_dir", default="runs/phase1_pooling_v2")
    parser.add_argument("--diagnostic_log", default=None)

    args = parser.parse_args()

    print("\nConfiguration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    best_f1 = train_with_pooling_v2(args)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Mode: {args.pooling_mode}")
    print(f"Best F1: {best_f1:.1%}")

    if best_f1 >= 0.30:
        print("✅ SUCCESS: Sequence compression works!")
    elif best_f1 >= 0.15:
        print("⚠️ MARGINAL: May work with improvements")
    else:
        print("❌ POOR: This approach doesn't work well")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
