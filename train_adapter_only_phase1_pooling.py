#!/usr/bin/env python3
"""
Phase 1 + Sequence Pooling: Testing if sequence compression works

ARCHITECTURE CHANGE from Phase 1a:
  Phase 1a: Text → Embed [300, 4096] → PCA [300, 1024] → Adapter [300, 4096]
  This:     Text → Embed [300, 4096] → PCA [300, 1024] → Pooler [M, 1024] → Adapter [M, 4096]
                                                          ^^^^^^^^ NEW LAYER

Tests: Can learned cross-attention pooling compress sequences (300→M tokens)
       while preserving task performance?
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
from typing import Dict, Any

from latentwire.data import load_squad_subset
from latentwire.models import Adapter

# Import Phase 1a components we'll reuse
import sys
sys.path.append('.')
from train_adapter_only_phase1 import (
    EmbeddingCompressor,
    log_diagnostics,
    compute_reconstruction_metrics,
    decode_embeddings_to_tokens
)


class SequencePooler(nn.Module):
    """
    Learned sequence compression via cross-attention pooling.

    Compresses [batch, seq_len, d_model] → [batch, M, d_model]
    where M << seq_len (e.g., 300 → 75 tokens, 4× compression)
    """

    def __init__(self, M=75, d_model=1024, num_heads=8):
        super().__init__()
        self.M = M
        self.d_model = d_model

        # Learned queries: what information to extract
        self.queries = nn.Parameter(torch.randn(M, d_model) * 0.02)

        # Cross-attention: queries attend to full sequence
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )

        self.norm = nn.LayerNorm(d_model)

        print(f"  SequencePooler: {d_model}d × {M} queries, {num_heads} heads")
        print(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            pooled: [batch, M, d_model]
        """
        batch_size = x.size(0)

        # Expand queries to batch
        q = self.queries.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, M, d_model]

        # Cross-attention: queries attend to full sequence
        pooled, _ = self.cross_attn(q, x, x)  # [batch, M, d_model]

        # Layer norm for stability
        pooled = self.norm(pooled)

        return pooled


def setup_lora(model, lora_r=8, lora_alpha=16, lora_layers=4):
    """
    Add LoRA to first N layers of the model.

    Args:
        model: LLM model
        lora_r: LoRA rank
        lora_alpha: LoRA scaling
        lora_layers: Number of layers to apply LoRA to (from bottom)

    Returns:
        model with LoRA applied
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType

        # Apply LoRA only to first N layers
        # Llama layer names: model.layers.{i}.self_attn.{q,k,v,o}_proj
        target_layers = list(range(lora_layers))

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],  # Apply to Q and V projections
            layers_to_transform=target_layers,
            lora_dropout=0.05,
            bias="none"
        )

        model = get_peft_model(model, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  LoRA trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

        return model

    except ImportError:
        print("ERROR: peft library not installed!")
        print("Install with: pip install peft")
        import sys
        sys.exit(1)


def train_with_pooling(args):
    """
    Phase 1 + Sequence Pooling training.
    """

    print("="*60)
    print("PHASE 1 + SEQUENCE POOLING")
    print(f"Model: {args.model_id}")
    print(f"Dimension compression: {args.input_dim} → {args.compress_dim}")
    print(f"Sequence compression: 300 → {args.sequence_pooling_target} tokens ({300/args.sequence_pooling_target:.1f}×)")
    if args.use_lora:
        print(f"LoRA: r={args.lora_r}, α={args.lora_alpha}, layers={args.lora_layers}")
    print("="*60)

    # GPU check
    if not torch.cuda.is_available():
        print("\nERROR: No CUDA GPUs detected!")
        import sys
        sys.exit(1)

    print(f"\nGPU count: {torch.cuda.device_count()}")

    # Setup diagnostics
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

    # Apply LoRA if requested
    if args.use_lora:
        print(f"\nApplying LoRA to first {args.lora_layers} layers...")
        model = setup_lora(model, args.lora_r, args.lora_alpha, args.lora_layers)

    embed_dim = model.config.hidden_size
    device = next(model.parameters()).device

    # Create pooler (NEW)
    print(f"\nCreating sequence pooler...")
    pooler = SequencePooler(
        M=args.sequence_pooling_target,
        d_model=args.compress_dim,  # Operates on PCA-compressed dimension
        num_heads=8
    ).to(device).to(torch.bfloat16)

    # Create adapter
    print(f"\nCreating adapter...")
    adapter = Adapter(
        d_z=args.compress_dim,
        d_model=embed_dim,
        latent_length=args.sequence_pooling_target,  # Now M tokens, not 300
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
        print(f"\nFitting PCA on {args.pca_samples} samples...")

        # Find best GPU for PCA
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
    # Train pooler + adapter (+ LoRA if enabled)
    trainable_params = list(pooler.parameters()) + list(adapter.parameters())
    if args.use_lora:
        trainable_params += [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params, lr=args.adapter_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(dataset) // args.batch_size
    )

    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}\n")

    best_f1 = 0
    step = 0

    for epoch in range(args.epochs):
        if args.use_lora:
            model.train()
        else:
            model.eval()  # Keep frozen if no LoRA
        pooler.train()
        adapter.train()

        epoch_loss = 0
        indices = torch.randperm(len(dataset))
        num_batches = len(dataset) // args.batch_size

        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch_idx in pbar:
            batch_indices = indices[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
            batch_items = [dataset[i] for i in batch_indices]

            texts = [item['source'] + "Answer: " for item in batch_items]

            inputs = tokenizer(texts, return_tensors="pt", truncation=True,
                             max_length=256, padding=True)
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)

            # Get embeddings
            with torch.no_grad():
                orig_embeds = model.get_input_embeddings()(input_ids)

            # Compress dimension: [batch, seq, 4096] → [batch, seq, 1024]
            compressed = compressor.compress(orig_embeds)

            # Compress sequence: [batch, seq, 1024] → [batch, M, 1024]  (NEW STEP)
            pooled = pooler(compressed)

            # Reconstruct: [batch, M, 1024] → [batch, M, 4096]
            reconstructed = adapter(pooled)

            if reconstructed.dtype != orig_embeds.dtype:
                reconstructed = reconstructed.to(orig_embeds.dtype)

            # Magnitude matching - use average of original embeddings as target
            # (since we can't match M tokens to seq_len tokens directly)
            with torch.no_grad():
                target_pooled = []
                for b in range(orig_embeds.size(0)):
                    valid_orig = orig_embeds[b][attention_mask[b].bool()]  # Get valid tokens
                    avg_emb = valid_orig.mean(dim=0)  # Average over sequence
                    target_pooled.append(avg_emb.unsqueeze(0).expand(args.sequence_pooling_target, -1))
                target_pooled = torch.stack(target_pooled, dim=0)  # [batch, M, 4096]

            # Match magnitude
            target_rms = target_pooled.pow(2).mean(dim=-1, keepdim=True).sqrt()
            recon_rms = reconstructed.pow(2).mean(dim=-1, keepdim=True).sqrt()
            reconstructed = reconstructed * (target_rms / (recon_rms + 1e-8))

            # Reconstruction loss
            rec_flat = reconstructed.reshape(-1, embed_dim).to(torch.float32)
            target_flat = target_pooled.reshape(-1, embed_dim).to(torch.float32)

            mse_loss = F.mse_loss(rec_flat, target_flat)
            cosine_sim = F.cosine_similarity(rec_flat, target_flat, dim=-1).mean()
            cosine_loss = 1.0 - cosine_sim

            loss = 0.1 * mse_loss + cosine_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            step += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "cos": f"{cosine_sim.item():.3f}"
            })

            # Log diagnostics
            if step % 10 == 0:
                log_diagnostics(diagnostic_log, step, epoch, {
                    "loss": loss.item(),
                    "recon_mse": mse_loss.item(),
                    "recon_cosine_sim": cosine_sim.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "sequence_compression_ratio": 300 / args.sequence_pooling_target
                })

        # Evaluation
        if (epoch + 1) % args.eval_every == 0:
            print(f"\nRunning evaluation...")
            model.eval()
            pooler.eval()
            adapter.eval()

            results = evaluate_with_pooling(
                model, tokenizer, adapter, pooler, compressor,
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

                if args.use_lora:
                    checkpoint['lora'] = model.state_dict()

                torch.save(checkpoint, save_dir / 'best_checkpoint.pt')
                print(f"  → New best F1! Saved checkpoint")

    print(f"\n{'='*60}")
    print(f"Training complete! Best F1: {best_f1:.3f}")
    print(f"{'='*60}")

    return best_f1


def evaluate_with_pooling(model, tokenizer, adapter, pooler, compressor,
                         dataset, device, batch_size=32):
    """Evaluate with sequence pooling"""
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
            pooled = pooler(compressed)  # NEW: sequence compression
            adapted = adapter(pooled)

            if adapted.dtype != orig_embeds.dtype:
                adapted = adapted.to(orig_embeds.dtype)

            # Create attention mask for pooled sequence
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
    parser = argparse.ArgumentParser(description="Phase 1 + Sequence Pooling")

    # Model
    parser.add_argument("--model_id", default="meta-llama/Meta-Llama-3.1-8B-Instruct")

    # Compression
    parser.add_argument("--compress_dim", type=int, default=1024)
    parser.add_argument("--compress_method", default="pca")
    parser.add_argument("--input_dim", type=int, default=4096)
    parser.add_argument("--pca_samples", type=int, default=5000)
    parser.add_argument("--sequence_pooling_target", type=int, default=75,
                       help="Target sequence length after pooling (e.g., 75 = 4× compression from 300)")

    # LoRA (optional)
    parser.add_argument("--use_lora", action="store_true", help="Add LoRA to LLM")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_layers", type=int, default=4)

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
    parser.add_argument("--save_dir", default="runs/phase1_pooling")
    parser.add_argument("--diagnostic_log", default=None)

    args = parser.parse_args()

    print("\nConfiguration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    best_f1 = train_with_pooling(args)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Best F1: {best_f1:.1%}")
    print(f"Sequence compression: 300 → {args.sequence_pooling_target} tokens ({300/args.sequence_pooling_target:.1f}×)")

    if best_f1 >= 0.40:
        print("✅ SUCCESS: Sequence compression works well!")
    elif best_f1 >= 0.30:
        print("⚠️ MODERATE: Compression viable but could be improved")
    elif best_f1 >= 0.20:
        print("⚠️ MARGINAL: Compression lossy, consider adjustments")
    else:
        print("❌ FAILURE: Compression too aggressive")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
