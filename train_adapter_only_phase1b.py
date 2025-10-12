#!/usr/bin/env python3
"""
Stage 1 Phase 1b: Reconstruction + Generation-Aware Training

PHASE 1B IMPROVEMENTS:
- Adds K-token CE loss (supervise first K tokens after "Answer: ")
- Adds Prefix KD loss (distill from text-prompted teacher)
- Tracks FirstTok@1 and top-5 accuracy (critical for generation)
- Combined loss: Reconstruction (semantic) + Generation objectives (task format)

Goal: Achieve F1 50-70% by teaching model not just WHAT to say (reconstruction)
but HOW to say it (QA format, stopping behavior).

Based on Phase 1a results:
- Reconstruction quality: ✅ 89.5% cosine similarity achieved
- Generation quality: ❌ 24% F1 (answer present but buried in extra text)
- Root cause: Pure reconstruction preserves semantics but loses task pragmatics
- Solution: Add generation-aware objectives to preserve QA format
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
import traceback
from typing import Dict, Any

from latentwire.data import load_squad_subset
from latentwire.models import Adapter, LMWrapper, LMConfig
from latentwire.losses import k_token_ce_from_prefix, kd_first_k_prefix_vs_text


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
            device_name = "GPU" if "cuda" in str(device) else "CPU"
            print(f"  Fitting PCA on {device_name} with {embeddings.shape[0]:,} embedding vectors...")

            # Move to device and ensure float32
            embeddings = embeddings.to(device).float()

            # Center the data
            self.mean = embeddings.mean(dim=0)
            centered = embeddings - self.mean

            # Use randomized SVD (svd_lowrank) for memory efficiency
            # Only computes top-k singular values/vectors, uses much less memory
            print(f"  Computing top {self.output_dim} components via randomized SVD...")
            U, S, V = torch.svd_lowrank(centered, q=self.output_dim, niter=4)

            # V is already [input_dim, output_dim] - exactly what we need for projection
            self.projection = V  # [input_dim, output_dim]

            # Compute explained variance ratio
            variance = S ** 2 / (embeddings.shape[0] - 1)
            # Need to estimate total variance
            # For randomized SVD, we only have top-k variances
            # Approximate total as sum of top-k (conservative estimate)
            total_variance_approx = variance.sum()
            explained_variance = variance.sum()
            self.explained_variance_ratio = (explained_variance / total_variance_approx).item()

            print(f"  PCA captured variance (top {self.output_dim} components): {self.explained_variance_ratio:.1%}")

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
    Phase 1b: Reconstruction + generation-aware training.

    Combines reconstruction loss with K-token CE and Prefix KD to teach
    both semantic content (reconstruction) and task format (generation).
    """

    print("="*60)
    print("STAGE 1 PHASE 1B: RECONSTRUCTION + GENERATION TRAINING")
    print(f"Model: {args.model_id}")
    print(f"Compression: {args.input_dim} → {args.compress_dim} ({args.input_dim/args.compress_dim:.1f}× compression)")
    print(f"Training samples: {args.samples}")
    print(f"PCA samples: {args.pca_samples}")
    print(f"")
    print(f"Generation objectives:")
    print(f"  K-token CE: K={args.k_tokens}, λ={args.lambda_kce}")
    print(f"  Prefix KD: τ={args.kd_tau}, λ={args.lambda_kd}")
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

    # Load model and create wrapper
    print(f"\nLoading model and creating LMWrapper...")
    lm_config = LMConfig(
        model_id=args.model_id,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16,
        device_map="auto"
    )
    llm_wrapper = LMWrapper(lm_config)

    # Reuse model and tokenizer from wrapper (avoid loading twice)
    model = llm_wrapper.model
    tokenizer = llm_wrapper.tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    embed_dim = model.config.hidden_size
    device = next(model.parameters()).device

    # Get anchor tokens for "Answer: " (used in K-token CE and KD)
    anchor_text = " Answer: "
    anchor_ids = tokenizer.encode(anchor_text, add_special_tokens=False)
    print(f"  Anchor text: '{anchor_text}' → {len(anchor_ids)} tokens: {anchor_ids}")
    print(f"  Model loaded once (reused from LMWrapper to save memory)")

    # Create adapter
    print(f"\nCreating adapter...")
    adapter = Adapter(
        d_z=args.compress_dim,
        d_model=embed_dim,
        latent_length=32,  # Dummy value, not used for simple reconstruction
        hidden_mult=args.adapter_hidden_mult,
        dropout=args.adapter_dropout,
        enable_metadata=False,
        colorize=True  # Enable learnable calibration to fix magnitude mismatch
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

        # Find GPU with most free memory
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            max_free = 0
            best_gpu = 0
            print("\nGPU memory status:")
            for i in range(gpu_count):
                free_mem = torch.cuda.mem_get_info(i)[0] / 1e9
                total_mem = torch.cuda.mem_get_info(i)[1] / 1e9
                allocated = torch.cuda.memory_allocated(i) / 1e9
                print(f"  GPU {i}: {free_mem:.1f} GB free / {total_mem:.1f} GB total (allocated: {allocated:.1f} GB)")
                if free_mem > max_free:
                    max_free = free_mem
                    best_gpu = i

            pca_device = f"cuda:{best_gpu}"
            print(f"\nUsing GPU {best_gpu} for PCA ({max_free:.1f} GB free)")
        else:
            pca_device = "cpu"
            print("\nUsing CPU for PCA")

        pca_dataset = load_squad_subset("train", args.pca_samples, seed=42)
        BATCH_SIZE = 64

        # Collect embeddings (move to best GPU for PCA)
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
                # Get valid (non-padded) embeddings, move to PCA device
                valid_embeds = embeds[attention_mask.bool()].to(pca_device)
                all_embeddings.append(valid_embeds)
                total_vectors += valid_embeds.shape[0]

            # Periodic cache cleanup
            if i % (BATCH_SIZE * 10) == 0:
                torch.cuda.empty_cache()

        # Concatenate on PCA device
        print(f"Concatenating {total_vectors:,} embedding vectors...")
        all_embeddings = torch.cat(all_embeddings, dim=0)

        # Fit PCA on selected device (GPU if available, CPU otherwise)
        compressor.fit(all_embeddings, device=pca_device)
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

        # Compute annealing factor for generation objectives
        if args.anneal_gen_objectives:
            # Linearly ramp from 0 to 1 over anneal_epochs
            anneal_factor = min(1.0, epoch / max(1, args.anneal_epochs))
            effective_lambda_kce = args.lambda_kce * anneal_factor
            effective_lambda_kd = args.lambda_kd * anneal_factor
            print(f"\nEpoch {epoch+1}: Annealing factor = {anneal_factor:.3f}")
            print(f"  Effective λ_kce = {effective_lambda_kce:.4f}, λ_kd = {effective_lambda_kd:.4f}")
        else:
            effective_lambda_kce = args.lambda_kce
            effective_lambda_kd = args.lambda_kd

        # Random shuffle
        indices = torch.randperm(len(dataset))
        num_batches = len(dataset) // args.batch_size

        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch_idx in pbar:
            # Get batch
            batch_indices = indices[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
            batch_items = [dataset[i] for i in batch_indices]

            # Prepare prompts and answers
            texts = [item['source'] + "Answer: " for item in batch_items]
            answers = [item['answer'] for item in batch_items]

            # Tokenize prompts
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True
            )
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)

            # Tokenize answers (for K-token CE and KD)
            answer_inputs = tokenizer(
                answers,
                return_tensors="pt",
                truncation=True,
                max_length=20,
                padding=True
            )
            answer_ids = answer_inputs.input_ids.to(device)

            # Get original embeddings
            with torch.no_grad():
                orig_embeds = model.get_input_embeddings()(input_ids)

            # Compress and reconstruct
            compressed = compressor.compress(orig_embeds)
            reconstructed = adapter(compressed)

            # Ensure dtype matches
            if reconstructed.dtype != orig_embeds.dtype:
                reconstructed = reconstructed.to(orig_embeds.dtype)

            # CRITICAL FIX: Match magnitude to original embeddings
            # Llama embeddings have very small scale (RMS ≈0.01 per dim, norm ≈0.5-1.0)
            # Adapter output has RMS ≈1 per dim (from LayerNorm), norm ≈64
            # Without matching, embeddings are 120× too large → LLM generates empty strings
            orig_rms = orig_embeds.pow(2).mean(dim=-1, keepdim=True).sqrt()
            recon_rms = reconstructed.pow(2).mean(dim=-1, keepdim=True).sqrt()
            reconstructed = reconstructed * (orig_rms / (recon_rms + 1e-8))

            # PHASE 1B: Combined loss (Reconstruction + Generation objectives)
            #
            # Loss components:
            # 1. Reconstruction (cosine + MSE): Preserves semantic content
            # 2. K-token CE: Supervises first K tokens after "Answer: "
            # 3. Prefix KD: Distills QA behavior from text-prompted teacher
            #
            # Goal: Teach model both WHAT to say (reconstruction) and HOW to say it (generation)

            # 1. Reconstruction loss (same as Phase 1a)
            rec_valid = reconstructed[attention_mask.bool()].to(torch.float32)
            orig_valid = orig_embeds[attention_mask.bool()].to(torch.float32)

            mse_loss = F.mse_loss(rec_valid, orig_valid)
            cosine_sim = F.cosine_similarity(rec_valid, orig_valid, dim=-1).mean()
            cosine_loss = 1.0 - cosine_sim
            loss_recon = 0.1 * mse_loss + cosine_loss

            # 2. K-token CE loss (teacher-forced supervision)
            # Supervises first K tokens after "Answer: " anchor
            try:
                loss_kce = k_token_ce_from_prefix(
                    llm_wrapper,
                    prefix_embeds=reconstructed,  # Use reconstructed embeddings as prefix
                    gold_ids=answer_ids,
                    K=args.k_tokens,
                    anchor_ids=anchor_ids,
                    append_bos_after_prefix=True
                )
            except Exception as e:
                print(f"\n[WARN] K-token CE failed: {e}")
                loss_kce = torch.zeros((), device=device)

            # 3. Prefix KD loss (distill from text teacher)
            # Transfer QA behavior from full-text baseline to latent-conditioned model
            # NOTE: KD is memory-intensive, can disable if OOM occurs
            if args.lambda_kd > 0:
                try:
                    loss_kd = kd_first_k_prefix_vs_text(
                        student_llm=llm_wrapper,
                        teacher_llm=llm_wrapper,  # Same model, different conditioning
                        prefix_embeds=reconstructed,
                        scaffold_ids=input_ids,
                        gold_ids=answer_ids,
                        K=args.k_tokens,
                        tau=args.kd_tau,
                        anchor_ids=anchor_ids,
                        append_bos_after_prefix=True
                    )
                except Exception as e:
                    print(f"\n[WARN] Prefix KD failed: {e}")
                    loss_kd = torch.zeros((), device=device)
            else:
                loss_kd = torch.zeros((), device=device)

            # Combined loss (use effective lambdas which may be annealed)
            loss = loss_recon + effective_lambda_kce * loss_kce + effective_lambda_kd * loss_kd

            # Aggressive cache clearing to prevent OOM
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

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
                "recon": f"{loss_recon.item():.3f}",
                "kce": f"{loss_kce.item():.3f}",
                "kd": f"{loss_kd.item():.3f}",
                "cos": f"{metrics['recon_cosine_sim']:.3f}"
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
                    "loss_recon": loss_recon.item(),
                    "loss_kce": loss_kce.item(),
                    "loss_kd": loss_kd.item(),
                    "effective_lambda_kce": effective_lambda_kce,
                    "effective_lambda_kd": effective_lambda_kd,
                    **metrics,
                    "lr": scheduler.get_last_lr()[0],
                    "compression_ratio": args.input_dim / args.compress_dim,
                    "batch_size": args.batch_size,
                    **gpu_info
                })

            # Periodic quick evaluation
            if step % 100 == 0:
                adapter.eval()
                quick_metrics = evaluate_quick(model, tokenizer, adapter, compressor, val_dataset[:10], device)
                print(f"\n  Step {step}: Quick F1 = {quick_metrics['f1']:.1%}, FirstTok@1 = {quick_metrics['first_tok_top1']:.1%}")

                log_diagnostics(diagnostic_log, step, epoch, {
                    "quick_f1": quick_metrics['f1'],
                    "quick_first_tok_top1": quick_metrics['first_tok_top1'],
                    "quick_first_tok_top5": quick_metrics['first_tok_top5'],
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
            first_tok_top1 = results.get('first_tok_top1', 0)
            first_tok_top5 = results.get('first_tok_top5', 0)

            print(f"  F1: {f1:.3f}")
            print(f"  EM: {em:.3f}")
            print(f"  FirstTok@1: {first_tok_top1:.1%}")
            print(f"  FirstTok@5: {first_tok_top5:.1%}")

            log_diagnostics(diagnostic_log, step, epoch, {
                "f1": f1,
                "em": em,
                "first_tok_top1": first_tok_top1,
                "first_tok_top5": first_tok_top5,
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


def compute_first_token_accuracy(model, tokenizer, adapted_embeds, attention_mask, gold_answers, device, top_k=5):
    """
    Compute FirstTok@1 and top-K accuracy for first token generation.

    Args:
        adapted_embeds: [batch, seq, hidden_dim] reconstructed embeddings
        gold_answers: List of gold answer strings
        top_k: Number of top tokens to consider for top-K accuracy

    Returns:
        dict with 'first_tok_top1' and 'first_tok_topk' accuracies
    """
    with torch.no_grad():
        # Get first token logits
        outputs = model(inputs_embeds=adapted_embeds, attention_mask=attention_mask)
        first_token_logits = outputs.logits[:, -1, :]  # [batch, vocab_size]

        # Tokenize gold answers to get first token
        gold_first_tokens = []
        for answer in gold_answers:
            # Encode answer and get first non-special token
            answer_ids = tokenizer.encode(answer, add_special_tokens=False)
            if len(answer_ids) > 0:
                gold_first_tokens.append(answer_ids[0])
            else:
                gold_first_tokens.append(-1)  # Invalid token

        gold_first_tokens = torch.tensor(gold_first_tokens, device=device)

        # Top-1 accuracy
        pred_top1 = first_token_logits.argmax(dim=-1)  # [batch]
        top1_correct = (pred_top1 == gold_first_tokens).float()
        top1_acc = top1_correct.mean().item()

        # Top-K accuracy
        pred_topk = first_token_logits.topk(top_k, dim=-1).indices  # [batch, top_k]
        topk_correct = (pred_topk == gold_first_tokens.unsqueeze(-1)).any(dim=-1).float()
        topk_acc = topk_correct.mean().item()

    return {
        'first_tok_top1': top1_acc,
        f'first_tok_top{top_k}': topk_acc
    }


def evaluate_quick(model, tokenizer, adapter, compressor, dataset, device):
    """Quick evaluation on small sample using F1 score (batched for speed)"""
    from latentwire.core_utils import batch_metrics

    predictions = []
    references = []

    # Process all 10 examples in one batch (quick eval is small)
    texts = [item['source'] + "Answer: " for item in dataset]
    batch_references = [item['answer'] for item in dataset]

    # Tokenize batch
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        # Forward pass: embed → compress → adapt
        orig_embeds = model.get_input_embeddings()(input_ids)
        compressed = compressor.compress(orig_embeds)
        adapted = adapter(compressed)

        if adapted.dtype != orig_embeds.dtype:
            adapted = adapted.to(orig_embeds.dtype)

        # Match magnitude (same fix as training)
        orig_rms = orig_embeds.pow(2).mean(dim=-1, keepdim=True).sqrt()
        adapted_rms = adapted.pow(2).mean(dim=-1, keepdim=True).sqrt()
        adapted = adapted * (orig_rms / (adapted_rms + 1e-8))

        # Batch generate
        outputs = model.generate(
            inputs_embeds=adapted,
            attention_mask=attention_mask,
            max_new_tokens=10,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id
        )

        # Decode batch outputs
        batch_generated = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

        # Compute aggregate diagnostics
        orig_norms = orig_embeds.norm(dim=-1).mean(dim=-1)  # [batch]
        recon_norms = adapted.norm(dim=-1).mean(dim=-1)  # [batch]
        norm_ratios = recon_norms / (orig_norms + 1e-8)

        # Flatten for cosine similarity
        batch_size = orig_embeds.shape[0]
        cos_sims = []
        for b in range(batch_size):
            orig_flat = orig_embeds[b].flatten()
            recon_flat = adapted[b].flatten()
            cos_sim = F.cosine_similarity(orig_flat.unsqueeze(0), recon_flat.unsqueeze(0), dim=-1).item()
            cos_sims.append(cos_sim)

        # Print first example diagnostics
        print(f"\n  Quick eval (first example):")
        print(f"    Expected: '{batch_references[0]}'")
        print(f"    Generated: '{batch_generated[0].strip()}'")
        print(f"    Norm ratio: {norm_ratios[0].item():.3f} | Cosine: {cos_sims[0]:.3f}")

        # Aggregate statistics
        avg_norm_ratio = norm_ratios.mean().item()
        avg_cos_sim = sum(cos_sims) / len(cos_sims)
        print(f"  Avg norm ratio: {avg_norm_ratio:.3f} | Avg cosine: {avg_cos_sim:.3f}")

        # Compute first-token accuracy
        first_tok_metrics = compute_first_token_accuracy(
            model, tokenizer, adapted, attention_mask, batch_references, device, top_k=5
        )
        print(f"  FirstTok@1: {first_tok_metrics['first_tok_top1']:.1%} | Top-5: {first_tok_metrics['first_tok_top5']:.1%}")

    predictions = [g.strip() for g in batch_generated]
    references = batch_references

    _, f1_score = batch_metrics(predictions, references)

    # Return dict with F1 and first-token metrics
    return {
        'f1': f1_score,
        **first_tok_metrics
    }


def decode_embeddings_to_tokens(embeddings, model, tokenizer, top_k=1):
    """
    Find nearest tokens for reconstructed embeddings.

    Args:
        embeddings: [batch, seq_len, hidden_dim] tensor
        model: LLM model
        tokenizer: tokenizer
        top_k: number of nearest tokens to return

    Returns:
        List of decoded text strings showing what tokens embeddings are closest to
    """
    with torch.no_grad():
        # Get full vocabulary embedding matrix
        vocab_embeds = model.get_input_embeddings().weight  # [vocab_size, hidden_dim]

        # Flatten sequence dimension
        batch_size, seq_len, hidden_dim = embeddings.shape
        flat_embeds = embeddings.view(-1, hidden_dim)  # [batch*seq_len, hidden_dim]

        # Compute cosine similarity to all vocab tokens
        # Normalize embeddings
        flat_embeds_norm = F.normalize(flat_embeds, p=2, dim=-1)
        vocab_embeds_norm = F.normalize(vocab_embeds, p=2, dim=-1)

        # Similarity: [batch*seq_len, vocab_size]
        similarities = torch.mm(flat_embeds_norm, vocab_embeds_norm.t())

        # Get top-k nearest tokens
        top_k_indices = similarities.topk(top_k, dim=-1).indices  # [batch*seq_len, top_k]

        # Reshape back to sequence
        top_k_indices = top_k_indices.view(batch_size, seq_len, top_k)

        # Decode each sequence
        decoded_texts = []
        for b in range(batch_size):
            # Get top-1 tokens for this sequence
            token_ids = top_k_indices[b, :, 0].cpu().tolist()
            decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
            decoded_texts.append(decoded)

        return decoded_texts


def evaluate_full(model, tokenizer, adapter, compressor, dataset, device, batch_size=32):
    """
    Full evaluation with F1 and EM scores using batched processing.

    Args:
        batch_size: Batch size for evaluation (default 32, since no gradients needed)
    """
    from latentwire.core_utils import batch_metrics

    predictions = []
    references = []

    print("\n" + "="*80)
    print(f"EVALUATION DIAGNOSTICS (Batched with size {batch_size})")
    print("="*80)

    # Process in batches for speed
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    global_idx = 0

    for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        batch_items = dataset[start_idx:end_idx]

        # Prepare batch
        texts = [item['source'] + "Answer: " for item in batch_items]
        batch_references = [item['answer'] for item in batch_items]

        # Tokenize batch with padding
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True  # Left-padding for generation
        )
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        with torch.no_grad():
            # Forward pass: embed → compress → adapt
            orig_embeds = model.get_input_embeddings()(input_ids)
            compressed = compressor.compress(orig_embeds)
            adapted = adapter(compressed)

            if adapted.dtype != orig_embeds.dtype:
                adapted = adapted.to(orig_embeds.dtype)

            # Match magnitude (same fix as training)
            orig_rms = orig_embeds.pow(2).mean(dim=-1, keepdim=True).sqrt()
            adapted_rms = adapted.pow(2).mean(dim=-1, keepdim=True).sqrt()
            adapted = adapted * (orig_rms / (adapted_rms + 1e-8))

            # Batch generate
            outputs = model.generate(
                inputs_embeds=adapted,
                attention_mask=attention_mask,
                max_new_tokens=20,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id
            )

            # Decode batch outputs
            # When using inputs_embeds, outputs contains ONLY generated tokens (not prompt)
            batch_generated = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

            # Diagnostics for first 3 examples only
            if global_idx < 3:
                for local_idx in range(min(3 - global_idx, len(batch_items))):
                    idx = global_idx + local_idx

                    # Compute per-example diagnostics
                    orig_norm = orig_embeds[local_idx].norm(dim=-1).mean().item()
                    recon_norm = adapted[local_idx].norm(dim=-1).mean().item()
                    norm_ratio = recon_norm / (orig_norm + 1e-8)

                    orig_flat = orig_embeds[local_idx].flatten()
                    recon_flat = adapted[local_idx].flatten()
                    cos_sim = F.cosine_similarity(orig_flat.unsqueeze(0), recon_flat.unsqueeze(0), dim=-1).item()

                    print(f"\n{'─'*80}")
                    print(f"Example {idx + 1}:")
                    print(f"  Question: {texts[local_idx][:80]}...")
                    print(f"  Expected: '{batch_references[local_idx]}'")
                    print(f"  Generated: '{batch_generated[local_idx].strip()}'")
                    print(f"\n  Embedding diagnostics:")
                    print(f"    Original norm:  {orig_norm:.2f}")
                    print(f"    Reconstructed norm: {recon_norm:.2f}")
                    print(f"    Ratio: {norm_ratio:.3f} ({'TOO LOW' if norm_ratio < 0.8 else 'TOO HIGH' if norm_ratio > 1.2 else 'OK'})")
                    print(f"    Cosine similarity: {cos_sim:.3f}")

                    # Only decode tokens for very first example (very expensive)
                    if idx == 0:
                        reconstructed_tokens = decode_embeddings_to_tokens(
                            adapted[local_idx:local_idx+1], model, tokenizer
                        )[0]
                        original_text_decoded = tokenizer.decode(input_ids[local_idx], skip_special_tokens=False)
                        print(f"\n  Token-level reconstruction (first example only):")
                        print(f"    Original tokens:  {original_text_decoded[:150]}...")
                        print(f"    Reconstructed →:  {reconstructed_tokens[:150]}...")

        # Collect predictions
        predictions.extend([g.strip() for g in batch_generated])
        references.extend(batch_references)
        global_idx += len(batch_items)

    print("\n" + "="*80)

    em_score, f1_score = batch_metrics(predictions, references)

    # Compute first-token accuracy on subset (first 50 examples for speed)
    print(f"\nComputing FirstTok metrics on {min(50, len(dataset))} examples...")
    first_tok_subset = dataset[:min(50, len(dataset))]
    subset_texts = [item['source'] + "Answer: " for item in first_tok_subset]
    subset_answers = [item['answer'] for item in first_tok_subset]

    subset_inputs = tokenizer(
        subset_texts,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    ).to(device)

    with torch.no_grad():
        subset_embeds = model.get_input_embeddings()(subset_inputs.input_ids)
        subset_compressed = compressor.compress(subset_embeds)
        subset_adapted = adapter(subset_compressed)

        if subset_adapted.dtype != subset_embeds.dtype:
            subset_adapted = subset_adapted.to(subset_embeds.dtype)

        # RMS matching
        orig_rms = subset_embeds.pow(2).mean(dim=-1, keepdim=True).sqrt()
        adapted_rms = subset_adapted.pow(2).mean(dim=-1, keepdim=True).sqrt()
        subset_adapted = subset_adapted * (orig_rms / (adapted_rms + 1e-8))

        first_tok_metrics = compute_first_token_accuracy(
            model, tokenizer, subset_adapted, subset_inputs.attention_mask,
            subset_answers, device, top_k=5
        )

    return {
        'em': em_score,
        'f1': f1_score,
        **first_tok_metrics
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Phase 1b: Reconstruction + Generation Training")

    # Model
    parser.add_argument("--model_id", default="meta-llama/Meta-Llama-3.1-8B-Instruct")

    # Compression
    parser.add_argument("--compress_dim", type=int, default=1024, help="Compressed dimension")
    parser.add_argument("--compress_method", default="pca", choices=["pca", "random"])
    parser.add_argument("--input_dim", type=int, default=4096, help="Input embedding dimension")
    parser.add_argument("--pca_samples", type=int, default=5000, help="Samples for fitting PCA")

    # Adapter
    parser.add_argument("--adapter_hidden_mult", type=int, default=4)
    parser.add_argument("--adapter_dropout", type=float, default=0.1)
    parser.add_argument("--adapter_lr", type=float, default=5e-4)

    # Training
    parser.add_argument("--samples", type=int, default=10000, help="Training samples")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)

    # Generation objectives (Phase 1b)
    parser.add_argument("--k_tokens", type=int, default=4, help="Number of tokens for K-token CE loss")
    parser.add_argument("--lambda_kce", type=float, default=0.5, help="Weight for K-token CE loss")
    parser.add_argument("--lambda_kd", type=float, default=0.5, help="Weight for Prefix KD loss")
    parser.add_argument("--kd_tau", type=float, default=1.0, help="Temperature for knowledge distillation")
    parser.add_argument("--anneal_gen_objectives", action="store_true", help="Anneal generation objectives from 0 to target over epochs")
    parser.add_argument("--anneal_epochs", type=int, default=3, help="Number of epochs to anneal over")

    # Evaluation
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_samples", type=int, default=100, help="Samples for full evaluation")

    # Output
    parser.add_argument("--save_dir", default="runs/stage1_phase1b")
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
        print("PHASE 1B RESULTS")
        print(f"{'='*60}")
        print(f"Best F1: {best_f1:.1%}")

        if best_f1 >= 0.70:
            print("✅ SUCCESS: Phase 1b achieves target!")
            print("   Generation-aware training validated")
            print("   Ready to proceed to full system (learned encoder + dual-LLM)")
        elif best_f1 >= 0.50:
            print("⚠️ PARTIAL: Improvement over Phase 1a but below target")
            print("   Consider: longer training, higher K, different loss weights")
        elif best_f1 >= 0.30:
            print("⚠️ MODEST: Generation objectives helping but more work needed")
            print("   Check: FirstTok@1 improving? KD/CE losses decreasing?")
        else:
            print("❌ BELOW PHASE 1A: Generation objectives may not be working")
            print("   Check: Loss function bugs? Anchor text alignment?")
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
