#!/usr/bin/env python
# telepathy/train_telepathy_v3.py
"""
Latent Telepathy Phase 3: Manifold Anchoring

Phase 2 failed due to "Semantic Drift" - contrastive learning pushed vectors
into dead zones (mathematically unique but semantically meaningless).

Phase 3 Fixes:
1. Learnable Normalizer: Fine-tune scale/shift during training
2. Output Clamping: Prevent 10^100 value explosion
3. Batch Anchor Loss: Pull soft tokens toward target answer embeddings

The anchor loss ensures soft tokens stay in the "valid language manifold"
of Mistral's embedding space.
"""
import argparse
import os
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from latent_bridge_v3 import LatentBridgeV3


def setup_ddp():
    """Initialize distributed training if environment variables are set."""
    if "RANK" in os.environ:
        torch.distributed.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True
    return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Latent Bridge V3 with Manifold Anchoring"
    )
    # Model configuration
    parser.add_argument("--source_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--target_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--stats_path", default="stats.pt")
    parser.add_argument("--source_layer", type=int, default=20)

    # Bridge architecture
    parser.add_argument("--soft_tokens", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Lower LR for stability")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=100)

    # V3 Loss weights
    parser.add_argument("--anchor_weight", type=float, default=1.0,
                        help="Weight for batch anchor loss (pulls toward answer embeddings)")
    parser.add_argument("--contrastive_weight", type=float, default=0.1,
                        help="Reduced contrastive weight (was 0.5 in V2)")
    parser.add_argument("--contrastive_temp", type=float, default=0.07)

    # I/O
    parser.add_argument("--save_path", default="bridge_v3.pt")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--bf16", action="store_true")

    return parser.parse_args()


def contrastive_loss_fn(soft_tokens: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """InfoNCE loss (same as V2, but with reduced weight)."""
    features = soft_tokens.mean(dim=1).float()
    features = F.normalize(features, dim=1)
    logits = torch.matmul(features, features.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


def batch_anchor_loss(
    soft_tokens: torch.Tensor,
    target_answer_embeds: torch.Tensor,
    target_mask: torch.Tensor
) -> torch.Tensor:
    """
    V3: Batch Anchor Loss (Scale-Invariant).

    Pulls soft tokens toward the target answer embeddings using COSINE SIMILARITY
    instead of MSE. This is critical because:
    - soft_tokens are clamped to ~±0.003 (tanh * output_scale)
    - answer_embeds have ~10x larger magnitude (~0.03)
    - MSE would be ~0.0000007 (no gradient signal)
    - Cosine similarity ignores scale, measures directional alignment

    Args:
        soft_tokens: [B, K, D] - Output from bridge (clamped small values)
        target_answer_embeds: [B, T, D] - Target model's answer embeddings
        target_mask: [B, T] - Mask for answer tokens

    Returns:
        Scalar loss (1 - cosine_similarity)
    """
    # Mean pool soft tokens: [B, K, D] -> [B, D]
    soft_mean = soft_tokens.mean(dim=1).float()

    # Mean pool answer embeddings (masked): [B, T, D] -> [B, D]
    mask = target_mask.unsqueeze(-1).float()
    answer_mean = (target_answer_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    answer_mean = answer_mean.float()

    # Normalize for cosine similarity
    soft_norm = F.normalize(soft_mean, dim=1)
    answer_norm = F.normalize(answer_mean, dim=1)

    # Cosine similarity loss: 1 - cos_sim (0 = perfect alignment, 2 = opposite)
    cos_sim = (soft_norm * answer_norm).sum(dim=1).mean()
    return 1.0 - cos_sim


def train_step(
    batch: dict,
    src_tok: AutoTokenizer,
    tgt_tok: AutoTokenizer,
    src_model: AutoModelForCausalLM,
    bridge: LatentBridgeV3,
    tgt_model: AutoModelForCausalLM,
    device: torch.device,
    args: argparse.Namespace
) -> tuple[torch.Tensor, dict]:
    """
    Single training step with LM + Anchor + Contrastive losses.

    Returns:
        total_loss, loss_dict
    """
    # 1. Prepare Prompts
    src_texts = [f"Question: {q}\nAnswer:" for q in batch['question']]
    tgt_texts = [f"{a}{tgt_tok.eos_token}" for a in batch['answer']]

    # 2. Source Forward (Extract Llama's thoughts)
    with torch.no_grad():
        src_enc = src_tok(
            src_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(device)
        src_out = src_model(**src_enc, output_hidden_states=True)
        src_h = src_out.hidden_states[args.source_layer]
        if args.bf16:
            src_h = src_h.bfloat16()
        src_mask = src_enc.attention_mask

    # 3. Bridge Forward (Compress to clamped soft tokens)
    soft_tokens = bridge(src_h, src_mask)

    # 4. Get Target Answer Embeddings (for anchor loss)
    tgt_enc = tgt_tok(
        tgt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        answer_embeds = tgt_model.get_input_embeddings()(tgt_enc.input_ids)

    # 5. BATCH ANCHOR LOSS (V3 key addition)
    # Pulls soft tokens toward valid embedding regions
    loss_anchor = batch_anchor_loss(soft_tokens, answer_embeds, tgt_enc.attention_mask)

    # 6. Contrastive Loss (reduced weight from V2)
    loss_contrastive = contrastive_loss_fn(soft_tokens, args.contrastive_temp)

    # 7. Target Forward (Teacher Forcing for LM loss)
    primer_text = "Answer: "  # Shorter, more direct primer
    B = len(batch['question'])

    primer_enc = tgt_tok(
        [primer_text] * B,
        return_tensors="pt",
        add_special_tokens=True
    ).to(device)
    primer_embeds = tgt_model.get_input_embeddings()(primer_enc.input_ids)

    # Concatenate: [Primer] + [Soft Tokens] + [Answer]
    inputs_embeds = torch.cat([primer_embeds, soft_tokens, answer_embeds], dim=1)

    # Create labels
    K = soft_tokens.shape[1]
    P_len = primer_embeds.shape[1]

    ignore_prefix = torch.full((B, P_len + K), -100, dtype=torch.long, device=device)
    answer_labels = tgt_enc.input_ids.clone()
    answer_labels[tgt_enc.attention_mask == 0] = -100
    labels = torch.cat([ignore_prefix, answer_labels], dim=1)

    # Attention mask
    soft_mask = torch.ones(B, K, dtype=torch.long, device=device)
    full_mask = torch.cat([primer_enc.attention_mask, soft_mask, tgt_enc.attention_mask], dim=1)

    # Forward pass
    outputs = tgt_model(
        inputs_embeds=inputs_embeds,
        attention_mask=full_mask,
        labels=labels
    )
    loss_lm = outputs.loss

    # Combined loss
    total_loss = (
        loss_lm +
        args.anchor_weight * loss_anchor +
        args.contrastive_weight * loss_contrastive
    )

    loss_dict = {
        "total": total_loss.item(),
        "lm": loss_lm.item(),
        "anchor": loss_anchor.item(),
        "contrastive": loss_contrastive.item(),
    }

    return total_loss, loss_dict


def main():
    start_time = time.time()
    is_distributed = setup_ddp()
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}")

    if local_rank == 0:
        print("=" * 70)
        print("Latent Telepathy Phase 3: Manifold Anchoring")
        print("=" * 70)
        print(f"Started at: {datetime.now().isoformat()}")
        print(f"Source: {args.source_model}")
        print(f"Target: {args.target_model}")
        print("")
        print("Phase 3 Fixes:")
        print("  1. Learnable Normalizer (unfrozen parameters)")
        print("  2. Output Clamping (tanh to prevent explosion)")
        print("  3. Batch Anchor Loss (pull toward answer embeddings)")
        print("")
        print(f"Loss Weights:")
        print(f"  LM:          1.0")
        print(f"  Anchor:      {args.anchor_weight}")
        print(f"  Contrastive: {args.contrastive_weight} (reduced from 0.5)")
        print("")
        print(f"Soft tokens: {args.soft_tokens}")
        print(f"Steps: {args.steps}, LR: {args.lr}")
        print(f"World size: {world_size}")
        print("=" * 70)

    # Load Source Model (Frozen)
    if local_rank == 0:
        print(f"\n[1/5] Loading Source Model...")
    src_model = AutoModelForCausalLM.from_pretrained(
        args.source_model,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank}
    ).eval()
    for p in src_model.parameters():
        p.requires_grad = False
    src_tok = AutoTokenizer.from_pretrained(args.source_model)
    src_tok.pad_token = src_tok.eos_token

    # Load Target Model (Frozen)
    if local_rank == 0:
        print(f"[2/5] Loading Target Model...")
    tgt_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank}
    ).eval()
    for p in tgt_model.parameters():
        p.requires_grad = False
    tgt_tok = AutoTokenizer.from_pretrained(args.target_model)
    tgt_tok.pad_token = tgt_tok.eos_token

    # Calculate target embedding RMS for initialization
    with torch.no_grad():
        tgt_embeds = tgt_model.get_input_embeddings().weight.float()
        target_rms = tgt_embeds.pow(2).mean(dim=1).sqrt().median().item()
    if local_rank == 0:
        print(f"  Target embedding RMS: {target_rms:.4f}")

    # Initialize V3 Bridge (Trainable)
    if local_rank == 0:
        print(f"[3/5] Initializing V3 Latent Bridge...")

    bridge = LatentBridgeV3(
        args,
        src_model.config.hidden_size,
        tgt_model.config.hidden_size,
        target_rms=target_rms
    ).to(device)

    if args.bf16:
        bridge = bridge.bfloat16()
    bridge.train()

    num_params = sum(p.numel() for p in bridge.parameters() if p.requires_grad)
    if local_rank == 0:
        print(f"  Bridge parameters: {num_params:,}")

    if is_distributed:
        bridge = DDP(bridge, device_ids=[local_rank])

    # Optimizer
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=0.01)

    def get_lr(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    # Data
    if local_rank == 0:
        print(f"[4/5] Loading dataset...")
    ds = load_dataset("gsm8k", "main", split="train")
    if is_distributed:
        ds = ds.shard(world_size, local_rank)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Training Loop
    if local_rank == 0:
        print(f"[5/5] Starting training...")
        print("-" * 70)

    progress = tqdm(range(args.steps), disable=(local_rank != 0), desc="Training")
    iter_dl = iter(dl)

    running = {"total": 0, "lm": 0, "anchor": 0, "contrastive": 0}
    log_interval = 50

    for step in progress:
        try:
            batch = next(iter_dl)
        except StopIteration:
            iter_dl = iter(dl)
            batch = next(iter_dl)

        loss, loss_dict = train_step(
            batch, src_tok, tgt_tok, src_model, bridge, tgt_model, device, args
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        for k in running:
            running[k] += loss_dict[k]

        if local_rank == 0:
            progress.set_description(
                f"L:{loss_dict['total']:.2f} LM:{loss_dict['lm']:.2f} "
                f"Anc:{loss_dict['anchor']:.3f} Ctr:{loss_dict['contrastive']:.2f}"
            )

            if (step + 1) % log_interval == 0:
                avg = {k: v / log_interval for k, v in running.items()}
                lr = scheduler.get_last_lr()[0]

                print(f"\n[Step {step+1}/{args.steps}] "
                      f"Total: {avg['total']:.3f} | LM: {avg['lm']:.3f} | "
                      f"Anchor: {avg['anchor']:.4f} | Ctr: {avg['contrastive']:.3f} | LR: {lr:.2e}")

                # Debug: Print soft token stats
                model = bridge.module if is_distributed else bridge
                if hasattr(model, 'resampler'):
                    print(f"  Output scale: {model.resampler.output_scale.item():.4f}")

                running = {k: 0 for k in running}

        # Periodic checkpoint
        if local_rank == 0 and (step + 1) % args.save_every == 0:
            ckpt_path = args.save_path.replace(".pt", f"_step{step+1}.pt")
            model_to_save = bridge.module if is_distributed else bridge
            torch.save(model_to_save.state_dict(), ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # Final save
    if local_rank == 0:
        model_to_save = bridge.module if is_distributed else bridge
        torch.save(model_to_save.state_dict(), args.save_path)

        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print("Phase 3 Training Complete!")
        print(f"  Final checkpoint: {args.save_path}")
        print(f"  Total time: {elapsed/60:.1f} minutes")
        print("=" * 70)

    if is_distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
