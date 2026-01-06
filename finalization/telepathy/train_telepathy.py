#!/usr/bin/env python
# telepathy/train_telepathy.py
"""
Latent Telepathy Training: DDP training loop for the Latent Bridge.

Trains a neural adapter (PerceiverResampler + StatisticalNormalizer) that enables
Llama 3.1 8B to inject its internal hidden states directly into Mistral 0.3 7B.

Training Objectives:
1. Language Modeling Loss: Target model predicts answer given soft tokens
2. Reconstruction Loss: Cosine similarity preserves direction of source thoughts

Key Design Decisions:
- Source and Target LLMs are FROZEN (only bridge is trained)
- Primer text fills target KV cache to prevent "Amnesia"
- Teacher-forcing on answer tokens
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

from latent_bridge import LatentBridge


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
        description="Train Latent Bridge for cross-model telepathy"
    )
    # Model configuration
    parser.add_argument(
        "--source_model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Source model ID"
    )
    parser.add_argument(
        "--target_model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Target model ID"
    )
    parser.add_argument(
        "--stats_path",
        type=str,
        default="stats.pt",
        help="Path to calibration statistics"
    )
    parser.add_argument(
        "--source_layer",
        type=int,
        default=20,
        help="Layer to extract hidden states from"
    )

    # Bridge architecture
    parser.add_argument(
        "--soft_tokens",
        type=int,
        default=64,
        help="Number of soft tokens (latent sequence length)"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Number of Perceiver layers"
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=8,
        help="Number of attention heads"
    )

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--recon_weight", type=float, default=1.0, help="Reconstruction loss weight")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-GPU batch size")
    parser.add_argument("--steps", type=int, default=3000, help="Training steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--warmup_steps", type=int, default=100, help="LR warmup steps")

    # I/O
    parser.add_argument("--save_path", type=str, default="telepathy_bridge.pt", help="Checkpoint save path")
    parser.add_argument("--save_every", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")

    return parser.parse_args()


def train_step(
    batch: dict,
    src_tok: AutoTokenizer,
    tgt_tok: AutoTokenizer,
    src_model: AutoModelForCausalLM,
    bridge: LatentBridge,
    tgt_model: AutoModelForCausalLM,
    device: torch.device,
    args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single training step.

    Returns:
        total_loss, lm_loss, recon_loss
    """
    # 1. Prepare Prompts
    src_texts = [f"Question: {q}\nAnswer:" for q in batch['question']]
    tgt_texts = [f"{a}{tgt_tok.eos_token}" for a in batch['answer']]

    # 2. Source Forward (Extract Thoughts)
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

    # 3. Bridge Forward (Telepathy: compress to soft tokens)
    soft_tokens = bridge(src_h, src_mask)

    # 4. Reconstruction Loss (Cosine Similarity)
    # Forces adapter to preserve the DIRECTION (meaning) of source thoughts
    # This prevents mode collapse where all inputs map to same output
    soft_mean = soft_tokens.mean(dim=1).float()
    src_mean = src_h.mean(dim=1).float()
    loss_recon = 1 - F.cosine_similarity(soft_mean, src_mean, dim=-1).mean()

    # 5. Target Forward (Mistral)
    # Primer text fills KV cache - prevents "Amnesia" where model has no context
    primer_text = "Analysis of received thought vector: "
    B = len(batch['question'])

    primer_enc = tgt_tok(
        [primer_text] * B,
        return_tensors="pt",
        add_special_tokens=True
    ).to(device)
    primer_embeds = tgt_model.get_input_embeddings()(primer_enc.input_ids)

    tgt_enc = tgt_tok(
        tgt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    answer_embeds = tgt_model.get_input_embeddings()(tgt_enc.input_ids)

    # Concatenate: [Primer] + [Soft Tokens] + [Answer]
    inputs_embeds = torch.cat([primer_embeds, soft_tokens, answer_embeds], dim=1)

    # Create labels: mask primer and soft tokens with -100
    K = soft_tokens.shape[1]  # num soft tokens
    P_len = primer_embeds.shape[1]  # primer length

    ignore_prefix = torch.full((B, P_len + K), -100, dtype=torch.long, device=device)
    answer_labels = tgt_enc.input_ids.clone()
    answer_labels[tgt_enc.attention_mask == 0] = -100  # Mask padding
    labels = torch.cat([ignore_prefix, answer_labels], dim=1)

    # Create attention mask
    soft_mask = torch.ones(B, K, dtype=torch.long, device=device)
    full_mask = torch.cat([primer_enc.attention_mask, soft_mask, tgt_enc.attention_mask], dim=1)

    # Forward pass through target model
    outputs = tgt_model(
        inputs_embeds=inputs_embeds,
        attention_mask=full_mask,
        labels=labels
    )
    loss_lm = outputs.loss

    # Total loss
    total_loss = loss_lm + (args.recon_weight * loss_recon)

    return total_loss, loss_lm, loss_recon


def main():
    start_time = time.time()
    is_distributed = setup_ddp()
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}")

    if local_rank == 0:
        print("=" * 70)
        print("Latent Telepathy Training")
        print("=" * 70)
        print(f"Started at: {datetime.now().isoformat()}")
        print(f"Source: {args.source_model}")
        print(f"Target: {args.target_model}")
        print(f"Soft tokens: {args.soft_tokens}")
        print(f"Perceiver depth: {args.depth}, heads: {args.heads}")
        print(f"Steps: {args.steps}, Batch size: {args.batch_size}")
        print(f"LR: {args.lr}, Recon weight: {args.recon_weight}")
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

    # Initialize Bridge (Trainable)
    if local_rank == 0:
        print(f"[3/5] Initializing Latent Bridge...")
    bridge = LatentBridge(
        args,
        src_model.config.hidden_size,
        tgt_model.config.hidden_size
    ).to(device)

    if args.bf16:
        bridge = bridge.bfloat16()
    bridge.train()

    # Count parameters
    num_params = sum(p.numel() for p in bridge.parameters() if p.requires_grad)
    if local_rank == 0:
        print(f"  Bridge parameters: {num_params:,}")

    if is_distributed:
        bridge = DDP(bridge, device_ids=[local_rank])

    # Optimizer with warmup
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

    running_loss = 0.0
    running_lm = 0.0
    running_recon = 0.0
    log_interval = 50

    for step in progress:
        # Get batch (cycle through dataset)
        try:
            batch = next(iter_dl)
        except StopIteration:
            iter_dl = iter(dl)
            batch = next(iter_dl)

        # Training step
        loss, lm_loss, recon_loss = train_step(
            batch, src_tok, tgt_tok, src_model, bridge, tgt_model, device, args
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        # Logging
        running_loss += loss.item()
        running_lm += lm_loss.item()
        running_recon += recon_loss.item()

        if local_rank == 0:
            progress.set_description(
                f"Loss: {loss.item():.3f} | LM: {lm_loss.item():.3f} | Recon: {recon_loss.item():.3f}"
            )

            if (step + 1) % log_interval == 0:
                avg_loss = running_loss / log_interval
                avg_lm = running_lm / log_interval
                avg_recon = running_recon / log_interval
                lr = scheduler.get_last_lr()[0]

                print(f"\n[Step {step+1}/{args.steps}] "
                      f"Loss: {avg_loss:.4f} | LM: {avg_lm:.4f} | Recon: {avg_recon:.4f} | LR: {lr:.2e}")

                running_loss = 0.0
                running_lm = 0.0
                running_recon = 0.0

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
        print("Training Complete!")
        print(f"  Final checkpoint: {args.save_path}")
        print(f"  Total time: {elapsed/60:.1f} minutes")
        print(f"  Steps/sec: {args.steps/elapsed:.2f}")
        print("=" * 70)

    if is_distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
