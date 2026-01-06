#!/usr/bin/env python
# telepathy/train_telepathy_v2.py
"""
Latent Telepathy Phase 2: Contrastive Learning Fix

Phase 1 achieved 0% accuracy but 75% "partial success" - Mistral generated
math problems but not the CORRECT math problem. This is "Posterior Collapse":
Mistral learned to ignore the soft tokens and just guess based on the primer.

This phase adds InfoNCE Contrastive Loss to force the bridge to produce
UNIQUE latent vectors for each input. If Question A and Question B produce
similar soft tokens, the model is punished.

Key Changes from v1:
- InfoNCE contrastive loss (forces uniqueness)
- Increased soft tokens: 64 -> 128 (more capacity)
- Larger batch size: 4 -> 8 (more negatives for contrastive)
- Higher learning rate: 1e-4 -> 2e-4

Usage:
    torchrun --standalone --nproc_per_node=4 telepathy/train_telepathy_v2.py
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
        description="Train Latent Bridge v2 with Contrastive Learning"
    )
    # Model configuration
    parser.add_argument("--source_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--target_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--stats_path", default="stats.pt")
    parser.add_argument("--source_layer", type=int, default=20)

    # Bridge architecture (increased capacity)
    parser.add_argument("--soft_tokens", type=int, default=128, help="Increased from 64")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)

    # Training hyperparameters (adjusted for Phase 2)
    parser.add_argument("--lr", type=float, default=2e-4, help="Increased from 1e-4")
    parser.add_argument("--batch_size", type=int, default=8, help="Increased for more negatives")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=100)

    # Loss weights
    parser.add_argument("--contrastive_weight", type=float, default=0.5,
                        help="Weight for InfoNCE contrastive loss (prevents collapse)")
    parser.add_argument("--contrastive_temp", type=float, default=0.07,
                        help="Temperature for contrastive loss")

    # I/O
    parser.add_argument("--save_path", default="bridge_v2.pt")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--bf16", action="store_true")

    return parser.parse_args()


def contrastive_loss_fn(soft_tokens: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    InfoNCE Contrastive Loss.

    Forces the latent vector for Batch[i] to be similar to itself (positive)
    and DISSIMILAR to Batch[j] for all j != i (negatives).

    This prevents 'Posterior Collapse' where all inputs map to similar outputs
    and the target model learns to ignore the input.

    Args:
        soft_tokens: [B, K, D] - Soft tokens from bridge
        temperature: Scaling factor (lower = sharper distinctions)

    Returns:
        Scalar loss value
    """
    # Mean pool over sequence: [B, K, D] -> [B, D]
    features = soft_tokens.mean(dim=1).float()

    # L2 normalize for cosine similarity
    features = F.normalize(features, dim=1)

    # Compute similarity matrix [B, B]
    # logits[i, j] = similarity between sample i and sample j
    logits = torch.matmul(features, features.T) / temperature

    # Labels: diagonal elements are positives (sample i matches sample i)
    labels = torch.arange(logits.shape[0], device=logits.device)

    # Cross-entropy pushes diagonal high, off-diagonal low
    return F.cross_entropy(logits, labels)


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
    Single training step with LM + Contrastive losses.

    Returns:
        total_loss, lm_loss, contrastive_loss
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

    # 3. Bridge Forward (Compress to soft tokens)
    soft_tokens = bridge(src_h, src_mask)

    # 4. Contrastive Loss (THE FIX for Posterior Collapse)
    # Forces each input to produce a UNIQUE latent representation
    loss_contrastive = contrastive_loss_fn(soft_tokens, args.contrastive_temp)

    # 5. Target Forward (Teacher Forcing)
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
    total_loss = loss_lm + (args.contrastive_weight * loss_contrastive)

    return total_loss, loss_lm, loss_contrastive


def main():
    start_time = time.time()
    is_distributed = setup_ddp()
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}")

    if local_rank == 0:
        print("=" * 70)
        print("Latent Telepathy Phase 2: Contrastive Learning")
        print("=" * 70)
        print(f"Started at: {datetime.now().isoformat()}")
        print(f"Source: {args.source_model}")
        print(f"Target: {args.target_model}")
        print("")
        print("Phase 2 Changes:")
        print(f"  Soft tokens:       128 (was 64)")
        print(f"  Batch size:        {args.batch_size} (more negatives)")
        print(f"  Contrastive weight: {args.contrastive_weight}")
        print(f"  Contrastive temp:   {args.contrastive_temp}")
        print("")
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

    # Initialize Bridge (Trainable) - Note: using args which has soft_tokens=128
    if local_rank == 0:
        print(f"[3/5] Initializing Latent Bridge (128 soft tokens)...")

    bridge = LatentBridge(
        args,
        src_model.config.hidden_size,
        tgt_model.config.hidden_size
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

    running_loss = 0.0
    running_lm = 0.0
    running_ctr = 0.0
    log_interval = 50

    for step in progress:
        try:
            batch = next(iter_dl)
        except StopIteration:
            iter_dl = iter(dl)
            batch = next(iter_dl)

        loss, lm_loss, ctr_loss = train_step(
            batch, src_tok, tgt_tok, src_model, bridge, tgt_model, device, args
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        running_lm += lm_loss.item()
        running_ctr += ctr_loss.item()

        if local_rank == 0:
            progress.set_description(
                f"Loss: {loss.item():.3f} | LM: {lm_loss.item():.3f} | Ctr: {ctr_loss.item():.3f}"
            )

            if (step + 1) % log_interval == 0:
                avg_loss = running_loss / log_interval
                avg_lm = running_lm / log_interval
                avg_ctr = running_ctr / log_interval
                lr = scheduler.get_last_lr()[0]

                print(f"\n[Step {step+1}/{args.steps}] "
                      f"Loss: {avg_loss:.4f} | LM: {avg_lm:.4f} | Ctr: {avg_ctr:.4f} | LR: {lr:.2e}")

                running_loss = 0.0
                running_lm = 0.0
                running_ctr = 0.0

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
        print("Phase 2 Training Complete!")
        print(f"  Final checkpoint: {args.save_path}")
        print(f"  Total time: {elapsed/60:.1f} minutes")
        print("=" * 70)

    if is_distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
