#!/usr/bin/env python
# telepathy/train_telepathy_v13.py
"""
Phase 13: High-Fidelity Cross-Attention Diffusion Training

THE KEY CHANGE: Reconstruct QUESTION embeddings, not Answer embeddings.

V12 (broken):
    tgt_texts = [f"{a}" for a in answers]  # "18"
    # Bridge learns to generate answer geometry
    # Mistral gets "18-like" noise

V13 (fixed):
    tgt_texts = [f"{q}" for q in questions]  # "Janet has 16 ducks..."
    # Bridge learns to translate Q -> Q
    # If Mistral "reads" the question, it will solve it naturally
"""
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from datasets import load_dataset
from tqdm import tqdm
import argparse

from latent_bridge_v13 import LatentBridgeV13


def setup_ddp():
    if "RANK" in os.environ:
        torch.distributed.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True
    return False


class EMA:
    """Exponential Moving Average for model weights."""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--target_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--source_layer", type=int, default=16)
    parser.add_argument("--soft_tokens", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--save_path", default="bridge_v13.pt")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def train_step(batch, src_tok, tgt_tok, src_model, bridge, tgt_model, device, args):
    """
    Phase 13 Training Step: Reconstruct QUESTION embeddings.

    Key difference from V12:
    - V12: target = answer embeddings ("18")
    - V13: target = question embeddings ("Janet has 16 ducks...")
    """
    questions = batch['question']

    # Source: Llama reads the question
    src_texts = [f"Question: {q}\nAnswer:" for q in questions]

    with torch.no_grad():
        src_enc = src_tok(
            src_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=1024
        ).to(device)
        src_out = src_model(**src_enc, output_hidden_states=True)
        src_h = src_out.hidden_states[args.source_layer]
        if args.bf16:
            src_h = src_h.bfloat16()
        src_mask = src_enc.attention_mask

    # THE KEY FIX: Target = QUESTION embeddings (not answer)
    # We reconstruct the question in Mistral's space
    tgt_q_texts = [f"{q}" for q in questions]

    with torch.no_grad():
        tgt_enc = tgt_tok(
            tgt_q_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=512
        ).to(device)
        tgt_embeds = tgt_model.get_input_embeddings()(tgt_enc.input_ids)
        if args.bf16:
            tgt_embeds = tgt_embeds.bfloat16()

    # Compute Rectified Flow loss
    bridge_module = bridge.module if hasattr(bridge, 'module') else bridge
    loss = bridge_module.forward_loss(src_h, src_mask, tgt_embeds)

    return loss


def main():
    setup_ddp()
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}")

    if local_rank == 0:
        print("=" * 70)
        print("Phase 13: High-Fidelity Cross-Attention Diffusion")
        print("=" * 70)
        print("")
        print("KEY CHANGES from V12:")
        print("  1. Full cross-attention to Llama sequence (no pooling)")
        print("  2. Target = QUESTION embeddings (not Answer)")
        print("")
        print("If bridge can reconstruct Q in Mistral's space,")
        print("Mistral will solve it naturally (7B params do reasoning).")
        print("=" * 70)

    # Load models
    src_model = AutoModelForCausalLM.from_pretrained(
        args.source_model, torch_dtype=torch.bfloat16, device_map={"": local_rank}
    ).eval()
    tgt_model = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=torch.bfloat16, device_map={"": local_rank}
    ).eval()

    src_tok = AutoTokenizer.from_pretrained(args.source_model)
    src_tok.pad_token = src_tok.eos_token
    tgt_tok = AutoTokenizer.from_pretrained(args.target_model)
    tgt_tok.pad_token = tgt_tok.eos_token

    # Initialize bridge
    bridge = LatentBridgeV13(
        args,
        src_dim=src_model.config.hidden_size,
        tgt_dim=tgt_model.config.hidden_size,
    )
    if args.bf16:
        bridge = bridge.bfloat16()
    bridge.train()
    bridge.to(device)

    if torch.distributed.is_initialized():
        bridge = DDP(bridge, device_ids=[local_rank])

    # Optimizer with cosine annealing
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.steps
    )

    # EMA for smoother outputs
    bridge_for_ema = bridge.module if torch.distributed.is_initialized() else bridge
    ema = EMA(bridge_for_ema, decay=args.ema_decay)

    # Load data
    ds = load_dataset("gsm8k", "main", split="train")
    if torch.distributed.is_initialized():
        ds = ds.shard(world_size, local_rank)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if local_rank == 0:
        print(f"\nTraining on {len(ds)} samples")
        print(f"Steps: {args.steps}, LR: {args.lr}, Batch: {args.batch_size}")
        print("Starting training loop...")

    progress = tqdm(range(args.steps), disable=(local_rank != 0),
                    desc="V13 DiT+CrossAttn", ncols=100)
    iter_dl = iter(dl)
    running_loss = 0.0

    for step in progress:
        try:
            batch = next(iter_dl)
        except StopIteration:
            iter_dl = iter(dl)
            batch = next(iter_dl)

        loss = train_step(batch, src_tok, tgt_tok, src_model, bridge,
                          tgt_model, device, args)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        ema.update()

        running_loss += loss.item()

        progress.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })

        # Periodic logging
        if local_rank == 0 and (step + 1) % 50 == 0:
            avg_loss = running_loss / 50
            current_lr = scheduler.get_last_lr()[0]
            print(f"\n[Step {step+1}/{args.steps}]")
            print(f"  Flow Loss: {avg_loss:.4f}")
            print(f"  LR: {current_lr:.2e}")
            running_loss = 0.0

        # Save checkpoints
        if local_rank == 0 and (step + 1) % args.save_every == 0:
            bridge_to_save = bridge.module if torch.distributed.is_initialized() else bridge

            # Save standard weights
            torch.save(bridge_to_save.state_dict(), args.save_path)

            # Save EMA weights
            ema.apply_shadow()
            ema_path = args.save_path.replace(".pt", "_ema.pt")
            torch.save(bridge_to_save.state_dict(), ema_path)
            ema.restore()

            print(f"  Checkpoints saved: {args.save_path} + {ema_path}")

    # Final save with EMA weights
    if local_rank == 0:
        bridge_to_save = bridge.module if torch.distributed.is_initialized() else bridge

        ema.apply_shadow()
        torch.save(bridge_to_save.state_dict(), args.save_path)
        ema_path = args.save_path.replace(".pt", "_ema.pt")
        torch.save(bridge_to_save.state_dict(), ema_path)

        print("\n" + "=" * 70)
        print("Phase 13 Training Complete!")
        print(f"Final checkpoint (EMA): {args.save_path}")
        print("=" * 70)
        print("\nNEXT: Run eval to check entity transfer rate")
        print("Success = >30% entity transfer")


if __name__ == "__main__":
    main()
