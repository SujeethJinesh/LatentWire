#!/usr/bin/env python
# telepathy/train_telepathy_v12.py
"""
Phase 12: Diffusion Bridge Training with Rectified Flow

THE FIX: Stop predicting (regression) → start generating (diffusion).

Rectified Flow Training:
    1. Sample timestep t ~ U[0, 1]
    2. Create noisy interpolation: x_t = t * target + (1-t) * noise
    3. Predict velocity: v_pred = model(x_t, t, source)
    4. Loss: MSE(v_pred, target - noise)

The velocity is CONSTANT in Rectified Flow (straight line from noise to target).
This makes training simpler than DDPM or score-based methods.

Key insight: We're NOT trying to match Mistral embeddings exactly.
We're learning to generate vectors that LIE ON the Mistral manifold.
"""
import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse

from latent_bridge_v12 import LatentBridgeV12


def setup_ddp():
    if "RANK" in os.environ:
        torch.distributed.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True
    return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--target_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--source_layer", type=int, default=16)
    parser.add_argument("--soft_tokens", type=int, default=128)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--save_path", default="bridge_v12.pt")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def get_target_embeddings(questions, tgt_tok, tgt_model, device, num_latents):
    """
    Get Mistral embeddings for the questions.
    These are the TARGET vectors we want to learn to generate.
    """
    # Format questions for Mistral
    texts = [f"Question: {q}\nAnswer:" for q in questions]

    with torch.no_grad():
        enc = tgt_tok(texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=512).to(device)
        # Get embeddings (not hidden states - we want the raw embedding layer)
        embeds = tgt_model.get_input_embeddings()(enc.input_ids)

        # Pool/compress to fixed number of latents
        # Use simple mean pooling across sequence
        mask = enc.attention_mask.unsqueeze(-1).float()
        # Reshape to target number of tokens
        B, S, D = embeds.shape
        if S >= num_latents:
            # Downsample: take evenly spaced positions
            indices = torch.linspace(0, S-1, num_latents).long().to(device)
            target = embeds[:, indices, :]
        else:
            # Upsample: repeat and interpolate
            target = torch.nn.functional.interpolate(
                embeds.permute(0, 2, 1),
                size=num_latents,
                mode='linear',
                align_corners=True
            ).permute(0, 2, 1)

    return target


def train_step(batch, src_tok, tgt_tok, src_model, bridge, tgt_model, device, args):
    """
    Rectified Flow training step.

    1. Get source hidden states (Llama reads question)
    2. Get target embeddings (Mistral embedding layer)
    3. Sample timestep and create noisy interpolation
    4. Predict velocity and compute MSE loss
    """
    questions = batch['question']

    # 1. Source: Llama reads the question
    src_texts = [f"Question: {q}\nAnswer:" for q in questions]

    with torch.no_grad():
        src_enc = src_tok(src_texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=1024).to(device)
        src_out = src_model(**src_enc, output_hidden_states=True)
        src_h = src_out.hidden_states[args.source_layer]
        if args.bf16:
            src_h = src_h.bfloat16()
        src_mask = src_enc.attention_mask

    # 2. Target: Mistral embeddings for the question
    with torch.no_grad():
        target = get_target_embeddings(
            questions, tgt_tok, tgt_model, device, args.soft_tokens
        )
        if args.bf16:
            target = target.bfloat16()

    B = target.shape[0]

    # 3. Sample noise and timestep
    noise = torch.randn_like(target)
    t = torch.rand(B, device=device, dtype=target.dtype)

    # 4. Create noisy interpolation: x_t = t * target + (1-t) * noise
    t_expand = t.view(B, 1, 1)
    x_t = t_expand * target + (1 - t_expand) * noise

    # 5. Predict velocity
    v_pred = bridge(x_t, t, src_h, src_mask)

    # 6. True velocity (constant in Rectified Flow)
    v_true = target - noise

    # 7. MSE loss
    loss = nn.functional.mse_loss(v_pred, v_true)

    # Compute auxiliary metrics
    with torch.no_grad():
        # Cosine similarity between predicted and true velocity
        cos_sim = nn.functional.cosine_similarity(
            v_pred.flatten(1), v_true.flatten(1), dim=1
        ).mean()

        # RMS of predictions
        pred_rms = v_pred.pow(2).mean().sqrt()
        true_rms = v_true.pow(2).mean().sqrt()

    return loss, {
        "loss": loss.item(),
        "cos_sim": cos_sim.item(),
        "pred_rms": pred_rms.item(),
        "true_rms": true_rms.item(),
    }


def main():
    setup_ddp()
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}")

    if local_rank == 0:
        print("=" * 70)
        print("Phase 12: Diffusion Bridge with Rectified Flow")
        print("=" * 70)
        print("THE FIX: Stop predicting → start generating")
        print("")
        print("Why this works:")
        print("  - Regression outputs AVERAGE of valid vectors (blurry)")
        print("  - Diffusion generates vectors ON the manifold (sharp)")
        print("")
        print(f"Architecture: DiT with {args.depth} layers, {args.heads} heads")
        print(f"Training: Rectified Flow (velocity prediction)")
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
    bridge = LatentBridgeV12(
        args,
        src_dim=src_model.config.hidden_size,
        tgt_dim=tgt_model.config.hidden_size,
        num_latents=args.soft_tokens,
        depth=args.depth,
        heads=args.heads,
    )
    if args.bf16:
        bridge = bridge.bfloat16()
    bridge.train()
    bridge.to(device)

    if torch.distributed.is_initialized():
        bridge = DDP(bridge, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=0.01)

    # Warmup scheduler
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Load data
    ds = load_dataset("gsm8k", "main", split="train")
    if torch.distributed.is_initialized():
        ds = ds.shard(world_size, local_rank)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if local_rank == 0:
        print(f"\nTraining on {len(ds)} samples")
        print(f"Steps: {args.steps}, LR: {args.lr}")
        print("Monitoring: loss (MSE), cos_sim (velocity alignment)")
        print("Starting training loop...")

    progress = tqdm(range(args.steps), disable=(local_rank != 0),
                    desc="V12 Diffusion", ncols=100)
    iter_dl = iter(dl)
    running_loss = 0.0
    running_cos = 0.0

    for step in progress:
        try:
            batch = next(iter_dl)
        except StopIteration:
            iter_dl = iter(dl)
            batch = next(iter_dl)

        loss, metrics = train_step(batch, src_tok, tgt_tok, src_model, bridge,
                                   tgt_model, device, args)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        running_loss += metrics['loss']
        running_cos += metrics['cos_sim']

        progress.set_postfix({
            "loss": f"{metrics['loss']:.4f}",
            "cos": f"{metrics['cos_sim']:.3f}"
        })

        # Periodic logging
        if local_rank == 0 and (step + 1) % 100 == 0:
            avg_loss = running_loss / 100
            avg_cos = running_cos / 100
            lr = scheduler.get_last_lr()[0]
            print(f"\n[Step {step+1}/{args.steps}]")
            print(f"  MSE Loss: {avg_loss:.4f}")
            print(f"  Cos Sim: {avg_cos:.3f} (1.0 = perfect velocity prediction)")
            print(f"  LR: {lr:.2e}")
            running_loss = 0.0
            running_cos = 0.0

        # Save checkpoints
        if local_rank == 0 and (step + 1) % args.save_every == 0:
            bridge_to_save = bridge.module if torch.distributed.is_initialized() else bridge
            torch.save(bridge_to_save.state_dict(), args.save_path)
            print(f"  Checkpoint saved: {args.save_path}")

    # Final save
    if local_rank == 0:
        bridge_to_save = bridge.module if torch.distributed.is_initialized() else bridge
        torch.save(bridge_to_save.state_dict(), args.save_path)
        print("\n" + "=" * 70)
        print("Phase 12 Training Complete!")
        print(f"Final checkpoint: {args.save_path}")
        print("=" * 70)
        print("\nNEXT: Run eval to generate soft tokens from noise")
        print("Success = Outputs vary per input and contain entities")


if __name__ == "__main__":
    main()
