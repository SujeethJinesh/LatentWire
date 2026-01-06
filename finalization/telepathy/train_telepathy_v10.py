#!/usr/bin/env python
# telepathy/train_telepathy_v10.py
"""
Phase 10: Auto-Encoder Pivot

THE FIX: Force Mistral to regenerate the QUESTION from soft tokens.

Previous phases failed because auxiliary losses (BoW, recon) could be satisfied
without encoding info in the format Mistral actually reads. The bridge "cheated"
by packing entity data into dimensions the auxiliary head could read but that
Mistral's attention filtered out.

V10 Solution: The only loss is LM loss on [Question + Answer].
If Mistral must output "Janet has 16 ducks", the bridge MUST encode that
information in a format Mistral can decode. No cheating possible.
"""
import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from latent_bridge_v9 import LatentBridgeV9  # Reuse V9 architecture
import argparse


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
    parser.add_argument("--stats_path", default="stats.pt")
    parser.add_argument("--source_layer", type=int, default=16)
    parser.add_argument("--soft_tokens", type=int, default=128)  # Compression challenge
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=100)

    parser.add_argument("--save_path", default="bridge_v10.pt")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def train_step(batch, src_tok, tgt_tok, src_model, bridge, tgt_model, device, args):
    """
    V10 Training Step: Reconstruction + Reasoning

    Key difference: Target includes QUESTION text, not just answer.
    Forces bridge to encode question content in Mistral-readable format.
    """
    questions = batch['question']
    answers = batch['answer']

    # 1. Source Input (Llama reads Question)
    src_texts = [f"Question: {q}\nAnswer:" for q in questions]

    with torch.no_grad():
        src_enc = src_tok(src_texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=1024).to(device)
        src_out = src_model(**src_enc, output_hidden_states=True)
        src_h = src_out.hidden_states[args.source_layer]
        if args.bf16:
            src_h = src_h.bfloat16()
        src_mask = src_enc.attention_mask

    # 2. Bridge Forward (ignore BoW logits - we rely on main pathway only)
    soft_tokens, _ = bridge(src_h, src_mask)

    # 3. THE V10 FIX: Target includes Question + Answer
    # Mistral must regenerate "Janet has 16 ducks... Answer: 18"
    # This forces soft tokens to encode the question content
    tgt_texts = [f"{q}\nAnswer: {a}{tgt_tok.eos_token}" for q, a in zip(questions, answers)]

    tgt_enc = tgt_tok(tgt_texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=1024).to(device)
    tgt_ids = tgt_enc.input_ids
    tgt_embeds = tgt_model.get_input_embeddings()(tgt_ids)

    # 4. Construct input: [Soft_Tokens] + [Target_Embeds]
    # Soft tokens act as the "prompt" that encodes the question
    # Target embeds are what we're training to predict
    combined_embeds = torch.cat([soft_tokens, tgt_embeds], dim=1)

    # 5. Labels: -100 on soft tokens (no loss), then target ids
    B, K, _ = soft_tokens.shape
    ignore_prefix = torch.full((B, K), -100, dtype=torch.long, device=device)
    labels = torch.cat([ignore_prefix, tgt_ids], dim=1)

    # 6. Attention mask
    soft_mask = torch.ones(B, K, dtype=torch.long, device=device)
    full_mask = torch.cat([soft_mask, tgt_enc.attention_mask], dim=1)

    # 7. Forward through Mistral
    outputs = tgt_model(inputs_embeds=combined_embeds, attention_mask=full_mask, labels=labels)

    # The loss includes both QUESTION reconstruction and ANSWER generation
    # Gradient forces soft tokens to encode question content in Mistral-readable format
    loss_lm = outputs.loss

    return loss_lm, {"loss": loss_lm.item()}


def main():
    setup_ddp()
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}")

    if local_rank == 0:
        print("=" * 70)
        print("Phase 10: Auto-Encoder Pivot")
        print("=" * 70)
        print("THE FIX: Force Mistral to regenerate QUESTION from soft tokens")
        print("If output must be 'Janet has ducks', bridge MUST encode that info")
        print("No cheating - main pathway IS the supervision signal")
        print("=" * 70)
        print(f"Source Layer: {args.source_layer}")
        print(f"Soft Tokens: {args.soft_tokens}")
        print(f"Steps: {args.steps}")
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

    # Get target RMS for scaling
    with torch.no_grad():
        tgt_embeds = tgt_model.get_input_embeddings().weight.float()
        target_rms = tgt_embeds.pow(2).mean(dim=1).sqrt().median().item()
        if local_rank == 0:
            print(f"Target embedding RMS: {target_rms:.4f}")

    # Using V9 Bridge architecture (BoW head ignored, main pathway used)
    bridge = LatentBridgeV9(
        args, src_model.config.hidden_size, tgt_model.config.hidden_size,
        target_rms=target_rms, src_vocab_size=src_model.config.vocab_size
    )
    if args.bf16:
        bridge = bridge.bfloat16()
    bridge.train()
    bridge.to(device)

    if torch.distributed.is_initialized():
        # find_unused_parameters=True because we ignore the BoW head from V9
        bridge = DDP(bridge, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=0.01)

    # Load data
    ds = load_dataset("gsm8k", "main", split="train")
    if torch.distributed.is_initialized():
        ds = ds.shard(world_size, local_rank)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if local_rank == 0:
        print(f"\nTraining on {len(ds)} samples")
        print("Target: [Question] + [Answer] (forces question reconstruction)")
        print("Starting training loop...")

    progress = tqdm(range(args.steps), disable=(local_rank != 0),
                    desc="V10 Training", ncols=100)
    iter_dl = iter(dl)
    running_loss = 0.0

    for step in progress:
        try:
            batch = next(iter_dl)
        except StopIteration:
            iter_dl = iter(dl)
            batch = next(iter_dl)

        loss, loss_dict = train_step(batch, src_tok, tgt_tok, src_model, bridge,
                                      tgt_model, device, args)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), args.grad_clip)
        optimizer.step()

        running_loss += loss_dict['loss']
        progress.set_postfix({"Loss": f"{loss_dict['loss']:.3f}"})

        # Periodic logging
        if local_rank == 0 and (step + 1) % 50 == 0:
            avg_loss = running_loss / 50
            bridge_inner = bridge.module if torch.distributed.is_initialized() else bridge
            current_scale = bridge_inner.output_scale.item()
            print(f"\n[Step {step+1}/{args.steps}] Loss: {avg_loss:.4f} | Scale: {current_scale:.4f}")
            running_loss = 0.0

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
        print("Phase 10 Training Complete!")
        print(f"Final checkpoint: {args.save_path}")
        print(f"Final output scale: {bridge_to_save.output_scale.item():.4f}")
        print("=" * 70)
        print("\nNEXT: Run eval to check if Mistral can 'read back' the question")
        print("Success = Output starts with question content (Janet, ducks, etc)")


if __name__ == "__main__":
    main()
