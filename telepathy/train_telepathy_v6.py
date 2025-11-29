#!/usr/bin/env python
# telepathy/train_telepathy_v6.py
"""
Latent Telepathy Phase 6: The Translator Pivot

Problem in V5: We anchored Soft Tokens to the ANSWER embeddings.
Result: The Bridge tried to solve the math (Q->A) and failed, losing entities.

Fix in V6: Anchor Soft Tokens to the QUESTION embeddings.
Goal: Bridge(Llama_Q) ~= Mistral(Question_Embeddings)
Logic: The bridge should transmit the Question to Mistral, and let Mistral
       do the reasoning to generate the Answer.
"""
import os
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from latent_bridge import LatentBridge
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
    parser.add_argument("--soft_tokens", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=100)

    # Loss Weights
    parser.add_argument("--anchor_weight", type=float, default=1.0)
    parser.add_argument("--contrastive_weight", type=float, default=0.1)
    parser.add_argument("--contrastive_temp", type=float, default=0.07)

    parser.add_argument("--save_path", default="bridge_v6.pt")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def contrastive_loss_fn(soft_tokens, temperature=0.07):
    """InfoNCE contrastive loss for unique representations per input."""
    features = soft_tokens.mean(dim=1).float()
    features = F.normalize(features, dim=1)
    logits = torch.matmul(features, features.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


def batch_anchor_loss(soft_tokens, target_embeds, target_mask):
    """
    Cosine similarity loss between soft tokens and target embeddings.
    V6 CHANGE: target_embeds are now QUESTION embeddings, not answer embeddings.
    """
    # Mean pool soft tokens
    soft_mean = soft_tokens.mean(dim=1).float()

    # Mean pool target embeddings (Question Embeddings now)
    mask = target_mask.unsqueeze(-1).float()
    sum_mask = mask.sum(dim=1).clamp(min=1)
    target_mean = (target_embeds * mask).sum(dim=1) / sum_mask
    target_mean = target_mean.float()

    soft_norm = F.normalize(soft_mean, dim=1)
    target_norm = F.normalize(target_mean, dim=1)
    cos_sim = (soft_norm * target_norm).sum(dim=1).mean()
    return 1.0 - cos_sim


def train_step(batch, src_tok, tgt_tok, src_model, bridge, tgt_model, device, args):
    """Single training step with V6 Question-anchored loss."""
    questions = batch['question']
    answers = batch['answer']

    # Format texts
    src_texts = [f"Question: {q}\nAnswer:" for q in questions]
    tgt_answer_texts = [f"{a}{tgt_tok.eos_token}" for a in answers]
    # V6: Question format for anchor target
    tgt_q_texts = [f"Question: {q}\nAnswer:" for q in questions]

    # 1. Source Forward (Llama processes Question)
    with torch.no_grad():
        src_enc = src_tok(src_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        src_out = src_model(**src_enc, output_hidden_states=True)
        src_h = src_out.hidden_states[args.source_layer].clone()
        if args.bf16:
            src_h = src_h.bfloat16()
        src_mask = src_enc.attention_mask.clone()
        del src_out  # Free memory
        torch.cuda.empty_cache()

    # 2. Bridge Forward
    soft_tokens = bridge(src_h, src_mask)

    # 3. V6 ANCHOR: Target is Mistral's QUESTION embeddings (not answer!)
    # Note: Using embedding layer directly (no forward pass) to save memory
    with torch.no_grad():
        tgt_q_enc = tgt_tok(tgt_q_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        target_q_embeds = tgt_model.get_input_embeddings()(tgt_q_enc.input_ids)

    loss_anchor = batch_anchor_loss(soft_tokens, target_q_embeds, tgt_q_enc.attention_mask)
    loss_contrastive = contrastive_loss_fn(soft_tokens, args.contrastive_temp)

    # 4. LM Loss: Still train to generate answers
    # Input: [Primer] + [Soft Tokens] + [Answer]
    primer_text = "Answer: "
    B = len(questions)
    primer_enc = tgt_tok([primer_text] * B, return_tensors="pt", add_special_tokens=True).to(device)
    primer_embeds = tgt_model.get_input_embeddings()(primer_enc.input_ids)

    tgt_ans_enc = tgt_tok(tgt_answer_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    answer_embeds = tgt_model.get_input_embeddings()(tgt_ans_enc.input_ids)

    inputs_embeds = torch.cat([primer_embeds, soft_tokens, answer_embeds], dim=1)

    # Labels: ignore primer and soft tokens
    K = soft_tokens.shape[1]
    P_len = primer_embeds.shape[1]
    ignore_prefix = torch.full((B, P_len + K), -100, dtype=torch.long, device=device)
    answer_labels = tgt_ans_enc.input_ids.clone()
    answer_labels[tgt_ans_enc.attention_mask == 0] = -100
    labels = torch.cat([ignore_prefix, answer_labels], dim=1)

    # Attention Mask
    soft_mask = torch.ones(B, K, dtype=torch.long, device=device)
    full_mask = torch.cat([primer_enc.attention_mask, soft_mask, tgt_ans_enc.attention_mask], dim=1)

    outputs = tgt_model(inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=labels)
    loss_lm = outputs.loss

    total_loss = loss_lm + args.anchor_weight * loss_anchor + args.contrastive_weight * loss_contrastive

    return total_loss, {
        "total": total_loss.item(),
        "lm": loss_lm.item(),
        "anchor": loss_anchor.item(),
        "contrastive": loss_contrastive.item()
    }


def main():
    setup_ddp()
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}")

    if local_rank == 0:
        print("=" * 70)
        print("Phase 6: The Translator Pivot")
        print("=" * 70)
        print("KEY CHANGE: Anchor to QUESTION embeddings, not ANSWER embeddings")
        print("Goal: Bridge translates Q->Q, Mistral does the reasoning")
        print("=" * 70)
        print(f"Source Layer: {args.source_layer}")
        print(f"Soft Tokens: {args.soft_tokens}")
        print(f"Anchor Weight: {args.anchor_weight}")
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

    # Initialize bridge
    bridge = LatentBridge(args, src_model.config.hidden_size, tgt_model.config.hidden_size)
    if args.bf16:
        bridge = bridge.bfloat16()
    bridge.train()
    bridge.to(device)

    if torch.distributed.is_initialized():
        bridge = DDP(bridge, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=0.01)

    # Load data
    ds = load_dataset("gsm8k", "main", split="train")
    if torch.distributed.is_initialized():
        ds = ds.shard(world_size, local_rank)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if local_rank == 0:
        print(f"\nTraining on {len(ds)} samples")
        print(f"Starting training loop...")

    progress = tqdm(range(args.steps), disable=(local_rank != 0),
                    desc="V6 Training", ncols=100)
    iter_dl = iter(dl)
    running = {"total": 0, "lm": 0, "anchor": 0, "contrastive": 0}

    for step in progress:
        try:
            batch = next(iter_dl)
        except StopIteration:
            iter_dl = iter(dl)
            batch = next(iter_dl)

        loss, loss_dict = train_step(batch, src_tok, tgt_tok, src_model, bridge, tgt_model, device, args)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), args.grad_clip)
        optimizer.step()

        for k in running:
            running[k] += loss_dict[k]

        # Update progress bar
        progress.set_postfix({
            "L": f"{loss_dict['total']:.2f}",
            "LM": f"{loss_dict['lm']:.2f}",
            "Anc": f"{loss_dict['anchor']:.3f}",
            "Ctr": f"{loss_dict['contrastive']:.2f}"
        })

        # Periodic logging
        if local_rank == 0 and (step + 1) % 50 == 0:
            avg = {k: v / 50 for k, v in running.items()}
            print(f"\n[Step {step+1}/{args.steps}] Total: {avg['total']:.3f} | LM: {avg['lm']:.3f} | "
                  f"Anchor: {avg['anchor']:.4f} | Ctr: {avg['contrastive']:.3f}")
            running = {k: 0 for k in running}

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
        print("Phase 6 Training Complete!")
        print(f"Final checkpoint: {args.save_path}")
        print("=" * 70)


if __name__ == "__main__":
    main()
