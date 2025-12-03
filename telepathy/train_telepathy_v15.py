#!/usr/bin/env python
# telepathy/train_telepathy_v15.py
"""
Phase 15: FSQ-Telepathy Training (Finite Scalar Quantization)

Training with LM Loss only:
- LM Loss: Cross-entropy on answer generation (functional correctness)
- FSQ: No auxiliary loss needed (unlike VQ)

Key differences from VQ attempt:
- FSQ has NO codebook = NO collapse possible
- 8^8 = 16M effective codes (vs 4096 VQ codes that collapsed)
- Pure LM loss training (no VQ loss balancing needed)
- 1-step inference, no diffusion
"""
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from datasets import load_dataset
from tqdm import tqdm
import argparse

from latent_bridge_v15 import LatentBridgeV15


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
    parser.add_argument("--stats_path", default=None, help="Path to calibration stats")
    parser.add_argument("--source_layer", type=int, default=16)
    parser.add_argument("--soft_tokens", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_path", default="bridge_v15.pt")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--bf16", action="store_true", default=True)
    return parser.parse_args()


def extract_answer_text(answer_str):
    """Extract just the numerical answer from GSM8K format."""
    # GSM8K answers have format: "explanation... #### 42"
    if "####" in answer_str:
        return answer_str.split("####")[-1].strip()
    return answer_str.strip()


def train_step(batch, src_tok, tgt_tok, src_model, bridge, tgt_model, device, args):
    """
    Single training step with LM loss (FSQ has no auxiliary loss).
    """
    questions = batch['question']
    answers = batch['answer']

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

    # Bridge Forward: Get quantized soft tokens (FSQ aux_loss is always 0)
    bridge_module = bridge.module if hasattr(bridge, 'module') else bridge
    soft_tokens, aux_loss, diversity, z_variance = bridge_module(src_h, src_mask)

    # Target: Mistral generates the answer
    # Use full answer for teacher forcing (includes reasoning)
    tgt_texts = [f"{a}{tgt_tok.eos_token}" for a in answers]

    with torch.no_grad():
        # Primer embeddings
        primer_text = "Answer: "
        primer_enc = tgt_tok(
            [primer_text] * len(questions),
            return_tensors="pt",
            add_special_tokens=False
        ).to(device)
        primer_embeds = tgt_model.get_input_embeddings()(primer_enc.input_ids)
        if args.bf16:
            primer_embeds = primer_embeds.bfloat16()

        # Answer embeddings (for teacher forcing)
        tgt_enc = tgt_tok(
            tgt_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=512
        ).to(device)
        answer_embeds = tgt_model.get_input_embeddings()(tgt_enc.input_ids)
        if args.bf16:
            answer_embeds = answer_embeds.bfloat16()

    # Concatenate: [soft_tokens] + [primer] + [answer]
    # Note: soft_tokens represent the question in Mistral's space
    inputs_embeds = torch.cat([soft_tokens, primer_embeds, answer_embeds], dim=1)

    B = len(questions)
    K = soft_tokens.shape[1]
    P_len = primer_embeds.shape[1]

    # Labels: ignore soft tokens and primer, predict answer
    ignore_prefix = torch.full((B, K + P_len), -100, dtype=torch.long, device=device)
    answer_labels = tgt_enc.input_ids.clone()
    answer_labels[tgt_enc.attention_mask == 0] = -100  # Mask padding
    labels = torch.cat([ignore_prefix, answer_labels], dim=1)

    # Attention mask
    soft_mask = torch.ones(B, K, dtype=torch.long, device=device)
    full_mask = torch.cat([soft_mask, primer_enc.attention_mask, tgt_enc.attention_mask], dim=1)

    # Forward through Mistral
    outputs = tgt_model(
        inputs_embeds=inputs_embeds,
        attention_mask=full_mask,
        labels=labels
    )
    loss_lm = outputs.loss

    # Total loss = LM loss only (no diversity loss for continuous mode)
    # FSQ diversity loss was removed because discrete bottlenecks collapsed in all tests
    total_loss = loss_lm

    return total_loss, {
        "total": total_loss.item(),
        "lm": loss_lm.item(),
        "z_var": z_variance.item() if hasattr(z_variance, 'item') else z_variance,
        "div": diversity  # 1.0 for continuous, 0-1 for FSQ
    }


def main():
    setup_ddp()
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}")

    if local_rank == 0:
        print("=" * 70)
        print("Phase 15: Continuous Telepathy (No Quantization)")
        print("=" * 70)
        print("")
        print("WHY CONTINUOUS INSTEAD OF DISCRETE:")
        print("  - VQ collapsed to 1 code despite entropy bonus")
        print("  - FSQ 8-dim collapsed to div=0")
        print("  - FSQ 32-dim collapsed to div=0")
        print("  - FSQ + diversity loss collapsed to div=0")
        print("  - FSQ + variance loss collapsed to z_var=0")
        print("  -> 6 discrete attempts ALL COLLAPSED")
        print("")
        print("CONTINUOUS ARCHITECTURE:")
        print("  - Perceiver resampler: compress to 128 soft tokens")
        print("  - NO quantization (continuous values)")
        print("  - Pure LM loss training")
        print("  - 1-step inference")
        print("")
        print(f"Training: {args.steps} steps, batch={args.batch_size}")
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

    # Compute target RMS for output scaling
    with torch.no_grad():
        tgt_embeds = tgt_model.get_input_embeddings().weight.float()
        target_rms = tgt_embeds.pow(2).mean(dim=1).sqrt().median().item()
        if local_rank == 0:
            print(f"Target embedding RMS: {target_rms:.4f}")

    # Initialize bridge
    bridge = LatentBridgeV15(
        args,
        src_dim=src_model.config.hidden_size,
        tgt_dim=tgt_model.config.hidden_size,
        target_rms=target_rms
    )
    if args.bf16:
        bridge = bridge.bfloat16()
    bridge.train()
    bridge.to(device)

    if torch.distributed.is_initialized():
        bridge = DDP(bridge, device_ids=[local_rank])

    # Optimizer with warmup
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.steps
    )

    # Load data
    ds = load_dataset("gsm8k", "main", split="train")
    if torch.distributed.is_initialized():
        ds = ds.shard(world_size, local_rank)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if local_rank == 0:
        print(f"\nTraining on {len(ds)} samples")
        print("Starting training loop...")

    progress = tqdm(range(args.steps), disable=(local_rank != 0),
                    desc="V15 Continuous", ncols=100)
    iter_dl = iter(dl)
    running = {"total": 0, "lm": 0, "z_var": 0, "div": 0}
    grad_accum = args.grad_accum

    for step in progress:
        optimizer.zero_grad()
        accum_loss_dict = {"total": 0, "lm": 0, "z_var": 0, "div": 0}

        # Gradient accumulation loop
        for accum_step in range(grad_accum):
            try:
                batch = next(iter_dl)
            except StopIteration:
                iter_dl = iter(dl)
                batch = next(iter_dl)

            loss, loss_dict = train_step(
                batch, src_tok, tgt_tok, src_model, bridge, tgt_model, device, args
            )

            # Scale loss for accumulation
            scaled_loss = loss / grad_accum
            scaled_loss.backward()

            for k in accum_loss_dict:
                accum_loss_dict[k] += loss_dict[k] / grad_accum

        torch.nn.utils.clip_grad_norm_(bridge.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        for k in running:
            running[k] += accum_loss_dict[k]

        progress.set_postfix({
            "lm": f"{accum_loss_dict['lm']:.2f}",
            "div": f"{accum_loss_dict['div']:.2f}"  # Diversity: should stay high (0-1)
        })

        # Periodic logging
        if local_rank == 0 and (step + 1) % 50 == 0:
            avg = {k: v / 50 for k, v in running.items()}
            current_lr = scheduler.get_last_lr()[0]
            print(f"\n[Step {step+1}/{args.steps}]")
            print(f"  LM Loss: {avg['lm']:.3f}")
            print(f"  Z Variance: {avg['z_var']:.4f} (output variance)")
            print(f"  LR: {current_lr:.2e}")
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
        print("Phase 15 FSQ Training Complete!")
        print(f"Final checkpoint: {args.save_path}")
        print("=" * 70)
        print("\nKEY METRICS TO CHECK:")
        print("  - LM loss should decrease steadily")
        print("  - NO diversity collapse possible (continuous)")
        print("  - Outputs should be coherent and relevant")
        print("  - Compare to text baseline in eval")


if __name__ == "__main__":
    main()
