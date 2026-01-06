#!/usr/bin/env python
# telepathy/train_telepathy_v11.py
"""
Phase 11: Bottleneck Supervision

THE FIX: Weight the first K tokens 100x more than the rest.

Why this works:
- Position 128 predicts Q[0] from ONLY [Soft Tokens] + [BOS]
- Position 129 predicts Q[1] from [Soft Tokens] + [BOS] + [Q[0]]
- ...
- Position 128+K predicts Q[K] from [Soft Tokens] + [BOS] + [Q[0:K]]

The first few positions have NO teacher-forced context to cheat from.
If the model can predict "Janet" at position 128, it MUST be reading soft tokens.

By weighting these positions 100x, we force the optimizer to make the bridge
actually encode information, rather than letting the model ignore soft tokens.
"""
import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from latent_bridge_v9 import LatentBridgeV9
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
    parser.add_argument("--soft_tokens", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=100)

    # Phase 11: Bottleneck Supervision
    parser.add_argument("--bottleneck_tokens", type=int, default=10,
                        help="Number of initial tokens to weight heavily")
    parser.add_argument("--bottleneck_weight", type=float, default=100.0,
                        help="Multiplier for bottleneck tokens (100x = critical)")

    parser.add_argument("--save_path", default="bridge_v11.pt")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def train_step(batch, src_tok, tgt_tok, src_model, bridge, tgt_model, device, args):
    """
    Phase 11 Training Step with Bottleneck Supervision.

    Key difference from V10: Custom loss weighting on first K tokens.
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

    # 2. Bridge Forward
    soft_tokens, _ = bridge(src_h, src_mask)

    # 3. Target: [Question] + [Answer]
    tgt_texts = [f"{q}\nAnswer: {a}{tgt_tok.eos_token}" for q, a in zip(questions, answers)]

    tgt_enc = tgt_tok(tgt_texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=1024).to(device)
    tgt_ids = tgt_enc.input_ids
    tgt_embeds = tgt_model.get_input_embeddings()(tgt_ids)

    # 4. Construct input: [Soft_Tokens] + [Target_Embeds]
    combined_embeds = torch.cat([soft_tokens, tgt_embeds], dim=1)

    # 5. Labels: -100 on soft tokens, then target ids
    B, K, _ = soft_tokens.shape
    ignore_prefix = torch.full((B, K), -100, dtype=torch.long, device=device)
    labels = torch.cat([ignore_prefix, tgt_ids], dim=1)

    # 6. Attention mask
    soft_mask = torch.ones(B, K, dtype=torch.long, device=device)
    full_mask = torch.cat([soft_mask, tgt_enc.attention_mask], dim=1)

    # 7. Forward through Mistral (get logits, not loss)
    outputs = tgt_model(inputs_embeds=combined_embeds, attention_mask=full_mask)
    logits = outputs.logits

    # 8. PHASE 11 FIX: Custom weighted loss
    # Shift for causal LM: predict next token
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Per-token cross entropy (no reduction)
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    loss_per_token = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    loss_per_token = loss_per_token.view(B, -1)

    # Create weight mask
    # Soft tokens are at positions 0 to K-1 in combined_embeds
    # In shifted labels, position K-1 corresponds to predicting tgt_ids[0]
    # (because we removed the last position and shifted)
    loss_weights = torch.ones_like(loss_per_token)

    # The "bottleneck zone": first few text tokens after soft tokens
    # These positions have minimal teacher-forced context
    bottleneck_start = K - 1  # First text token prediction
    bottleneck_end = min(bottleneck_start + args.bottleneck_tokens, loss_weights.size(1))

    if bottleneck_start < loss_weights.size(1):
        loss_weights[:, bottleneck_start:bottleneck_end] = args.bottleneck_weight

    # Compute weighted mean over valid (non-masked) positions
    valid_mask = (shift_labels != -100).float()
    weighted_loss = (loss_per_token * loss_weights * valid_mask).sum()
    normalizer = (loss_weights * valid_mask).sum()

    if normalizer > 0:
        weighted_loss = weighted_loss / normalizer
    else:
        weighted_loss = loss_per_token.mean()

    # Also compute unweighted loss for monitoring
    unweighted_loss = (loss_per_token * valid_mask).sum() / valid_mask.sum()

    # Compute bottleneck-only loss for monitoring
    bottleneck_mask = torch.zeros_like(loss_per_token)
    if bottleneck_start < bottleneck_mask.size(1):
        bottleneck_mask[:, bottleneck_start:bottleneck_end] = 1.0
    bottleneck_loss = (loss_per_token * bottleneck_mask * valid_mask).sum()
    bottleneck_normalizer = (bottleneck_mask * valid_mask).sum()
    if bottleneck_normalizer > 0:
        bottleneck_loss = bottleneck_loss / bottleneck_normalizer
    else:
        bottleneck_loss = torch.tensor(0.0)

    return weighted_loss, {
        "weighted_loss": weighted_loss.item(),
        "unweighted_loss": unweighted_loss.item(),
        "bottleneck_loss": bottleneck_loss.item()
    }


def main():
    setup_ddp()
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}")

    if local_rank == 0:
        print("=" * 70)
        print("Phase 11: Bottleneck Supervision")
        print("=" * 70)
        print("THE FIX: Weight first K tokens 100x to break Teacher Forcing Trap")
        print("")
        print(f"Bottleneck tokens: {args.bottleneck_tokens}")
        print(f"Bottleneck weight: {args.bottleneck_weight}x")
        print("")
        print("If the model can predict 'Janet' at position 128, it MUST")
        print("be reading from soft tokens (no teacher context available).")
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

    # Bridge (reuse V9 architecture)
    bridge = LatentBridgeV9(
        args, src_model.config.hidden_size, tgt_model.config.hidden_size,
        target_rms=target_rms, src_vocab_size=src_model.config.vocab_size
    )
    if args.bf16:
        bridge = bridge.bfloat16()
    bridge.train()
    bridge.to(device)

    if torch.distributed.is_initialized():
        # find_unused_parameters=True because BoW head from V9 is unused
        bridge = DDP(bridge, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=0.01)

    # Load data
    ds = load_dataset("gsm8k", "main", split="train")
    if torch.distributed.is_initialized():
        ds = ds.shard(world_size, local_rank)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if local_rank == 0:
        print(f"\nTraining on {len(ds)} samples")
        print("Monitoring: weighted_loss, unweighted_loss, bottleneck_loss")
        print("Starting training loop...")

    progress = tqdm(range(args.steps), disable=(local_rank != 0),
                    desc="V11 Training", ncols=100)
    iter_dl = iter(dl)
    running_weighted = 0.0
    running_unweighted = 0.0
    running_bottleneck = 0.0

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

        running_weighted += loss_dict['weighted_loss']
        running_unweighted += loss_dict['unweighted_loss']
        running_bottleneck += loss_dict['bottleneck_loss']

        progress.set_postfix({
            "W": f"{loss_dict['weighted_loss']:.2f}",
            "BN": f"{loss_dict['bottleneck_loss']:.2f}"
        })

        # Periodic logging
        if local_rank == 0 and (step + 1) % 50 == 0:
            bridge_inner = bridge.module if torch.distributed.is_initialized() else bridge
            current_scale = bridge_inner.output_scale.item()
            print(f"\n[Step {step+1}/{args.steps}]")
            print(f"  Weighted Loss: {running_weighted/50:.4f}")
            print(f"  Unweighted Loss: {running_unweighted/50:.4f}")
            print(f"  Bottleneck Loss: {running_bottleneck/50:.4f} (CRITICAL)")
            print(f"  Output Scale: {current_scale:.4f}")
            running_weighted = 0.0
            running_unweighted = 0.0
            running_bottleneck = 0.0

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
        print("Phase 11 Training Complete!")
        print(f"Final checkpoint: {args.save_path}")
        print(f"Final output scale: {bridge_to_save.output_scale.item():.4f}")
        print("=" * 70)
        print("\nNEXT: Run eval to check if first tokens are predicted correctly")
        print("Success = Output starts with actual question content, not templates")


if __name__ == "__main__":
    main()
