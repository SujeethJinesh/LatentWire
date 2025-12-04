#!/usr/bin/env python
# telepathy/train_telepathy_gsm8k.py
"""
Phase 18: GSM8K Reasoning with Latent Chain-of-Thought

GOAL: Transfer mathematical reasoning from Llama to Mistral via latent CoT.

GSM8K: Grade School Math 8K
- ~8K training examples of multi-step math word problems
- Each has: question, chain-of-thought, final numeric answer
- Example: "Janet has 3 apples..." -> "First... Then... #### 7"

APPROACH: Latent Chain-of-Thought
- Llama reads the question and produces hidden states
- Bridge generates N reasoning steps (latent tokens)
- Each step can attend to question + previous reasoning
- Mistral receives all latent tokens and predicts final answer

SUCCESS CRITERIA:
- Random: ~0% (math is hard)
- If accuracy > 10%: Bridge transmits some reasoning signal
- If accuracy > 30%: Bridge is working for reasoning
- If accuracy matches Mistral text: Perfect reasoning transfer
"""
import os
import re
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse

from latent_cot_bridge import LatentCoTBridge


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
    # Optimal from SST-2/AG News ablation
    parser.add_argument("--source_layer", type=int, default=31)
    parser.add_argument("--soft_tokens", type=int, default=8)  # Per reasoning step
    parser.add_argument("--cot_steps", type=int, default=4)    # Number of reasoning steps
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)      # Lower LR for reasoning
    parser.add_argument("--batch_size", type=int, default=8)   # Smaller batch (longer sequences)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5000)     # More steps for reasoning
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--save_path", default="bridge_gsm8k.pt")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--diversity_weight", type=float, default=0.1)
    parser.add_argument("--bf16", action="store_true", default=True)
    return parser.parse_args()


def extract_answer(text):
    """Extract numeric answer from GSM8K format '#### number'."""
    # Look for #### pattern
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    if match:
        # Remove commas and convert to number
        return match.group(1).replace(',', '')

    # Fallback: look for last number in text
    numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(',', '')

    return None


def quick_eval(bridge, src_model, tgt_model, src_tok, tgt_tok, eval_ds, device, args, step):
    """Quick evaluation during training."""
    bridge_module = bridge.module if hasattr(bridge, 'module') else bridge
    bridge_module.eval()

    correct = 0
    total = 0
    indices = list(range(0, min(50, len(eval_ds))))

    print(f"\n{'='*60}")
    print(f"QUICK EVAL @ Step {step}")
    print(f"{'='*60}")

    for i in indices:
        item = eval_ds[i]
        question = item['question']
        gold_answer = extract_answer(item['answer'])

        if gold_answer is None:
            continue

        # Source: Llama reads question
        src_input = f"Question: {question}\nLet me solve this step by step."
        with torch.no_grad():
            src_enc = src_tok(src_input, return_tensors="pt", truncation=True, max_length=256).to(device)
            src_out = src_model(**src_enc, output_hidden_states=True)
            src_h = src_out.hidden_states[args.source_layer]
            if args.bf16:
                src_h = src_h.bfloat16()

            # Bridge with CoT
            soft_tokens, _, _, _ = bridge_module(src_h, src_enc.attention_mask)

            # Target: Mistral generates answer
            primer = "The answer is"
            primer_enc = tgt_tok(primer, return_tensors="pt", add_special_tokens=False).to(device)
            primer_embeds = tgt_model.get_input_embeddings()(primer_enc.input_ids)
            if args.bf16:
                primer_embeds = primer_embeds.bfloat16()

            combined_embeds = torch.cat([primer_embeds, soft_tokens], dim=1)
            attn_mask = torch.ones(combined_embeds.shape[:2], device=device, dtype=torch.long)

            out_ids = tgt_model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=attn_mask,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tgt_tok.eos_token_id,
            )
            output = tgt_tok.decode(out_ids[0], skip_special_tokens=True).strip()

        # Extract predicted answer
        pred_answer = extract_answer(output)
        if pred_answer is None:
            # Try to find any number
            numbers = re.findall(r'-?\d+', output)
            pred_answer = numbers[0] if numbers else "?"

        is_correct = str(pred_answer) == str(gold_answer)
        if is_correct:
            correct += 1
        total += 1

        # Print first 5 samples
        if i < 5:
            status = "CORRECT" if is_correct else "wrong"
            print(f"[{i}] {status}")
            print(f"  Q: {question[:60]}...")
            print(f"  Gold: {gold_answer} | Pred: {pred_answer}")
            print(f"  Output: {output[:50]}")

    accuracy = 100 * correct / max(total, 1)
    print(f"\nAccuracy: {correct}/{total} ({accuracy:.1f}%)")
    print(f"{'='*60}\n")

    bridge_module.train()
    return accuracy


def train_step(batch, src_tok, tgt_tok, src_model, bridge, tgt_model, device, args):
    """Single training step."""
    questions = batch['question']
    answers = batch['answer']
    B = len(questions)

    # Extract gold answers
    gold_answers = [extract_answer(a) for a in answers]

    # Source: Llama reads question
    src_texts = [f"Question: {q}\nLet me solve this step by step." for q in questions]

    with torch.no_grad():
        src_enc = src_tok(
            src_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=256
        ).to(device)
        src_out = src_model(**src_enc, output_hidden_states=True)
        src_h = src_out.hidden_states[args.source_layer]
        if args.bf16:
            src_h = src_h.bfloat16()
        src_mask = src_enc.attention_mask

    # Bridge with CoT (produces num_steps * soft_tokens latent tokens)
    bridge_module = bridge.module if hasattr(bridge, 'module') else bridge
    soft_tokens, aux_loss, diversity, z_variance = bridge_module(src_h, src_mask)

    # Batch diversity loss
    batch_div_loss = torch.tensor(0.0, device=device)
    if B > 1:
        flat_tokens = soft_tokens.reshape(B, -1).float()
        flat_norm = F.normalize(flat_tokens, dim=1)
        sim_matrix = torch.mm(flat_norm, flat_norm.t())
        mask = ~torch.eye(B, dtype=torch.bool, device=device)
        off_diag_sim = sim_matrix[mask].mean()
        batch_div_loss = off_diag_sim

    # Target: Mistral predicts answer
    primer_text = "The answer is"
    with torch.no_grad():
        primer_enc = tgt_tok(
            [primer_text] * B, return_tensors="pt", add_special_tokens=False
        ).to(device)
        primer_embeds = tgt_model.get_input_embeddings()(primer_enc.input_ids)
        if args.bf16:
            primer_embeds = primer_embeds.bfloat16()

        # Target answers: " {number}<eos>"
        tgt_texts = [f" {a}{tgt_tok.eos_token}" if a else f" 0{tgt_tok.eos_token}" for a in gold_answers]
        tgt_enc = tgt_tok(
            tgt_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=32, add_special_tokens=False
        ).to(device)
        answer_embeds = tgt_model.get_input_embeddings()(tgt_enc.input_ids)
        if args.bf16:
            answer_embeds = answer_embeds.bfloat16()

    # Concatenate: [Primer] + [Soft Tokens (CoT)] + [Answer]
    inputs_embeds = torch.cat([primer_embeds, soft_tokens, answer_embeds], dim=1)

    K = soft_tokens.shape[1]  # Total latent tokens (cot_steps * soft_tokens)
    P_len = primer_embeds.shape[1]

    # Labels: Mask primer and soft tokens, predict answer
    ignore_prefix = torch.full((B, P_len + K), -100, dtype=torch.long, device=device)
    answer_labels = tgt_enc.input_ids.clone()
    answer_labels[tgt_enc.attention_mask == 0] = -100
    labels_tensor = torch.cat([ignore_prefix, answer_labels], dim=1)

    # Attention mask
    soft_mask = torch.ones(B, K, dtype=torch.long, device=device)
    full_mask = torch.cat([primer_enc.attention_mask, soft_mask, tgt_enc.attention_mask], dim=1)

    # Forward through Mistral
    outputs = tgt_model(
        inputs_embeds=inputs_embeds,
        attention_mask=full_mask,
        labels=labels_tensor
    )
    loss_lm = outputs.loss

    # Total loss
    total_loss = loss_lm + args.diversity_weight * batch_div_loss

    return total_loss, {
        "total": total_loss.item(),
        "lm": loss_lm.item(),
        "div": batch_div_loss.item(),
        "z_var": z_variance.item() if isinstance(z_variance, torch.Tensor) else z_variance
    }


def main():
    setup_ddp()
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}")

    if local_rank == 0:
        print("=" * 60)
        print("Phase 18: GSM8K with Latent Chain-of-Thought")
        print("=" * 60)
        print("")
        print("GOAL: Transfer mathematical reasoning via latent CoT")
        print("  - Task: Grade school math word problems")
        print("  - Method: Multi-step latent reasoning")
        print("")
        print("ARCHITECTURE:")
        print(f"  - Source layer: {args.source_layer}")
        print(f"  - Soft tokens per step: {args.soft_tokens}")
        print(f"  - CoT steps: {args.cot_steps}")
        print(f"  - Total latent tokens: {args.soft_tokens * args.cot_steps}")
        print("")
        print(f"Training: {args.steps} steps, batch={args.batch_size}")
        print("=" * 60)

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

    # Compute target RMS
    with torch.no_grad():
        tgt_embeds = tgt_model.get_input_embeddings().weight.float()
        target_rms = tgt_embeds.pow(2).mean(dim=1).sqrt().median().item()
        if local_rank == 0:
            print(f"Target embedding RMS: {target_rms:.4f}")

    # Initialize CoT bridge
    bridge = LatentCoTBridge(
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

    # Load GSM8K
    train_ds = load_dataset("openai/gsm8k", "main", split="train")
    eval_ds = load_dataset("openai/gsm8k", "main", split="test")

    if torch.distributed.is_initialized():
        train_ds = train_ds.shard(world_size, local_rank)

    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if local_rank == 0:
        print(f"\nTraining on {len(train_ds)} samples")
        print(f"Test set: {len(eval_ds)} samples")
        print("Starting training...\n")

    progress = tqdm(range(args.steps), disable=(local_rank != 0), desc="GSM8K-CoT", ncols=100)
    iter_dl = iter(dl)
    running = {"total": 0, "lm": 0, "div": 0, "z_var": 0}
    grad_accum = args.grad_accum

    for step in progress:
        optimizer.zero_grad()
        accum_loss_dict = {"total": 0, "lm": 0, "div": 0, "z_var": 0}

        for _ in range(grad_accum):
            try:
                batch = next(iter_dl)
            except StopIteration:
                iter_dl = iter(dl)
                batch = next(iter_dl)

            loss, loss_dict = train_step(
                batch, src_tok, tgt_tok, src_model, bridge, tgt_model, device, args
            )

            scaled_loss = loss / grad_accum
            scaled_loss.backward()

            for k in accum_loss_dict:
                accum_loss_dict[k] += loss_dict[k] / grad_accum

        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        optimizer.step()

        for k in running:
            running[k] += accum_loss_dict[k]

        progress.set_postfix({
            "lm": f"{accum_loss_dict['lm']:.2f}",
            "div": f"{accum_loss_dict['div']:.3f}"
        })

        # Periodic logging
        if local_rank == 0 and (step + 1) % 50 == 0:
            avg = {k: v / 50 for k, v in running.items()}
            print(f"\n[Step {step+1}/{args.steps}]")
            print(f"  LM Loss: {avg['lm']:.3f}")
            print(f"  Batch Div Loss: {avg['div']:.4f}")
            print(f"  Z Variance: {avg['z_var']:.4f}")
            running = {k: 0 for k in running}

        # Quick eval
        if local_rank == 0 and (step + 1) % args.eval_every == 0:
            quick_eval(bridge, src_model, tgt_model, src_tok, tgt_tok, eval_ds, device, args, step + 1)

        # Save checkpoint
        if local_rank == 0 and (step + 1) % args.save_every == 0:
            bridge_to_save = bridge.module if torch.distributed.is_initialized() else bridge
            torch.save(bridge_to_save.state_dict(), args.save_path)
            print(f"  Checkpoint saved: {args.save_path}")

    # Final save
    if local_rank == 0:
        bridge_to_save = bridge.module if torch.distributed.is_initialized() else bridge
        torch.save(bridge_to_save.state_dict(), args.save_path)
        print("\n" + "=" * 60)
        print("Phase 18 GSM8K-CoT Training Complete!")
        print(f"Checkpoint: {args.save_path}")
        print("=" * 60)


if __name__ == "__main__":
    main()
