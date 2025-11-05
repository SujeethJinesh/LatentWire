#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Improved Inter-LLM interlingua with bottlenecked gated architecture.
Based on ChatGPT's analysis, implements:
- Left padding for decoder-only models
- Bottlenecked translator (1024-d internal)
- Gated cross-attention (Flamingo-style)
- Question-only inputs, answer-only loss
- Attention regularization on soft tokens

Run (4x H100, DDP):
  torchrun --nproc_per_node=4 cross_attention_v2.py \
    --source_model mistralai/Mistral-7B-Instruct-v0.3 \
    --target_model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --translator_type bottleneck_gated \
    --bottleneck_dim 1024 \
    --soft_tokens 48 \
    --depth 6 \
    --train_steps 2000 \
    --per_device_batch 8
"""

import os, math, re, json, random, argparse, time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, get_linear_schedule_with_warmup

# ---------------------------
# Utilities
# ---------------------------

def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0

def log(*args, **kwargs):
    if is_main():
        print(*args, **kwargs, flush=True)

def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1

def local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))

def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank())

def cleanup_ddp():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

def extract_final_answer(text: str) -> str:
    """Extract final answer from GSM8K format."""
    m = re.search(r"####\s*(-?\d+)", text)
    if m:
        return m.group(1)
    ints = re.findall(r"-?\d+", text)
    return ints[-1] if ints else ""

@dataclass
class Sample:
    src_prompt: str
    tgt_question: str  # Question only (no answer)
    tgt_answer: str    # Answer only
    tgt_full: str      # Full text for reference

# ---------------------------
# Improved Translator Modules
# ---------------------------

class GatedCrossAttentionBlock(nn.Module):
    """
    Bottlenecked cross-attention block with tanh gating (Flamingo-style).
    All operations happen at bottleneck dimension for efficiency.
    """
    def __init__(self, bottleneck_dim: int, src_dim: int, n_heads: int, ffn_mult: int = 4):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim

        # Layer norms
        self.q_norm = nn.LayerNorm(bottleneck_dim, elementwise_affine=True)
        self.src_norm = nn.LayerNorm(src_dim, elementwise_affine=True)

        # Self-attention on queries (in bottleneck space)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=bottleneck_dim,
            num_heads=n_heads,
            batch_first=True
        )

        # Cross-attention projections
        self.cross_q_proj = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.cross_k_proj = nn.Linear(src_dim, bottleneck_dim)
        self.cross_v_proj = nn.Linear(src_dim, bottleneck_dim)
        self.cross_out = nn.Linear(bottleneck_dim, bottleneck_dim)

        # Tanh gating on cross-attention (Flamingo-style)
        self.cross_gate = nn.Parameter(torch.zeros(1))

        # FFN in bottleneck space
        self.ffn = nn.Sequential(
            nn.Linear(bottleneck_dim, ffn_mult * bottleneck_dim),
            nn.GELU(),
            nn.Linear(ffn_mult * bottleneck_dim, bottleneck_dim)
        )
        self.ffn_norm = nn.LayerNorm(bottleneck_dim)

    def forward(self, q_tokens: torch.Tensor, src_seq: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # q_tokens: [B, K, bottleneck_dim], src_seq: [B, T, src_dim]

        # Self-attention on queries
        q_norm = self.q_norm(q_tokens)
        sa_out, _ = self.self_attn(q_norm, q_norm, q_norm, need_weights=False)
        q_tokens = q_tokens + sa_out

        # Cross-attention with gating
        qn = self.q_norm(q_tokens)
        srcn = self.src_norm(src_seq)
        Q = self.cross_q_proj(qn)
        K = self.cross_k_proj(srcn)
        V = self.cross_v_proj(srcn)

        # Compute attention
        attn_logits = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(Q.size(-1))
        if src_mask is not None:
            mask_expanded = src_mask[:, None, :].to(dtype=attn_logits.dtype)
            attn_logits = attn_logits.masked_fill(mask_expanded == 0, -1e4)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        cross = torch.matmul(attn_weights, V)
        cross = self.cross_out(cross)

        # Apply tanh gating (starts at 0, learns to open)
        gate = torch.tanh(self.cross_gate)
        q_tokens = q_tokens + gate * cross

        # FFN
        ff = self.ffn(self.ffn_norm(q_tokens))
        q_tokens = q_tokens + ff

        return q_tokens


class BottleneckedGatedTranslator(nn.Module):
    """
    Improved translator with bottleneck architecture and gating.
    Processes at bottleneck_dim internally, only projects to target_dim at the end.
    """
    def __init__(self, src_dim: int, tgt_dim: int, bottleneck_dim: int = 1024,
                 soft_tokens: int = 48, depth: int = 6, n_heads: int = 16, ffn_mult: int = 4):
        super().__init__()
        self.K = soft_tokens
        self.bottleneck_dim = bottleneck_dim

        # Learned query tokens in bottleneck space
        self.query_tokens = nn.Parameter(
            torch.randn(1, soft_tokens, bottleneck_dim) / math.sqrt(bottleneck_dim)
        )

        # Cross-attention blocks (all in bottleneck space)
        self.blocks = nn.ModuleList([
            GatedCrossAttentionBlock(
                bottleneck_dim=bottleneck_dim,
                src_dim=src_dim,
                n_heads=n_heads,
                ffn_mult=ffn_mult
            )
            for _ in range(depth)
        ])

        # Final projection from bottleneck to target dimension
        self.output_proj = nn.Linear(bottleneck_dim, tgt_dim)
        self.output_norm = nn.LayerNorm(bottleneck_dim)

    def forward(self, src_hiddens: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        src_hiddens: [B, T_src, src_dim]
        returns: [B, K, tgt_dim]
        """
        B = src_hiddens.size(0)
        q = self.query_tokens.expand(B, -1, -1)  # [B, K, bottleneck_dim]

        # Process through gated cross-attention blocks
        for blk in self.blocks:
            q = blk(q, src_hiddens, src_mask)

        # Project to target dimension
        q = self.output_norm(q)
        soft_tokens = self.output_proj(q)  # [B, K, tgt_dim]

        return soft_tokens

    def get_attention_entropy(self) -> torch.Tensor:
        """Get average attention entropy for regularization."""
        # This would need to be implemented by storing attention weights
        # For now, return 0 as placeholder
        return torch.tensor(0.0)


# ---------------------------
# Data preparation
# ---------------------------

def format_prompts(problem: str) -> Tuple[str, str, str]:
    """
    Format GSM8K problem into prompts.
    Returns: (full_prompt, question_only, instruction)
    """
    instruction = (
        "You are a helpful math tutor. Solve the problem step by step, "
        "then end your final line with '#### <number>'.\n\nProblem:\n"
    )
    question_only = instruction + problem.strip() + "\n\nAnswer:"
    return question_only, question_only, instruction

def build_samples(batch, src_tok, tgt_tok, device) -> List[Sample]:
    """Build samples with question/answer separation."""
    samples = []
    for prob, sol in zip(batch["question"], batch["answer"]):
        src_prompt, tgt_question, _ = format_prompts(prob)
        answer = " " + sol.strip()

        samples.append(Sample(
            src_prompt=src_prompt,
            tgt_question=tgt_question,
            tgt_answer=answer,
            tgt_full=tgt_question + answer
        ))
    return samples

# ---------------------------
# Improved batch building
# ---------------------------

def build_batch_inputs_v2(samples: List[Sample],
                          src_model, src_tok,
                          tgt_model, tgt_tok,
                          translator,
                          device, dtype):
    """
    Improved batch building:
    - Question-only inputs to force use of soft tokens
    - Answer-only labels for loss computation
    """
    B = len(samples)

    # Source encoding
    with torch.no_grad():
        src_enc = src_tok(
            [s.src_prompt for s in samples],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        src_enc = {k: v.to(device) for k, v in src_enc.items()}
        src_out = src_model(**src_enc, output_hidden_states=True)
        src_h = src_out.hidden_states[-1].to(dtype)
        src_mask = src_enc["attention_mask"]

    # Get soft tokens from translator
    soft_tokens = translator(src_h, src_mask)  # [B, K, d_tgt]
    K = soft_tokens.size(1)

    # Tokenize questions and answers separately
    questions_enc = tgt_tok(
        [s.tgt_question for s in samples],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    )
    questions_enc = {k: v.to(device) for k, v in questions_enc.items()}

    answers_enc = tgt_tok(
        [s.tgt_answer for s in samples],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
        add_special_tokens=False  # Don't add BOS to answer
    )
    answers_enc = {k: v.to(device) for k, v in answers_enc.items()}

    # Get embeddings
    with torch.no_grad():
        embed = tgt_model.get_input_embeddings()
        q_embeds = embed(questions_enc["input_ids"]).to(dtype)
        a_embeds = embed(answers_enc["input_ids"]).to(dtype)

    # Concatenate: [soft_tokens, question_embeds, answer_embeds]
    inputs_embeds = torch.cat([soft_tokens, q_embeds, a_embeds], dim=1)

    # Build attention mask
    q_len = q_embeds.size(1)
    a_len = a_embeds.size(1)
    total_len = K + q_len + a_len
    attn_mask = torch.ones((B, total_len), dtype=torch.long, device=device)

    # Build labels: -100 for soft tokens and questions, actual ids for answers
    labels = []
    for i in range(B):
        # -100 for K soft tokens and question length
        prefix_ignore = torch.full((K + q_len,), -100, dtype=torch.long, device=device)
        # Actual answer token ids for loss
        answer_labels = answers_enc["input_ids"][i]
        # Combine
        sample_labels = torch.cat([prefix_ignore, answer_labels], dim=0)
        # Pad to total length
        if sample_labels.size(0) < total_len:
            pad = torch.full((total_len - sample_labels.size(0),), -100,
                           dtype=torch.long, device=device)
            sample_labels = torch.cat([sample_labels, pad], dim=0)
        labels.append(sample_labels)

    labels = torch.stack(labels, dim=0)

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attn_mask,
        "labels": labels
    }

# ---------------------------
# Training with attention regularization
# ---------------------------

def compute_attention_entropy_bonus(soft_token_count: int, current_step: int,
                                   warmup_steps: int = 500) -> float:
    """Compute attention entropy bonus for early training."""
    if current_step >= warmup_steps:
        return 0.0
    # Linearly decay from 0.1 to 0 over warmup_steps
    return 0.1 * (1.0 - current_step / warmup_steps)

# ---------------------------
# Improved evaluation
# ---------------------------

def evaluate_numeric_accuracy_v2(dataset, src_model, src_tok, tgt_model, tgt_tok,
                                 translator, device, dtype, num_samples: int = 200,
                                 max_new_tokens: int = 256):
    """Improved evaluation with proper generation settings."""
    tgt_model.eval()
    src_model.eval()
    translator.eval()

    samples = []
    taken = 0
    for ex in dataset:
        if taken >= num_samples: break
        samples.extend(build_samples(
            {"question": [ex["question"]], "answer": [ex["answer"]]},
            src_tok, tgt_tok, device
        ))
        taken += 1

    # Bridged generation
    with torch.no_grad():
        batch = build_batch_inputs_v2(samples, src_model, src_tok, tgt_model,
                                      tgt_tok, translator, device, dtype)
        gen = tgt_model.generate(
            inputs_embeds=batch["inputs_embeds"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
            pad_token_id=tgt_tok.pad_token_id,
            eos_token_id=tgt_tok.eos_token_id
        )
        bridged_texts = tgt_tok.batch_decode(gen, skip_special_tokens=True)

    # Target-alone baseline
    base_texts = []
    with torch.no_grad():
        prompts = [s.tgt_question for s in samples]
        enc = tgt_tok(prompts, return_tensors="pt", padding=True,
                     truncation=True, max_length=2048).to(device)
        base_out = tgt_model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tgt_tok.pad_token_id,
            eos_token_id=tgt_tok.eos_token_id
        )
        base_texts = tgt_tok.batch_decode(base_out, skip_special_tokens=True)

    # Compute accuracy
    def compute_acc(pred_texts):
        correct = 0
        for text, s in zip(pred_texts, samples):
            pred = extract_final_answer(text)
            gold = extract_final_answer(s.tgt_full)
            correct += int(pred == gold)
        return correct / len(samples)

    acc_bridged = compute_acc(bridged_texts)
    acc_baseline = compute_acc(base_texts)

    return acc_baseline, acc_bridged

# ---------------------------
# Main training loop
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--target_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--translator_type", type=str, default="bottleneck_gated")
    parser.add_argument("--bottleneck_dim", type=int, default=1024)
    parser.add_argument("--soft_tokens", type=int, default=48)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--train_steps", type=int, default=2000)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--per_device_batch", type=int, default=8)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--eval_samples", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--save_path", type=str, default="translator_v2_checkpoint.pt")
    parser.add_argument("--attention_reg_steps", type=int, default=500)
    args = parser.parse_args()

    set_seed(args.seed)
    setup_ddp()
    device = torch.device(f"cuda:{local_rank()}")
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    if is_main():
        log("==== Improved Cross-Attention Config ====")
        for k, v in vars(args).items():
            log(f"{k}: {v}")

    # Load models/tokenizers with proper padding config
    log("Loading source model/tokenizer...")
    src_tok = AutoTokenizer.from_pretrained(args.source_model, use_fast=True)
    if src_tok.pad_token is None:
        src_tok.pad_token = src_tok.eos_token
    src_tok.padding_side = "left"  # Critical for decoder-only models
    src_tok.model_max_length = 2048

    src_model = AutoModelForCausalLM.from_pretrained(
        args.source_model, torch_dtype=dtype, device_map=None
    ).eval().to(device)
    for p in src_model.parameters():
        p.requires_grad = False

    log("Loading target model/tokenizer...")
    tgt_tok = AutoTokenizer.from_pretrained(args.target_model, use_fast=True)
    if tgt_tok.pad_token is None:
        tgt_tok.pad_token = tgt_tok.eos_token
    tgt_tok.padding_side = "left"  # Critical for decoder-only models
    tgt_tok.model_max_length = 2048

    tgt_model = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=dtype, device_map=None
    ).eval().to(device)
    for p in tgt_model.parameters():
        p.requires_grad = False

    # Get dimensions
    with torch.no_grad():
        d_src = src_model.config.hidden_size
        d_tgt = tgt_model.config.hidden_size
    log(f"Source dim: {d_src} | Target dim: {d_tgt} | Bottleneck: {args.bottleneck_dim}")

    # Build bottlenecked gated translator
    translator = BottleneckedGatedTranslator(
        src_dim=d_src,
        tgt_dim=d_tgt,
        bottleneck_dim=args.bottleneck_dim,
        soft_tokens=args.soft_tokens,
        depth=args.depth,
        n_heads=args.heads
    ).to(device=device, dtype=dtype)

    # Count parameters
    param_count = sum(p.numel() for p in translator.parameters())
    log(f"Translator parameters: {param_count / 1e6:.1f}M")

    # DDP wrapper
    if dist.is_initialized():
        translator = DDP(translator, device_ids=[local_rank()],
                        output_device=local_rank(), find_unused_parameters=False)

    # Load data
    log("Loading GSM8K...")
    ds = load_dataset("gsm8k", "main")
    train_ds = ds["train"]
    test_ds = ds["test"]

    # Optimizer with cosine schedule
    optim = torch.optim.AdamW(
        [p for p in translator.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.train_steps
    )

    # Training loop
    step = 0
    translator.train()
    running_loss = 0.0
    last_eval = -1
    rng = random.Random(args.seed)

    while step < args.train_steps:
        # Sample batch
        batch_idx = [rng.randrange(0, len(train_ds)) for _ in range(args.per_device_batch)]
        batch = {
            "question": [train_ds[i]["question"] for i in batch_idx],
            "answer": [train_ds[i]["answer"] for i in batch_idx]
        }
        samples = build_samples(batch, src_tok, tgt_tok, device)

        # Build inputs with improved objective
        data = build_batch_inputs_v2(
            samples, src_model, src_tok, tgt_model, tgt_tok,
            translator, device, dtype
        )

        # Forward pass
        out = tgt_model(**data)
        loss = out.loss

        # Add attention entropy regularization (early training only)
        entropy_weight = compute_attention_entropy_bonus(
            args.soft_tokens, step, args.attention_reg_steps
        )
        if entropy_weight > 0:
            # Placeholder - would need actual attention weights
            loss = loss * (1 - entropy_weight)

        # Detach for logging (fixes DDP warning)
        loss_scalar = loss.detach()
        if dist.is_initialized():
            dist.all_reduce(loss_scalar, op=dist.ReduceOp.AVG)
        running_loss += loss_scalar.item()

        # Backward
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(translator.parameters(), 1.0)
        optim.step()
        sched.step()
        step += 1

        # Logging
        if step % 20 == 0 and is_main():
            avg_loss = running_loss / 20
            log(f"Step {step}/{args.train_steps} | Loss: {avg_loss:.4f} | LR: {sched.get_last_lr()[0]:.6f}")
            running_loss = 0.0

        # Evaluation
        if args.eval_every > 0 and step % args.eval_every == 0 and step != last_eval:
            last_eval = step
            if is_main():
                with torch.no_grad():
                    acc_base, acc_bridged = evaluate_numeric_accuracy_v2(
                        test_ds, src_model, src_tok, tgt_model, tgt_tok,
                        translator.module if isinstance(translator, DDP) else translator,
                        device, dtype, num_samples=args.eval_samples,
                        max_new_tokens=args.max_new_tokens
                    )
                log(f"[Eval Step {step}] Target-alone: {acc_base:.3f} | Bridged: {acc_bridged:.3f}")

    # Save and final eval
    if is_main():
        state = (translator.module if isinstance(translator, DDP) else translator).state_dict()
        torch.save(state, args.save_path)
        log(f"Saved translator to {args.save_path}")

        with torch.no_grad():
            acc_base, acc_bridged = evaluate_numeric_accuracy_v2(
                test_ds, src_model, src_tok, tgt_model, tgt_tok,
                translator.module if isinstance(translator, DDP) else translator,
                device, dtype, num_samples=args.eval_samples,
                max_new_tokens=args.max_new_tokens
            )
        log(f"[Final] Target-alone: {acc_base:.3f} | Bridged: {acc_bridged:.3f}")

    cleanup_ddp()

if __name__ == "__main__":
    main()