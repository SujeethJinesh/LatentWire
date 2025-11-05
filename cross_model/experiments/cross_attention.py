#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inter-LLM "interlingua" prototype:
- Source LLM A → (get hidden states) → Translator → K soft tokens in target space
- Target LLM B consumes [soft tokens || B's own embeddings] and generates answers.
- Train only the translator with teacher-forced LM loss on GSM8K.
- Evaluate numeric exact-match accuracy on GSM8K.

Run (4x H100, DDP):
  torchrun --nproc_per_node=4 interlingua_bridge.py \
    --source_model Qwen/Qwen2.5-1.5B-Instruct \
    --target_model meta-llama/Llama-3.2-1B-Instruct \
    --translator_type cross_attn \
    --soft_tokens 32 \
    --train_steps 2000 \
    --per_device_batch 2 \
    --eval_samples 200 \
    --bf16

Notes:
- Defaults pick small-ish models; swap to your pair as desired.
- Both models are frozen; only the translator trains (few million params).
- Script uses BF16 on Ampere/Hopper and DDP across 4 GPUs.
"""

import os, math, re, json, random, argparse, time
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
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

def to_dtype_device(x, dtype, device):
    return x.to(device=device, dtype=dtype)

def extract_final_answer(text: str) -> str:
    """
    GSM8K solutions often end with '#### <number>'.
    If not found, fallback to last integer in the string.
    """
    m = re.search(r"####\s*(-?\d+)", text)
    if m: 
        return m.group(1)
    # fallback: last integer
    ints = re.findall(r"-?\d+", text)
    return ints[-1] if ints else ""

@dataclass
class Sample:
    src_prompt: str
    tgt_prompt: str
    tgt_full: str
    tgt_prompt_len_tokens: int
    label_ids: torch.Tensor  # full sequence labels (w/ -100 before answer)

# ---------------------------
# Translator modules
# ---------------------------

class LinearTranslator(nn.Module):
    """
    Maps pooled source hidden states (mean-pool over tokens, last layer)
    to K soft tokens in target embedding space via MLP.
    """
    def __init__(self, src_dim: int, tgt_dim: int, soft_tokens: int = 32, hidden: int = 2048):
        super().__init__()
        self.soft_tokens = soft_tokens
        self.mlp = nn.Sequential(
            nn.Linear(src_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, soft_tokens * tgt_dim)
        )
        self.tgt_dim = tgt_dim

    def forward(self, src_hiddens: torch.Tensor) -> torch.Tensor:
        """
        src_hiddens: [B, T_src, d_src] (last layer states of source)
        Returns soft_tokens: [B, K, d_tgt]
        """
        pooled = src_hiddens.mean(dim=1)  # [B, d_src]
        out = self.mlp(pooled)            # [B, K*d_tgt]
        return out.view(src_hiddens.size(0), self.soft_tokens, self.tgt_dim)


class CrossAttentionBlock(nn.Module):
    """
    One block of (self-attn on queries) + (cross-attn from queries to src) + FFN, with RMS norms.
    """
    def __init__(self, d_q: int, d_kv: int, n_heads: int, ffn_mult: int = 4):
        super().__init__()
        self.q_norm = nn.LayerNorm(d_q, elementwise_affine=True)
        self.src_norm = nn.LayerNorm(d_kv, elementwise_affine=True)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_q, num_heads=n_heads, batch_first=True)
        self.cross_q_proj = nn.Linear(d_q, d_q)
        self.cross_k_proj = nn.Linear(d_kv, d_q)
        self.cross_v_proj = nn.Linear(d_kv, d_q)
        self.cross_out = nn.Linear(d_q, d_q)
        self.ffn = nn.Sequential(
            nn.Linear(d_q, ffn_mult*d_q),
            nn.GELU(),
            nn.Linear(ffn_mult*d_q, d_q)
        )

    def forward(self, q_tokens: torch.Tensor, src_seq: torch.Tensor, src_mask: torch.Tensor = None):
        # q_tokens: [B, K, d_q], src_seq: [B, T, d_kv]
        B, K, d_q = q_tokens.shape
        # Self-attention on queries
        q_norm = self.q_norm(q_tokens)
        sa_out, _ = self.self_attn(q_norm, q_norm, q_norm, need_weights=False)
        q_tokens = q_tokens + sa_out

        # Cross-attention: queries attend to source sequence
        qn = self.q_norm(q_tokens)
        srcn = self.src_norm(src_seq)
        Q = self.cross_q_proj(qn)
        K_ = self.cross_k_proj(srcn)
        V_ = self.cross_v_proj(srcn)

        # Compute scaled dot-product attention manually to support mask
        attn_logits = torch.matmul(Q, K_.transpose(1, 2)) / math.sqrt(Q.size(-1))  # [B, K, T]
        if src_mask is not None:
            mask = src_mask[:, None, :].to(dtype=attn_logits.dtype)  # [B,1,T]
            attn_logits = attn_logits.masked_fill(mask == 0, -1e4)
        attn_weights = torch.softmax(attn_logits, dim=-1)           # [B, K, T]
        cross = torch.matmul(attn_weights, V_)                       # [B, K, d_q]
        cross = self.cross_out(cross)
        q_tokens = q_tokens + cross

        # FFN
        ff = self.ffn(self.q_norm(q_tokens))
        q_tokens = q_tokens + ff
        return q_tokens


class CrossAttnResamplerTranslator(nn.Module):
    """
    Learn K target-space 'query tokens' which (after some depth) cross-attend to the source sequence
    and become the soft tokens passed into the target LLM (akin to BLIP-2/Perceiver-resampler style).
    """
    def __init__(self, src_dim: int, tgt_dim: int, soft_tokens: int = 32, depth: int = 2, n_heads: int = 8, ffn_mult: int = 4):
        super().__init__()
        self.K = soft_tokens
        self.query_tokens = nn.Parameter(torch.randn(1, soft_tokens, tgt_dim) / math.sqrt(tgt_dim))
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(d_q=tgt_dim, d_kv=src_dim, n_heads=n_heads, ffn_mult=ffn_mult)
            for _ in range(depth)
        ])

    def forward(self, src_hiddens: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        src_hiddens: [B, T_src, d_src]
        returns soft tokens: [B, K, d_tgt]
        """
        B = src_hiddens.size(0)
        q = self.query_tokens.expand(B, -1, -1)  # [B, K, d_tgt]
        for blk in self.blocks:
            q = blk(q, src_hiddens, src_mask)
        return q

# ---------------------------
# Data prep (GSM8K)
# ---------------------------

def format_prompts(problem: str) -> Tuple[str, str]:
    """
    Build comparable prompts for source and target.
    Keep it simple and consistent so we don't inject extra prompt mismatch.
    """
    base = (
        "You are a helpful math tutor. Solve the problem step by step, "
        "then end your final line with '#### <number>'.\n\nProblem:\n"
    )
    prompt = base + problem.strip() + "\n\nAnswer:"
    return prompt, prompt  # use same text for A and B

def build_samples(batch, src_tok, tgt_tok, device, tgt_model, answer_as_label=True) -> List[Sample]:
    """
    Prepare per-example structures for collating later. We tokenize here for the target to slice label regions.
    """
    samples: List[Sample] = []
    for prob, sol in zip(batch["question"], batch["answer"]):
        src_prompt, tgt_prompt = format_prompts(prob)
        # Full target text for supervised LM loss
        tgt_full = tgt_prompt + " " + sol.strip()

        # Token counts for label masking
        with torch.no_grad():
            tgt_prompt_ids = tgt_tok(tgt_prompt, add_special_tokens=True).input_ids
            tgt_full_ids = tgt_tok(tgt_full, add_special_tokens=True).input_ids

        prompt_len = len(tgt_prompt_ids)
        labels = torch.tensor(tgt_full_ids, dtype=torch.long)
        # mask everything up to the last token of the prompt
        labels[:prompt_len] = -100

        samples.append(Sample(
            src_prompt=src_prompt,
            tgt_prompt=tgt_prompt,
            tgt_full=tgt_full,
            tgt_prompt_len_tokens=prompt_len,
            label_ids=labels
        ))
    return samples

# ---------------------------
# Collation building inputs_embeds with soft tokens
# ---------------------------

def build_batch_inputs(samples: List[Sample],
                       src_model, src_tok,
                       tgt_model, tgt_tok,
                       translator,
                       device, dtype):
    """
    For each sample:
      - Encode source prompt with A; get last-layer hidden states (no grad)
      - Translator -> K soft tokens in B's space
      - Tokenize tgt_full with B; embed its input ids
      - Concatenate soft tokens + tgt embeddings; build attention mask and labels
    Returns: dict ready for target forward
    """
    B = len(samples)

    # Source encoding (no grad)
    with torch.no_grad():
        src_enc = src_tok([s.src_prompt for s in samples], return_tensors="pt", padding=True, truncation=True, max_length=2048)
        src_enc = {k: v.to(device) for k, v in src_enc.items()}
        src_out = src_model(**src_enc, output_hidden_states=True)
        # last hidden states: [B, T_src, d_src]
        src_h = src_out.hidden_states[-1].to(dtype)

        # Source mask
        src_mask = src_enc["attention_mask"]

    # Translator (grad flows)
    soft_tokens = translator(src_h, src_mask) if isinstance(translator, CrossAttnResamplerTranslator) \
                  else translator(src_h)  # [B, K, d_tgt]

    # Target tokenization
    tgt_batch = tgt_tok([s.tgt_full for s in samples], return_tensors="pt", padding=True, truncation=True, max_length=2048)
    tgt_batch = {k: v.to(device) for k, v in tgt_batch.items()}

    # Target input embeddings
    with torch.no_grad():
        embed = tgt_model.get_input_embeddings()  # nn.Embedding
        tgt_embeds = embed(tgt_batch["input_ids"]).to(dtype)  # [B, T_tgt, d_tgt]

    # Concatenate soft tokens in front
    K = soft_tokens.size(1)
    max_T = tgt_embeds.size(1)
    inputs_embeds = torch.cat([soft_tokens, tgt_embeds], dim=1)  # [B, K+T, d]

    # Attention mask (1s for all positions we pass)
    attn_mask = torch.ones((B, K + max_T), dtype=torch.long, device=device)
    # Labels — need to left-pad with -100 for K soft tokens (no loss on them)
    # Also preserve the per-example masked prompt region already set in label_ids
    labels_list = []
    for i, s in enumerate(samples):
        labels = s.label_ids.to(device)
        # Pad labels to max_T
        if labels.size(0) < max_T:
            pad = torch.full((max_T - labels.size(0),), -100, dtype=labels.dtype, device=device)
            labels = torch.cat([labels, pad], dim=0)
        # Prepend -100 for soft tokens
        labels = torch.cat([torch.full((K,), -100, dtype=labels.dtype, device=device), labels], dim=0)
        labels_list.append(labels)
    labels = torch.stack(labels_list, dim=0)  # [B, K+T]

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attn_mask,
        "labels": labels
    }

# ---------------------------
# Training / Evaluation loops
# ---------------------------

def evaluate_numeric_accuracy(dataset, src_model, src_tok, tgt_model, tgt_tok, translator,
                              device, dtype, num_samples: int = 200, max_new_tokens: int = 256):
    """
    Compare (i) Target alone vs (ii) Source→Translator→Target.
    """
    tgt_model.eval()
    src_model.eval()
    translator.eval()

    samples = []
    taken = 0
    for ex in dataset:
        if taken >= num_samples: break
        samples.extend(build_samples({"question":[ex["question"]], "answer":[ex["answer"]]},
                                     src_tok, tgt_tok, device, tgt_model))
        taken += 1

    # Build bridged inputs (uses inputs_embeds)
    with torch.no_grad():
        batch = build_batch_inputs(samples, src_model, src_tok, tgt_model, tgt_tok, translator, device, dtype)
        gen = tgt_model.generate(
            inputs_embeds=batch["inputs_embeds"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tgt_tok.eos_token_id
        )
        # Strip off the K soft tokens from positions when decoding
        # We can't directly decode inputs_embeds, so decode the generated tail.
        # Take only newly generated tokens:
        # Transformers returns full sequence when using inputs_embeds. We approximate by decoding all and taking answer lines.
        bridged_texts = tgt_tok.batch_decode(gen, skip_special_tokens=True)

    # Target-alone baseline
    base_texts = []
    with torch.no_grad():
        prompts = [s.tgt_prompt for s in samples]
        enc = tgt_tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        base_out = tgt_model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      eos_token_id=tgt_tok.eos_token_id)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--target_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--translator_type", type=str, choices=["cross_attn", "linear"], default="cross_attn")
    parser.add_argument("--soft_tokens", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--train_steps", type=int, default=2000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--per_device_batch", type=int, default=2)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--eval_samples", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--save_path", type=str, default="translator_ckpt.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    setup_ddp()
    device = torch.device(f"cuda:{local_rank()}")
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    if is_main():
        log("==== Config ====")
        for k,v in vars(args).items(): log(f"{k}: {v}")

    # Load models/tokenizers (frozen)
    log("Loading source model/tokenizer...")
    src_tok = AutoTokenizer.from_pretrained(args.source_model, use_fast=True)
    if src_tok.pad_token is None:
        src_tok.pad_token = src_tok.eos_token
    src_model = AutoModelForCausalLM.from_pretrained(
        args.source_model, torch_dtype=dtype, device_map=None
    ).eval().to(device)
    for p in src_model.parameters(): p.requires_grad = False

    log("Loading target model/tokenizer...")
    tgt_tok = AutoTokenizer.from_pretrained(args.target_model, use_fast=True)
    if tgt_tok.pad_token is None:
        tgt_tok.pad_token = tgt_tok.eos_token
    tgt_model = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=dtype, device_map=None
    ).eval().to(device)
    for p in tgt_model.parameters(): p.requires_grad = False

    # Infer dims
    with torch.no_grad():
        d_src = src_model.config.hidden_size
        d_tgt = tgt_model.config.hidden_size
    log(f"Source hidden dim: {d_src} | Target hidden dim: {d_tgt}")

    # Build translator
    if args.translator_type == "cross_attn":
        translator = CrossAttnResamplerTranslator(
            src_dim=d_src, tgt_dim=d_tgt, soft_tokens=args.soft_tokens, depth=args.depth, n_heads=args.heads
        ).to(device=device, dtype=dtype)
    else:
        translator = LinearTranslator(
            src_dim=d_src, tgt_dim=d_tgt, soft_tokens=args.soft_tokens, hidden=4*d_tgt
        ).to(device=device, dtype=dtype)

    # DDP only on translator (models are frozen)
    if dist.is_initialized():
        translator = DDP(translator, device_ids=[local_rank()], output_device=local_rank(), find_unused_parameters=False)

    # Data
    log("Loading GSM8K...")
    ds = load_dataset("gsm8k", "main")
    train_ds = ds["train"]
    test_ds  = ds["test"]

    # A very simple random sampler over training set
    rng = random.Random(args.seed)

    # Optimizer & sched
    optim = torch.optim.AdamW([p for p in translator.parameters() if p.requires_grad],
                              lr=args.lr, weight_decay=args.weight_decay)
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=args.warmup_steps,
                                            num_training_steps=args.train_steps)

    # --------- Train loop ---------
    step = 0
    translator.train()
    running = 0.0
    last_eval = -1

    while step < args.train_steps:
        # sample a batch
        batch_idx = [rng.randrange(0, len(train_ds)) for _ in range(args.per_device_batch)]
        batch = {"question": [train_ds[i]["question"] for i in batch_idx],
                 "answer":   [train_ds[i]["answer"]   for i in batch_idx]}
        samples = build_samples(batch, src_tok, tgt_tok, device, tgt_model)

        data = build_batch_inputs(samples, src_model, src_tok, tgt_model, tgt_tok,
                                  translator, device, dtype)

        # Forward (compute loss on target)
        out = tgt_model(**data)
        loss = out.loss
        if dist.is_initialized():
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        running += loss.item()

        # Backward into translator only
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        sched.step()
        step += 1

        if step % 20 == 0 and is_main():
            log(f"Step {step}/{args.train_steps} | Loss (avg over last 20): {running/20:.4f}")
            running = 0.0

        # Periodic eval
        if args.eval_every > 0 and step % args.eval_every == 0 and step != last_eval:
            last_eval = step
            if is_main():
                with torch.no_grad():
                    acc_base, acc_bridged = evaluate_numeric_accuracy(
                        test_ds, src_model, src_tok, tgt_model, tgt_tok, translator.module if isinstance(translator, DDP) else translator,
                        device, dtype, num_samples=args.eval_samples, max_new_tokens=args.max_new_tokens
                    )
                log(f"[Eval] Step {step} | Target-alone acc: {acc_base:.3f} | Bridged acc: {acc_bridged:.3f}")

    # Final save (translator weights)
    if is_main():
        state = (translator.module if isinstance(translator, DDP) else translator).state_dict()
        torch.save(state, args.save_path)
        log(f"Saved translator to {args.save_path}")

        # Final eval
        with torch.no_grad():
            acc_base, acc_bridged = evaluate_numeric_accuracy(
                test_ds, src_model, src_tok, tgt_model, tgt_tok, translator.module if isinstance(translator, DDP) else translator,
                device, dtype, num_samples=args.eval_samples, max_new_tokens=args.max_new_tokens
            )
        log(f"[Final Eval] Target-alone acc: {acc_base:.3f} | Bridged acc: {acc_bridged:.3f}")

    cleanup_ddp()

if __name__ == "__main__":
    main()
