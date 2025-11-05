#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DiT Interlingua: map Model-A's representation of a question into Model-B's
*target prompt embeddings* for the same question, then let B generate the answer.

- Source: mistralai/Mistral-7B-Instruct-v0.3 (default)
- Target: meta-llama/Meta-Llama-3.1-8B-Instruct (default)
- Translator: Perceiver-style resampler -> Diffusion Transformer (DiT) in bottleneck dim b
- Losses: diffusion (epsilon-pred), + optional LM loss on answer tokens using B with predicted prompt embeddings
- Eval: GSM8K numeric exact match for target-alone vs DiT-bridged

Run (4× H100):
  torchrun --nproc_per_node=4 dit_interlingua.py \
      --source_model mistralai/Mistral-7B-Instruct-v0.3 \
      --target_model meta-llama/Meta-Llama-3.1-8B-Instruct \
      --bf16

Dependencies:
  pip install torch transformers datasets

Notes:
  * We set pad_token and left-padding for decoder-only models.
  * We freeze both LLMs; only the translator trains.
"""

import os, math, argparse, random, re, time
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, get_linear_schedule_with_warmup

# -------------------------
# DDP helpers
# -------------------------
def is_dist(): return "RANK" in os.environ and "WORLD_SIZE" in os.environ
def setup_ddp():
    if is_dist():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
def cleanup_ddp():
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()
def is_main(): return (not is_dist()) or dist.get_rank() == 0
def log(*a, **k):
    if is_main(): print(*a, **k, flush=True)
def world_size(): return dist.get_world_size() if is_dist() else 1
def local_rank(): return int(os.environ.get("LOCAL_RANK", "0"))

# -------------------------
# Small utils
# -------------------------
def extract_final_answer(text: str) -> str:
    m = re.search(r"####\s*(-?\d+)", text)
    if m: return m.group(1)
    ints = re.findall(r"-?\d+", text)
    return ints[-1] if ints else ""

def format_prompt(question: str) -> str:
    base = (
        "You are a helpful math tutor. Solve the problem step by step, "
        "then end your final line with '#### <number>'.\n\nProblem:\n"
    )
    return base + question.strip() + "\n\nAnswer:"

# -------------------------
# Token & schedule helpers
# -------------------------
def ensure_leftpad_with_pad(tokenizer, max_len):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = max_len

def cosine_alpha_bar(t):
    # t in [0,1] (continuous), return alpha_bar(t)
    return torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

def timestep_embedding(timesteps, dim, max_period=10000):
    # Classic sinusoidal time embedding (like in DiT)
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb

# -------------------------
# Modules: Resampler & DiT
# -------------------------
class Resampler(nn.Module):
    """
    Perceiver/Flamingo-style resampler: learned query tokens cross-attend to source features
    to produce S conditioning tokens at width b.
    """
    def __init__(self, src_dim, b, S=64, depth=2, heads=8):
        super().__init__()
        self.S = S
        self.query = nn.Parameter(torch.randn(1, S, b) / math.sqrt(b))
        self.src_proj = nn.Linear(src_dim, b)
        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict(dict(
                ln_q=nn.LayerNorm(b),
                ln_kv=nn.LayerNorm(b),
                self_attn=nn.MultiheadAttention(b, heads, batch_first=True),
                cross_q=nn.Linear(b, b),
                cross_k=nn.Linear(b, b),
                cross_v=nn.Linear(b, b),
                cross_o=nn.Linear(b, b),
                ffn=nn.Sequential(nn.Linear(b, 4*b), nn.GELU(), nn.Linear(4*b, b)),
            )))
        self.gate = nn.Parameter(torch.zeros(1))  # gated cross-attn residual

    def forward(self, src_h, src_mask=None):
        """
        src_h: [B, T_src, src_dim]
        returns cond: [B, S, b]
        """
        B = src_h.size(0)
        x = self.query.expand(B, -1, -1)                 # [B, S, b]
        kv = self.src_proj(src_h)                        # [B, T_src, b]
        for blk in self.blocks:
            # self-attention on queries
            xn = blk["ln_q"](x)
            sa_out, _ = blk["self_attn"](xn, xn, xn, need_weights=False)
            x = x + sa_out
            # cross-attend to source
            q = blk["cross_q"](blk["ln_q"](x))
            k = blk["cross_k"](blk["ln_kv"](kv))
            v = blk["cross_v"](blk["ln_kv"](kv))
            attn = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.size(-1))
            if src_mask is not None:
                mask = src_mask[:, None, :].to(dtype=attn.dtype)  # [B,1,T]
                attn = attn.masked_fill(mask == 0, -1e4)
            attn = torch.softmax(attn, dim=-1)
            cross = torch.matmul(attn, v)
            x = x + torch.tanh(self.gate) * blk["cross_o"](cross)
            # ffn
            x = x + blk["ffn"](blk["ln_q"](x))
        return x

class DiTBlock(nn.Module):
    def __init__(self, b, heads=16, ffn_mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(b)
        self.sa  = nn.MultiheadAttention(b, heads, batch_first=True)
        self.ln2x = nn.LayerNorm(b)
        self.ln2c = nn.LayerNorm(b)
        self.cross_q = nn.Linear(b, b)
        self.cross_k = nn.Linear(b, b)
        self.cross_v = nn.Linear(b, b)
        self.cross_o = nn.Linear(b, b)
        self.cross_gate = nn.Parameter(torch.zeros(1))
        self.ln3 = nn.LayerNorm(b)
        self.ffn = nn.Sequential(nn.Linear(b, ffn_mult*b), nn.GELU(), nn.Linear(ffn_mult*b, b))

    def forward(self, x, cond, cond_mask=None):
        # Self-attn on x
        xs = self.ln1(x)
        sa, _ = self.sa(xs, xs, xs, need_weights=False)
        x = x + sa
        # Cross-attn from x -> cond
        q = self.cross_q(self.ln2x(x))
        k = self.cross_k(self.ln2c(cond))
        v = self.cross_v(self.ln2c(cond))
        attn = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.size(-1))
        if cond_mask is not None:
            mask = cond_mask[:, None, :].to(dtype=attn.dtype)
            attn = attn.masked_fill(mask == 0, -1e4)
        attn = torch.softmax(attn, dim=-1)
        cross = torch.matmul(attn, v)
        x = x + torch.tanh(self.cross_gate) * self.cross_o(cross)
        # FFN
        x = x + self.ffn(self.ln3(x))
        return x

class DiTTranslator(nn.Module):
    """
    Diffusion Transformer that predicts epsilon over target prompt embeddings (in bottleneck dim b).
    """
    def __init__(self, b=1024, depth=6, heads=16, src_dim=4096, tgt_dim=4096, S=64):
        super().__init__()
        self.b = b
        self.down_tgt = nn.Linear(tgt_dim, b)       # d_tgt -> b
        self.up_tgt   = nn.Linear(b, tgt_dim)       # b -> d_tgt
        self.t_embed  = nn.Sequential(nn.Linear(b, b*4), nn.GELU(), nn.Linear(b*4, b))
        self.resampler = Resampler(src_dim=src_dim, b=b, S=S, depth=2, heads=8)
        self.blocks = nn.ModuleList([DiTBlock(b=b, heads=heads, ffn_mult=4) for _ in range(depth)])
        self.ln_out = nn.LayerNorm(b)
        self.out = nn.Linear(b, b)  # predict epsilon in b-dim

    def forward(self, src_h, src_mask, x_t, x_mask, t_scalar):
        """
        src_h:   [B, T_src, src_dim]
        x_t:     [B, T_tgt, d_tgt] (noised target embeddings in d_tgt)
        x_mask:  [B, T_tgt]
        t_scalar: [B] in [0,1]
        """
        # Resample source to cond tokens in b
        cond = self.resampler(src_h, src_mask)     # [B, S, b]
        cond_mask = torch.ones(cond.size()[:2], dtype=torch.long, device=cond.device)

        # Project x_t to bottleneck
        x = self.down_tgt(x_t)                     # [B, T, b]

        # Time embedding (FiLM-ish: add to tokens)
        t_emb = timestep_embedding(t_scalar, self.b).to(x.dtype)  # [B, b]
        t_emb = self.t_embed(t_emb)                               # [B, b]
        x = x + t_emb[:, None, :]

        for blk in self.blocks:
            x = blk(x, cond, cond_mask)

        eps_b = self.out(self.ln_out(x))           # [B, T, b]
        return eps_b

    # Utility helpers for caller
    def x0_from_pred(self, x_t, eps_b, alpha_bar_t):
        # x0_hat = (x_t - sqrt(1-alpha_bar_t)*eps_pred_b_up) / sqrt(alpha_bar_t)
        sqrt_ab = alpha_bar_t.sqrt()[:, None, None]
        sqrt_om = (1 - alpha_bar_t).sqrt()[:, None, None]
        eps_up = self.up_tgt(eps_b)
        return (x_t - sqrt_om * eps_up) / (sqrt_ab + 1e-8)

# -------------------------
# Data
# -------------------------
@dataclass
class Sample:
    src_prompt: str
    tgt_prompt_ids: torch.Tensor   # [T_prompt]
    tgt_prompt_mask: torch.Tensor  # [T_prompt]
    tgt_answer_ids: torch.Tensor   # [T_ans]
    tgt_answer_mask: torch.Tensor  # [T_ans] (all ones)

def build_samples(batch, src_tok, tgt_tok, max_prompt_len, device) -> List[Sample]:
    samples = []
    for q, a in zip(batch["question"], batch["answer"]):
        p = format_prompt(q)
        # Target prompt & answer tokenization (left-pad/truncate prompt to fixed len)
        tgt_p = tgt_tok(p, add_special_tokens=True, padding="max_length",
                        truncation=True, max_length=max_prompt_len, return_tensors="pt")
        ans_ids = tgt_tok(" " + a.strip(), add_special_tokens=False, return_tensors="pt")
        samples.append(Sample(
            src_prompt=p,
            tgt_prompt_ids=tgt_p.input_ids[0].to(device),
            tgt_prompt_mask=tgt_p.attention_mask[0].to(device),
            tgt_answer_ids=ans_ids.input_ids[0].to(device),
            tgt_answer_mask=torch.ones_like(ans_ids.input_ids[0]).to(device)
        ))
    return samples

# -------------------------
# Losses and schedules
# -------------------------
def diffusion_training_step(translator, src_model, src_tok, tgt_model, tgt_tok,
                            samples: List[Sample], device, dtype,
                            lm_aux_weight: float,
                            t_min=1e-4, t_max=0.999):
    B = len(samples)
    # 1) Source features (last hidden layer)
    with torch.no_grad():
        enc = src_tok([s.src_prompt for s in samples], return_tensors="pt",
                      padding=True, truncation=True).to(device)
        src_out = src_model(**enc, output_hidden_states=True)
        src_h = src_out.hidden_states[-1].to(dtype)         # [B, T_src, d_src]
        src_mask = enc["attention_mask"]

    # 2) Target prompt embeddings + masks
    with torch.no_grad():
        embed = tgt_model.get_input_embeddings()
        tgt_prompt_ids = torch.stack([s.tgt_prompt_ids for s in samples], dim=0)
        tgt_prompt_mask = torch.stack([s.tgt_prompt_mask for s in samples], dim=0)
        x0 = embed(tgt_prompt_ids).to(dtype)                 # [B, T, d_tgt]

    # 3) Sample diffusion time and noise
    t = torch.rand(B, device=device) * (t_max - t_min) + t_min            # (0,1)
    alpha_bar_t = cosine_alpha_bar(t).to(device).to(dtype)                # [B]
    noise = torch.randn_like(x0)
    x_t = alpha_bar_t.sqrt()[:, None, None] * x0 + (1 - alpha_bar_t).sqrt()[:, None, None] * noise

    # 4) Predict epsilon in bottleneck space
    eps_b = translator(src_h, src_mask, x_t, tgt_prompt_mask, t)          # [B, T, b]
    # Project to d_tgt for epsilon target
    eps_pred = translator.up_tgt(eps_b)                                   # [B, T, d_tgt]

    # 5) Diffusion loss over real tokens only
    mask = tgt_prompt_mask[:, :, None].to(dtype)
    diff_loss = ((eps_pred - noise) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)

    # 6) Optional LM auxiliary loss (answer-only)
    lm_loss = torch.tensor(0.0, device=device, dtype=x0.dtype)
    if lm_aux_weight > 0:
        with torch.no_grad():
            x0_hat = translator.x0_from_pred(x_t, eps_b, alpha_bar_t)     # [B, T, d_tgt]
        # Build inputs_embeds = [x0_hat || answer_embeds]
        ans_ids = [s.tgt_answer_ids for s in samples]
        ans_lens = [ids.size(0) for ids in ans_ids]
        max_ans = max(ans_lens)
        ans_pad = tgt_tok.pad_token_id
        ans_ids_pad = torch.full((B, max_ans), ans_pad, device=device, dtype=torch.long)
        for i, ids in enumerate(ans_ids):
            ans_ids_pad[i, :ids.size(0)] = ids
        with torch.no_grad():
            ans_emb = tgt_model.get_input_embeddings()(ans_ids_pad).to(dtype)
        inputs_embeds = torch.cat([x0_hat, ans_emb], dim=1)                # [B, T+T_ans, d]
        attn_mask = torch.cat([tgt_prompt_mask, torch.ones(B, max_ans, device=device, dtype=torch.long)], dim=1)
        # Labels: -100 for prompt, ids for answer
        labels = torch.full((B, x0_hat.size(1) + max_ans), -100, device=device, dtype=torch.long)
        for i, ids in enumerate(ans_ids):
            labels[i, x0_hat.size(1): x0_hat.size(1) + ids.size(0)] = ids
        out = tgt_model(inputs_embeds=inputs_embeds, attention_mask=attn_mask, labels=labels)
        lm_loss = out.loss

    total = diff_loss + lm_aux_weight * lm_loss
    return total, diff_loss.detach(), lm_loss.detach()

@torch.no_grad()
def sample_prompt_embeddings(translator, src_model, src_tok, tgt_model, tgt_tok,
                             prompts: List[str], T_prompt: int, device, dtype,
                             steps=25):
    """
    DDIM-like deterministic sampling of target prompt embeddings given source prompts.
    """
    B = len(prompts)
    enc = src_tok(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    src_out = src_model(**enc, output_hidden_states=True)
    src_h = src_out.hidden_states[-1].to(dtype)
    src_mask = enc["attention_mask"]

    # Prepare shapes/masks for the *target* prompt
    tgt = tgt_tok(prompts, add_special_tokens=True, padding="max_length",
                  truncation=True, max_length=T_prompt, return_tensors="pt").to(device)
    mask = tgt.attention_mask
    # Start from Gaussian in d_tgt
    d_tgt = tgt_model.config.hidden_size
    x = torch.randn(B, T_prompt, d_tgt, device=device, dtype=dtype)

    # Deterministic DDIM schedule over s steps
    ts = torch.linspace(0.999, 1e-4, steps, device=device)
    for t in ts:
        t_batch = t.repeat(B)
        ab = cosine_alpha_bar(t_batch).to(dtype)            # [B]
        eps_b = translator(src_h, src_mask, x, mask, t_batch)
        eps = translator.up_tgt(eps_b)
        # x0 estimate
        x0 = (x - (1 - ab).sqrt()[:, None, None] * eps) / (ab.sqrt()[:, None, None] + 1e-8)
        # Move to earlier time (deterministic)
        if t.item() > 1e-4:
            t_prev = (t - (0.999 - 1e-4) / (steps - 1)).clamp(min=1e-4)
            ab_prev = cosine_alpha_bar(t_prev.repeat(B)).to(dtype)
            x = ab_prev.sqrt()[:, None, None] * x0 + (1 - ab_prev).sqrt()[:, None, None] * eps * 0.0
        else:
            x = x0
    return x, mask  # [B, T, d_tgt], [B, T]

# -------------------------
# Evaluation
# -------------------------
@torch.no_grad()
def evaluate_gsm8k(test_ds, src_model, src_tok, tgt_model, tgt_tok, translator,
                   T_prompt, device, dtype, num_samples=200, max_new_tokens=256):
    # Select first num_samples
    rows = [test_ds[i] for i in range(min(num_samples, len(test_ds)))]
    questions = [r["question"] for r in rows]
    answers = [r["answer"] for r in rows]
    prompts = [format_prompt(q) for q in questions]

    # Target-alone baseline
    enc = tgt_tok(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    base_out = tgt_model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                  eos_token_id=tgt_tok.eos_token_id)
    base_texts = tgt_tok.batch_decode(base_out, skip_special_tokens=True)

    # Bridged: sample prompt embeddings with DiT, then generate
    pred_prompt_embeds, mask = sample_prompt_embeddings(
        translator, src_model, src_tok, tgt_model, tgt_tok, prompts, T_prompt, device, dtype, steps=25
    )
    attn_mask = mask
    gen = tgt_model.generate(inputs_embeds=pred_prompt_embeds, attention_mask=attn_mask,
                             max_new_tokens=max_new_tokens, do_sample=False,
                             eos_token_id=tgt_tok.eos_token_id)
    bridged_texts = tgt_tok.batch_decode(gen, skip_special_tokens=True)

    def acc(texts):
        ok = 0
        for txt, gold in zip(texts, answers):
            p = extract_final_answer(txt)
            g = extract_final_answer(gold)
            ok += int(p == g)
        return ok / len(answers)

    return acc(base_texts), acc(bridged_texts)

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--target_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--max_prompt_len", type=int, default=1024)
    ap.add_argument("--per_device_batch", type=int, default=2)
    ap.add_argument("--train_steps", type=int, default=2000)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_samples", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--bf16", action="store_true")
    # Translator sizing
    ap.add_argument("--bottleneck_dim", type=int, default=1024)
    ap.add_argument("--cond_tokens", type=int, default=64)   # resampler S
    ap.add_argument("--dit_depth", type=int, default=6)
    ap.add_argument("--dit_heads", type=int, default=16)
    ap.add_argument("--lm_aux_weight", type=float, default=0.5)
    ap.add_argument("--save_path", type=str, default="runs/dit_interlingua/translator.pt")
    # Source feature type
    ap.add_argument("--src_feature_type", choices=["last_hidden", "embeds"], default="last_hidden")
    args = ap.parse_args()

    set_seed(args.seed)
    setup_ddp()
    device = torch.device(f"cuda:{local_rank()}")
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    if is_main():
        log("==== Config ====")
        for k, v in vars(args).items(): log(f"{k}: {v}")

    # Load models and tokenizers (frozen)
    log("Loading source model/tokenizer...")
    src_tok = AutoTokenizer.from_pretrained(args.source_model, use_fast=True)
    ensure_leftpad_with_pad(src_tok, args.max_prompt_len)
    src_model = AutoModelForCausalLM.from_pretrained(args.source_model, torch_dtype=dtype).eval().to(device)
    for p in src_model.parameters(): p.requires_grad = False

    log("Loading target model/tokenizer...")
    tgt_tok = AutoTokenizer.from_pretrained(args.target_model, use_fast=True)
    ensure_leftpad_with_pad(tgt_tok, args.max_prompt_len)
    tgt_model = AutoModelForCausalLM.from_pretrained(args.target_model, torch_dtype=dtype).eval().to(device)
    for p in tgt_model.parameters(): p.requires_grad = False

    d_src = src_model.config.hidden_size
    d_tgt = tgt_model.config.hidden_size
    if is_main(): log(f"Source hidden dim: {d_src} | Target hidden dim: {d_tgt}")

    translator = DiTTranslator(
        b=args.bottleneck_dim, depth=args.dit_depth, heads=args.dit_heads,
        src_dim=d_src, tgt_dim=d_tgt, S=args.cond_tokens
    ).to(device)
    if is_dist():
        translator = DDP(translator, device_ids=[local_rank()], output_device=local_rank(),
                         find_unused_parameters=False)

    log("Loading GSM8K...")
    ds = load_dataset("gsm8k", "main")
    train_ds, test_ds = ds["train"], ds["test"]

    # Optimizer & scheduler
    optim = torch.optim.AdamW([p for p in translator.parameters() if p.requires_grad],
                              lr=args.lr, weight_decay=args.weight_decay)
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=args.warmup_steps,
                                            num_training_steps=args.train_steps)

    step, running = 0, 0.0
    rng = random.Random(args.seed)
    if is_main(): os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    while step < args.train_steps:
        idxs = [rng.randrange(0, len(train_ds)) for _ in range(args.per_device_batch)]
        batch = {"question":[train_ds[i]["question"] for i in idxs],
                 "answer":[train_ds[i]["answer"]   for i in idxs]}
        samples = build_samples(batch, src_tok, tgt_tok, args.max_prompt_len, device)

        translator.train()
        tot_loss, dloss, lmloss = diffusion_training_step(
            translator.module if isinstance(translator, DDP) else translator,
            src_model, src_tok, tgt_model, tgt_tok, samples, device, dtype,
            lm_aux_weight=args.lm_aux_weight
        )
        optim.zero_grad(set_to_none=True)
        tot_loss.backward()
        optim.step()
        sched.step()
        step += 1
        running += tot_loss.item()

        if step % 20 == 0 and is_main():
            log(f"Step {step}/{args.train_steps} | Loss(avg/20) {running/20:.4f} | diff {dloss.item():.4f} | lm {lmloss.item():.4f}")
            running = 0.0

        if args.eval_every > 0 and step % args.eval_every == 0 and is_main():
            translator.eval()
            acc_base, acc_bridge = evaluate_gsm8k(
                test_ds, src_model, src_tok, tgt_model, tgt_tok,
                translator.module if isinstance(translator, DDP) else translator,
                args.max_prompt_len, device, dtype, num_samples=args.eval_samples, max_new_tokens=args.max_new_tokens
            )
            log(f"[Eval] Step {step} | Target-alone acc: {acc_base:.3f} | DiT-bridged acc: {acc_bridge:.3f}")

    if is_main():
        state = (translator.module if isinstance(translator, DDP) else translator).state_dict()
        torch.save(state, args.save_path)
        log(f"Saved translator to {args.save_path}")
        translator.eval()
        acc_base, acc_bridge = evaluate_gsm8k(
            test_ds, src_model, src_tok, tgt_model, tgt_tok,
            translator.module if isinstance(translator, DDP) else translator,
            args.max_prompt_len, device, dtype, num_samples=args.eval_samples, max_new_tokens=args.max_new_tokens
        )
        log(f"[Final Eval] Target-alone acc: {acc_base:.3f} | DiT-bridged acc: {acc_bridge:.3f}")
    cleanup_ddp()

if __name__ == "__main__":
    main()
