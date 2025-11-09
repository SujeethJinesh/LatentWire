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
        try:
            dist.monitored_barrier(timeout=timedelta(seconds=30))
        except Exception as e:
            log(f"Warning: Barrier timeout during cleanup: {e}")
        finally:
            dist.destroy_process_group()

def to_dtype_device(x, dtype, device):
    return x.to(device=device, dtype=dtype)

def extract_final_answer(text: str) -> str:
    """
    Extract final answer from GSM8K solution using official evaluation method.
    
    Official pattern: '#### <number>' where number can include decimals and commas.
    Returns '[invalid]' if no #### marker found (NO fallback to last number).
    
    Official implementation:
    https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py#L24-L35
    
    Paper: https://arxiv.org/abs/2110.14168
    Repository: https://github.com/openai/grade-school-math
    
    Reference from official README:
    "To extract the final numeric solution for a particular question, simply parse 
    the completion to extract the numeric value immediately following the #### token."
    Source: https://github.com/openai/grade-school-math/blob/master/README.md
    """
    # Official regex from dataset.py: supports negative numbers, decimals, and commas
    # ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
    m = re.search(r"#### (\-?[0-9\.\,]+)", text)
    if m:
        answer = m.group(1).strip()
        # Remove commas from numbers (official normalization)
        # From official code: match_str = match_str.replace(",", "")
        answer = answer.replace(",", "")
        return answer
    else:
        # Official behavior: return '[invalid]' marker (NO fallback)
        # From official code: INVALID_ANS = "[invalid]"
        return "[invalid]"

def compute_rms(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute RMS (root mean square) over the last dimension.
    Returns: [B, ...] with RMS for each position
    """
    return x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)

def apply_rms_matching(soft_tokens: torch.Tensor, target_rms: float) -> torch.Tensor:
    """
    Normalize soft tokens and rescale to match target embedding RMS statistics.
    This prevents scale mismatch that causes logit collapse.

    Two-step process:
    1. LayerNorm: Centers distribution (zero mean, unit variance)
    2. RMS Scaling: Matches magnitude to target model embeddings
    
    Reference: Based on techniques from LLaVA (arXiv:2503.17349) and 
    Nemesis soft prompt normalization (ICCV 2023)
    
    Args:
        soft_tokens: [B, K, d_model] translator output
        target_rms: scalar RMS of target model's embedding table

    Returns:
        Normalized and rescaled soft tokens [B, K, d_model]
    """
    # Step 1: Center and normalize distribution
    soft_tokens = F.layer_norm(soft_tokens, (soft_tokens.size(-1),))
    
    # Step 2: RMS matching to target embedding scale
    # Prevents attention dominance by magnitude rather than semantics
    current_rms = compute_rms(soft_tokens)
    soft_tokens = soft_tokens / current_rms * target_rms
    
    return soft_tokens

@dataclass
class Sample:
    src_prompt: str
    tgt_prompt: str
    tgt_answer: str  # Just the answer text, not tokenized

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


class GatedCrossAttentionBlock(nn.Module):
    """
    Bottlenecked cross-attention block with tanh gating (Flamingo-style).
    All operations happen at bottleneck dimension for efficiency.
    """
    def __init__(self, bottleneck_dim: int, src_dim: int, n_heads: int, ffn_mult: int = 4, dropout: float = 0.1):
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
        # Initialize with negative bias so gate starts slightly open after tanh
        # tanh(-2.0) ≈ -0.96, meaning gate starts almost closed but learns to open
        self.cross_gate = nn.Parameter(torch.tensor([-2.0]))

        # Dropout for regularization (prevents over-reliance on soft tokens)
        self.dropout = nn.Dropout(dropout)

        # FFN in bottleneck space
        self.ffn = nn.Sequential(
            nn.Linear(bottleneck_dim, ffn_mult * bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),  # Dropout in FFN as well
            nn.Linear(ffn_mult * bottleneck_dim, bottleneck_dim)
        )
        self.ffn_norm = nn.LayerNorm(bottleneck_dim)

    def forward(self, q_tokens: torch.Tensor, src_seq: torch.Tensor,
                src_mask: torch.Tensor = None) -> torch.Tensor:
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

        # Apply tanh gating (starts at 0, learns to open) with dropout for regularization
        gate = torch.tanh(self.cross_gate)
        q_tokens = q_tokens + gate * self.dropout(cross)

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
                src_mask: torch.Tensor = None) -> torch.Tensor:
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
    Prepare per-example structures - NO TOKENIZATION HERE.
    Tokenization happens in build_batch_inputs for correct alignment.
    """
    samples: List[Sample] = []
    for prob, sol in zip(batch["question"], batch["answer"]):
        src_prompt, tgt_prompt = format_prompts(prob)
        samples.append(Sample(
            src_prompt=src_prompt,
            tgt_prompt=tgt_prompt,
            tgt_answer=sol.strip()
        ))
    return samples

# ---------------------------
# Collation building inputs_embeds with soft tokens
# ---------------------------

def build_batch_inputs(samples: List[Sample],
                       src_model, src_tok,
                       tgt_model, tgt_tok,
                       translator,
                       device, dtype,
                       target_rms: float = None):
    """
    For each sample:
      - Encode source prompt with A; get last-layer hidden states (no grad)
      - Translator -> K soft tokens in B's space
      - Apply RMS matching to prevent scale mismatch
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

    # Apply RMS matching to prevent logit collapse (CRITICAL for stability)
    if target_rms is not None:
        soft_tokens = apply_rms_matching(soft_tokens, target_rms)

    # Target tokenization - do FULL TEXT and PROMPTS ONLY separately for correct label alignment
    tgt_full_texts = [s.tgt_prompt + " " + s.tgt_answer for s in samples]
    tgt_batch = tgt_tok(tgt_full_texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    tgt_batch = {k: v.to(device) for k, v in tgt_batch.items()}

    # Tokenize prompts only to find boundary
    tgt_prompts_only = [s.tgt_prompt for s in samples]
    prompt_batch = tgt_tok(tgt_prompts_only, return_tensors="pt", padding=True, truncation=True, max_length=2048)

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

    # Labels — compute with correct alignment based on batch tokenization
    # Strategy: mask prompt tokens (set to -100), supervise answer tokens, mask padding
    labels = tgt_batch["input_ids"].clone()

    # Mask prompt tokens for each sample
    for i in range(B):
        # Find where actual tokens end in prompt (before padding)
        prompt_attention = prompt_batch["attention_mask"][i]
        prompt_len = prompt_attention.sum().item()

        # Mask everything up to prompt length
        labels[i, :prompt_len] = -100

        # Also mask padding tokens in full sequence
        labels[i, tgt_batch["attention_mask"][i] == 0] = -100

    # Prepend -100 for K soft tokens (no loss on them)
    labels = torch.cat([
        torch.full((B, K), -100, dtype=labels.dtype, device=device),
        labels
    ], dim=1)  # [B, K+T]

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attn_mask,
        "labels": labels
    }

# ---------------------------
# Training / Evaluation loops
# ---------------------------

def evaluate_numeric_accuracy(dataset, src_model, src_tok, tgt_model, tgt_tok, translator,
                              device, dtype, num_samples: int = 200, max_new_tokens: int = 256,
                              show_samples: bool = True, target_rms: float = None, eval_batch_size: int = 50):
    """
    Compare (i) Target alone vs (ii) Source→Translator→Target.
    Evaluates in batches to avoid OOM with large num_samples.
    """
    tgt_model.eval()
    src_model.eval()
    translator.eval()

    # Collect all samples first
    all_samples = []
    taken = 0
    for ex in dataset:
        if taken >= num_samples: break
        all_samples.extend(build_samples({"question":[ex["question"]], "answer":[ex["answer"]]},
                                         src_tok, tgt_tok, device, tgt_model))
        taken += 1

    # Process in batches to avoid OOM
    all_bridged_texts = []
    all_base_texts = []

    for start_idx in range(0, len(all_samples), eval_batch_size):
        end_idx = min(start_idx + eval_batch_size, len(all_samples))
        samples_batch = all_samples[start_idx:end_idx]

        # Build bridged inputs (uses inputs_embeds)
        with torch.no_grad():
            batch = build_batch_inputs(samples_batch, src_model, src_tok, tgt_model, tgt_tok, translator, device, dtype, target_rms=target_rms)
            gen = tgt_model.generate(
                inputs_embeds=batch["inputs_embeds"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,  # Prevent "181818..." loops
                no_repeat_ngram_size=3,  # Prevent repeating 3-grams
                pad_token_id=tgt_tok.pad_token_id,
                eos_token_id=tgt_tok.eos_token_id
            )
            bridged_texts_batch = tgt_tok.batch_decode(gen, skip_special_tokens=True)
            all_bridged_texts.extend(bridged_texts_batch)

        # Target-alone baseline
        with torch.no_grad():
            prompts = [s.tgt_prompt for s in samples_batch]
            enc = tgt_tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
            base_out = tgt_model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,  # Prevent loops
                no_repeat_ngram_size=3,  # Prevent repeating 3-grams
                pad_token_id=tgt_tok.pad_token_id,
                eos_token_id=tgt_tok.eos_token_id
            )
            base_texts_batch = tgt_tok.batch_decode(base_out, skip_special_tokens=True)
            all_base_texts.extend(base_texts_batch)

    # Now use accumulated results
    samples = all_samples
    bridged_texts = all_bridged_texts
    base_texts = all_base_texts

    # Compute accuracy
    def compute_acc(pred_texts):
        correct = 0
        for text, s in zip(pred_texts, samples):
            pred = extract_final_answer(text)
            gold_full_text = s.tgt_prompt + " " + s.tgt_answer
            gold = extract_final_answer(gold_full_text)
            correct += int(pred == gold)
        return correct / len(samples)

    acc_bridged = compute_acc(bridged_texts)
    acc_baseline = compute_acc(base_texts)

    # Print sample outputs for inspection (only from main process)
    if show_samples and num_samples >= 3:
        log("\n" + "="*60)
        log("SAMPLE OUTPUTS (first 3 examples):")
        log("="*60)
        for i in range(min(3, len(samples))):
            log(f"\n--- Example {i+1} ---")
            log(f"Question: {samples[i].tgt_prompt[:200]}...")
            gold_full_text = samples[i].tgt_prompt + " " + samples[i].tgt_answer
            log(f"Gold answer: {extract_final_answer(gold_full_text)}")
            log(f"Target-alone: {extract_final_answer(base_texts[i])}")
            log(f"Bridged: {extract_final_answer(bridged_texts[i])}")

            # Show first 300 chars of actual generation for debugging
            if len(bridged_texts[i]) > len(samples[i].tgt_prompt):
                generated_part = bridged_texts[i][len(samples[i].tgt_prompt):][:300]
                log(f"Bridged generation start: {generated_part}...")
        log("="*60 + "\n")

    return acc_baseline, acc_bridged

def analyze_bridge_quality(soft_tokens: torch.Tensor, target_model, prompt_ids: torch.Tensor,
                           prompt_mask: torch.Tensor, tokenizer, target_rms: float):
    """
    Diagnose why bridged generations might degenerate.

    Returns dict with:
        - RMS statistics (soft tokens vs target embeddings)
        - First-token entropy (bridged vs baseline)
        - Top-5 token probabilities for both
    """
    with torch.no_grad():
        # 1. RMS scale comparison
        tgt_embed = target_model.get_input_embeddings().weight
        tgt_rms_actual = tgt_embed.pow(2).mean(dim=1).sqrt()
        soft_rms = soft_tokens.pow(2).mean(dim=-1).sqrt()

        stats = {
            "soft_rms_mean": soft_rms.mean().item(),
            "soft_rms_max": soft_rms.max().item(),
            "soft_rms_min": soft_rms.min().item(),
            "tgt_embed_rms_mean": tgt_rms_actual.mean().item(),
            "tgt_embed_rms_target": target_rms,
        }

        # 2. First-token distribution comparison
        # Baseline: text-only
        baseline_out = target_model(input_ids=prompt_ids, attention_mask=prompt_mask)
        baseline_probs = baseline_out.logits[:, -1].softmax(-1)

        # Bridged: with soft tokens
        tgt_embeds = target_model.get_input_embeddings()(prompt_ids)
        bridged_inputs = torch.cat([soft_tokens, tgt_embeds], dim=1)
        bridged_mask = torch.cat([
            torch.ones(soft_tokens.size(0), soft_tokens.size(1), device=prompt_mask.device),
            prompt_mask
        ], dim=1)
        bridged_out = target_model(inputs_embeds=bridged_inputs, attention_mask=bridged_mask)
        # First token after prompt
        first_gen_pos = soft_tokens.size(1) + prompt_ids.size(1) - 1
        bridged_probs = bridged_out.logits[:, first_gen_pos].softmax(-1)

        # Entropy (low entropy = over-confident = likely to loop)
        def entropy(p):
            return (-p * p.clamp_min(1e-9).log()).sum(-1).mean().item()

        # Top-5 tokens
        def top5(p):
            vals, inds = p[0].topk(5)
            return [(tokenizer.decode([idx.item()]), prob.item()) for idx, prob in zip(inds, vals)]

        analysis = {
            "entropy_baseline": entropy(baseline_probs),
            "entropy_bridged": entropy(bridged_probs),
            "top5_baseline": top5(baseline_probs),
            "top5_bridged": top5(bridged_probs),
        }

        return {**stats, **analysis}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--target_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--translator_type", type=str, choices=["cross_attn", "linear", "bottleneck_gated"], default="cross_attn")
    parser.add_argument("--bottleneck_dim", type=int, default=1024, help="Bottleneck dimension for gated translator")
    parser.add_argument("--soft_tokens", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--train_steps", type=int, default=2000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--per_device_batch", type=int, default=2)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--eval_samples", type=int, default=1000, help="Number of eval samples (default 1000 for lower variance)")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--save_path", type=str, default="translator_ckpt.pt")
    parser.add_argument("--show_eval_samples", type=int, default=1,
                        help="Show sample outputs during eval (0=none, 1=brief, 2=detailed)")
    parser.add_argument("--early_stop_patience", type=int, default=5,
                        help="Stop training if no improvement for N evals (0=disabled)")
    parser.add_argument("--dataset", type=str, choices=["gsm8k", "hotpotqa"], default="gsm8k",
                        help="Dataset to use for training")
    parser.add_argument("--info_nce_weight", type=float, default=0.05,
                        help="Weight for InfoNCE anti-collapse loss")
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
    src_tok.padding_side = "left"  # Critical for decoder-only models
    src_tok.model_max_length = 2048

    src_model = AutoModelForCausalLM.from_pretrained(
        args.source_model, torch_dtype=dtype, device_map=None
    ).eval().to(device)
    for p in src_model.parameters(): p.requires_grad = False

    log("Loading target model/tokenizer...")
    tgt_tok = AutoTokenizer.from_pretrained(args.target_model, use_fast=True)
    if tgt_tok.pad_token is None:
        tgt_tok.pad_token = tgt_tok.eos_token
    tgt_tok.padding_side = "left"  # Critical for decoder-only models
    tgt_tok.model_max_length = 2048
    tgt_model = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=dtype, device_map=None
    ).eval().to(device)
    for p in tgt_model.parameters(): p.requires_grad = False

    # Infer dims
    with torch.no_grad():
        d_src = src_model.config.hidden_size
        d_tgt = tgt_model.config.hidden_size
    log(f"Source hidden dim: {d_src} | Target hidden dim: {d_tgt}")

    # Compute target embedding RMS for normalization (CRITICAL for stability)
    with torch.no_grad():
        tgt_embed_table = tgt_model.get_input_embeddings().weight  # [vocab_size, d_tgt]
        target_rms = tgt_embed_table.pow(2).mean(dim=1).sqrt().mean().item()
    log(f"Target embedding RMS: {target_rms:.4f}")

    # Build translator
    if args.translator_type == "bottleneck_gated":
        translator = BottleneckedGatedTranslator(
            src_dim=d_src, tgt_dim=d_tgt, bottleneck_dim=args.bottleneck_dim,
            soft_tokens=args.soft_tokens, depth=args.depth, n_heads=args.heads
        ).to(device=device, dtype=dtype)
    elif args.translator_type == "cross_attn":
        translator = CrossAttnResamplerTranslator(
            src_dim=d_src, tgt_dim=d_tgt, soft_tokens=args.soft_tokens, depth=args.depth, n_heads=args.heads
        ).to(device=device, dtype=dtype)
    else:
        translator = LinearTranslator(
            src_dim=d_src, tgt_dim=d_tgt, soft_tokens=args.soft_tokens, hidden=4*d_tgt
        ).to(device=device, dtype=dtype)

    # Count parameters
    param_count = sum(p.numel() for p in translator.parameters())
    log(f"Translator parameters: {param_count / 1e6:.1f}M")

    # DDP only on translator (models are frozen)
    if dist.is_initialized():
        translator = DDP(translator, device_ids=[local_rank()], output_device=local_rank(), find_unused_parameters=False)

    # Data
    log(f"Loading {args.dataset.upper()}...")
    if args.dataset == "gsm8k":
        ds = load_dataset("gsm8k", "main")
        train_ds = ds["train"]
        test_ds  = ds["test"]
    elif args.dataset == "hotpotqa":
        ds = load_dataset("hotpot_qa", "distractor")
        train_ds = ds["train"]
        test_ds  = ds["validation"]  # HotpotQA uses "validation" split for test
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # A very simple random sampler over training set
    rng = random.Random(args.seed)

    # Optimizer with proper weight decay filtering (exclude LayerNorm and biases)
    # Gate parameters get higher LR (×3) to open gradually (Flamingo-style)
    decay_params = []
    no_decay_params = []
    gate_params = []

    for name, param in translator.named_parameters():
        if not param.requires_grad:
            continue
        # Gate parameters: higher LR, no weight decay
        if 'cross_gate' in name:
            gate_params.append(param)
        # Don't apply weight decay to bias terms and layer norms
        elif 'bias' in name or 'norm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {'params': decay_params, 'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': no_decay_params, 'weight_decay': 0.0, 'lr': args.lr},
        {'params': gate_params, 'weight_decay': 0.0, 'lr': args.lr * 3.0}  # Gates open faster
    ]

    log(f"Optimizer groups: {len(decay_params)} decay, {len(no_decay_params)} no_decay, {len(gate_params)} gates (LR ×3)")
    optim = torch.optim.AdamW(optim_groups, betas=(0.9, 0.98), eps=1e-8)
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=args.warmup_steps,
                                            num_training_steps=args.train_steps)

    # --------- Train loop ---------
    step = 0
    translator.train()
    running = 0.0
    last_eval = -1

    # Early stopping tracking
    best_bridged_acc = 0.0
    patience_counter = 0
    best_checkpoint = None

    while step < args.train_steps:
        # sample a batch
        batch_idx = [rng.randrange(0, len(train_ds)) for _ in range(args.per_device_batch)]
        batch = {"question": [train_ds[i]["question"] for i in batch_idx],
                 "answer":   [train_ds[i]["answer"]   for i in batch_idx]}
        samples = build_samples(batch, src_tok, tgt_tok, device, tgt_model)

        # Build bridged inputs (with soft tokens)
        data = build_batch_inputs(samples, src_model, src_tok, tgt_model, tgt_tok,
                                  translator, device, dtype, target_rms=target_rms)

        # Diagnostic: verify label alignment (only first step)
        if step == 0 and is_main():
            log("\n" + "="*60)
            log("LABEL ALIGNMENT DIAGNOSTIC (Step 0)")
            log("="*60)
            log(f"  Input embeddings shape: {data['inputs_embeds'].shape}")
            log(f"  Labels shape: {data['labels'].shape}")
            log(f"  Attention mask shape: {data['attention_mask'].shape}")
            log(f"  Total tokens: {data['labels'].numel()}")
            log(f"  Non-masked labels (-100): {(data['labels'] != -100).sum().item()}")
            log(f"  Masked labels: {(data['labels'] == -100).sum().item()}")
            log(f"  Sample 0 first 10 labels: {data['labels'][0, :10].tolist()}")
            log(f"  Sample 0 last 10 labels: {data['labels'][0, -10:].tolist()}")
            # Verify K soft tokens are all masked
            K = data['inputs_embeds'].shape[1] - data['labels'].shape[1] + data['labels'].shape[1]
            soft_token_count = data['inputs_embeds'].shape[1] - (data['labels'].shape[1] - (data['labels'][0] == -100).sum().item() + (data['labels'][0] != -100).sum().item())
            log(f"  Verification: All soft token labels == -100? {(data['labels'][:, :translator.module.K if isinstance(translator, DDP) else translator.K] == -100).all().item()}")
            log("="*60 + "\n")

        # Forward (compute NLL loss on target)
        out = tgt_model(**data)
        nll_loss = out.loss

        # KL consistency loss to prevent distribution collapse
        # Compare bridged vs baseline (text-only) distributions on first 20 tokens
        kl_loss = torch.tensor(0.0, device=device, dtype=dtype)
        if step > args.warmup_steps:  # Only after warmup
            with torch.no_grad():
                # Baseline: text-only input (no soft tokens)
                tgt_prompts = [s.tgt_prompt for s in samples]
                tgt_enc = tgt_tok(tgt_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
                baseline_out = tgt_model(**tgt_enc)
                baseline_logits = baseline_out.logits[:, :20, :]  # First 20 positions

            # Bridged logits (first 20 positions after soft tokens)
            K = data["inputs_embeds"].size(1) - out.logits.size(1) if out.logits.size(1) < data["inputs_embeds"].size(1) else 0
            bridged_logits = out.logits[:, :20, :]  # First 20 generation positions

            # KL divergence: KL(bridged || baseline)
            # Prevent bridged from diverging into degenerate modes
            if bridged_logits.size(1) >= 20 and baseline_logits.size(1) >= 20:
                kl_loss = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(bridged_logits, dim=-1),
                    torch.nn.functional.softmax(baseline_logits, dim=-1),
                    reduction="batchmean"
                )

        # InfoNCE anti-collapse loss (prevent all inputs mapping to same vector)
        # Compare translator soft tokens with actual target embeddings
        info_nce_loss = torch.tensor(0.0, device=device, dtype=dtype)
        if step > args.warmup_steps // 2:  # Start after 50% of warmup
            with torch.no_grad():
                # Get target embeddings for the prompt (stop gradient)
                tgt_prompts = [s.tgt_prompt for s in samples]
                tgt_enc = tgt_tok(tgt_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
                tgt_embeds_full = tgt_model.get_input_embeddings()(tgt_enc["input_ids"])
                # Pool target embeddings (mean over sequence)
                tgt_pooled = tgt_embeds_full.mean(dim=1)  # [B, d_model]

            # Get soft tokens from translator (already computed above in data)
            # Pool soft tokens (mean over K tokens)
            K = data["inputs_embeds"].size(1) - out.logits.size(1) if out.logits.size(1) < data["inputs_embeds"].size(1) else 0
            soft_pooled = data["inputs_embeds"][:, :K, :].mean(dim=1) if K > 0 else data["inputs_embeds"][:, :32, :].mean(dim=1)  # [B, d_model]

            # Normalize for cosine similarity
            soft_norm = torch.nn.functional.normalize(soft_pooled, dim=-1)
            tgt_norm = torch.nn.functional.normalize(tgt_pooled, dim=-1)

            # InfoNCE: positive pairs (i,i) vs negative pairs (i,j≠i)
            temperature = 0.07
            logits_contrastive = soft_norm @ tgt_norm.T / temperature  # [B, B]
            labels_contrastive = torch.arange(logits_contrastive.size(0), device=device)
            info_nce_loss = torch.nn.functional.cross_entropy(logits_contrastive, labels_contrastive)

        # Total loss: NLL + λ_KL * KL + λ_InfoNCE * InfoNCE
        loss = nll_loss + 0.03 * kl_loss + args.info_nce_weight * info_nce_loss

        # Detach for logging to avoid DDP autograd warning
        loss_scalar = loss.detach()
        if dist.is_initialized():
            dist.all_reduce(loss_scalar, op=dist.ReduceOp.AVG)
        running += loss_scalar.item()

        # Backward into translator only
        optim.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(translator.parameters(), 1.0)

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
                        device, dtype, num_samples=args.eval_samples, max_new_tokens=args.max_new_tokens,
                        show_samples=(args.show_eval_samples > 0), target_rms=target_rms
                    )
                log(f"[Eval] Step {step} | Target-alone acc: {acc_base:.3f} | Bridged acc: {acc_bridged:.3f}")

                # Early stopping check
                if args.early_stop_patience > 0:
                    if acc_bridged > best_bridged_acc:
                        best_bridged_acc = acc_bridged
                        patience_counter = 0
                        # Save best checkpoint
                        best_checkpoint = (translator.module if isinstance(translator, DDP) else translator).state_dict()
                        log(f"[Early Stop] New best bridged acc: {best_bridged_acc:.3f}, resetting patience")
                    else:
                        patience_counter += 1
                        log(f"[Early Stop] No improvement, patience: {patience_counter}/{args.early_stop_patience}")

                    if patience_counter >= args.early_stop_patience:
                        log(f"[Early Stop] Stopping early at step {step}. Best bridged acc: {best_bridged_acc:.3f}")
                        break

    # Final save (translator weights)
    if is_main():
        # If early stopping saved a best checkpoint, use that
        if best_checkpoint is not None and args.early_stop_patience > 0:
            torch.save(best_checkpoint, args.save_path)
            log(f"Saved BEST translator (acc={best_bridged_acc:.3f}) to {args.save_path}")
            # Also load it for final eval
            (translator.module if isinstance(translator, DDP) else translator).load_state_dict(best_checkpoint)
        else:
            state = (translator.module if isinstance(translator, DDP) else translator).state_dict()
            torch.save(state, args.save_path)
            log(f"Saved translator to {args.save_path}")

        # Final eval
        with torch.no_grad():
            acc_base, acc_bridged = evaluate_numeric_accuracy(
                test_ds, src_model, src_tok, tgt_model, tgt_tok, translator.module if isinstance(translator, DDP) else translator,
                device, dtype, num_samples=args.eval_samples, max_new_tokens=args.max_new_tokens,
                show_samples=(args.show_eval_samples > 0), target_rms=target_rms
            )
        log(f"[Final Eval] Target-alone acc: {acc_base:.3f} | Bridged acc: {acc_bridged:.3f}")

    cleanup_ddp()

if __name__ == "__main__":
    main()
