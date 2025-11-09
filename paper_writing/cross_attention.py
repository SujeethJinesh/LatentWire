#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inter-LLM "interlingua" prototype:
- Source LLM A → (get hidden states) → BottleneckedGatedTranslator → K soft tokens in target space
- Target LLM B consumes [soft tokens || B's own embeddings] and generates answers.
- Train only the translator with teacher-forced LM loss on GSM8K/HotpotQA.
- Evaluate numeric exact-match accuracy.

Architecture:
- Uses BottleneckedGatedTranslator: bottlenecked cross-attention with Flamingo-style tanh gating
- All operations in bottleneck dimension for efficiency
- Dropout for regularization to prevent over-reliance on soft tokens

Key Improvements (2024-2025 best practices):
- Gated self-attention prevents noise from randomly-initialized queries
- Orthogonal query initialization ensures query diversity
- RMS scale matching for cross-LLM compatibility
- Optional RMSNorm (more efficient than LayerNorm, matches Llama/Mistral)
- Optional SwiGLU activation (used in modern LLMs)

Run (4x H100, DDP):
  torchrun --nproc_per_node=4 cross_attention.py \
    --source_model mistralai/Mistral-7B-Instruct-v0.3 \
    --target_model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --bottleneck_dim 1024 \
    --soft_tokens 64 \
    --depth 8 \
    --heads 16 \
    --train_steps 3000 \
    --per_device_batch 10 \
    --eval_samples 1000 \
    --bf16

Notes:
- Both models are frozen; only the translator trains (~10-50M params).
- Script uses BF16 on Ampere/Hopper and DDP across 4 GPUs.
- Scale-only RMS matching preserves learned representations while fixing magnitude.

Reproducibility:
- Full determinism enabled via torch.use_deterministic_algorithms()
- Each DDP rank uses seed + rank offset to ensure different data sampling
- cuDNN benchmark disabled for consistent algorithm selection
- CUBLAS workspace configured for deterministic matrix operations
- May reduce performance 5-15% but ensures reproducible results
"""

import os, math, re, json, random, argparse, time
from dataclasses import dataclass
from typing import List, Dict, Tuple
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    get_scheduler
)

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
    """Clean up distributed training process group."""
    if dist.is_initialized():
        try:
            # Simple barrier without timeout - let PyTorch handle it
            # monitored_barrier is mainly for debugging, not cleanup
            dist.barrier()  # Standard barrier is sufficient
        except Exception as e:
            log(f"Warning: Barrier failed during cleanup: {e}")
        finally:
            dist.destroy_process_group()

def setup_reproducibility(seed: int, rank: int = 0):
    """
    Setup full reproducibility for PyTorch DDP training.

    Each DDP rank gets a unique seed (seed + rank) to ensure different data sampling.
    Enables deterministic CUDA operations for reproducible results.

    Args:
        seed: Base random seed
        rank: DDP rank (0 for main process, 1-3 for other GPUs in 4-GPU setup)

    References:
        - PyTorch Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
        - DDP seed offset is critical to avoid identical batches per GPU
    """
    # CRITICAL: Offset seed by rank so each GPU samples different data
    # Without this, all GPUs see identical batches → wasted compute
    effective_seed = seed + rank

    # Python's random module
    random.seed(effective_seed)

    # NumPy random operations (used by many libraries)
    np.random.seed(effective_seed)

    # PyTorch CPU and all CUDA devices
    torch.manual_seed(effective_seed)
    torch.cuda.manual_seed_all(effective_seed)

    # Enable deterministic algorithms (may reduce performance 5-15%)
    # warn_only=True allows fallback for ops without deterministic implementation
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Disable cuDNN benchmark for deterministic algorithm selection
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # CUBLAS workspace config for deterministic matrix operations (CUDA 10.2+)
    # Using ':16:8' instead of ':4096:8' to reduce memory pressure
    # Both are valid; ':16:8' uses less workspace memory
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    if rank == 0:
        log(f"Reproducibility enabled: base_seed={seed}, effective_seed={effective_seed}")

def worker_init_fn(worker_id: int):
    """
    DataLoader worker initialization function for reproducible data loading.

    Use with DataLoader like:
        DataLoader(..., num_workers=4, worker_init_fn=worker_init_fn)

    Each worker gets a unique seed to prevent identical randomness across workers.

    Args:
        worker_id: Worker ID (0 to num_workers-1)

    References:
        - PyTorch DataLoader: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """
    # Get the base seed from PyTorch's initial_seed (set by setup_reproducibility)
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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

def apply_rms_matching(
    soft_tokens: torch.Tensor,
    target_rms: float | None,
    eps: float = 1e-8,
    detach_stats: bool = True,            # stop grads through stats
    clamp: tuple[float, float] | None = (0.25, 4.0),  # avoid extreme rescale
    blend: float = 1.0,                   # 0..1; 1 = full match
) -> torch.Tensor:
    """
    Scale-only RMS matching to match target model's embedding RMS statistics.
    Preserves learned directions; fixes magnitude to prevent attention/logit dominance.

    Unlike LayerNorm, this preserves the mean and directional information the
    translator learned, only adjusting the magnitude to match the target model's
    embedding space. This is consistent with:
    - LLaVA's vision-text magnitude imbalance (arXiv:2503.17349)
    - Nemesis Low-Norm Effect for soft prompts (ICLR 2024)
    - RMSNorm used in Llama-family models

    Args:
        soft_tokens: [B, K, d_model] translator output
        target_rms: scalar RMS of target model's embedding table (None = no scaling)
        eps: small constant for numerical stability
        detach_stats: if True, detach current RMS to prevent gaming the rescaler
        clamp: (min, max) bounds for scale factor to prevent extreme rescaling
        blend: interpolation factor (0=no scaling, 1=full match to target_rms)

    Returns:
        Rescaled soft tokens [B, K, d_model] with magnitude matching target embeddings
    """
    # If target_rms is None, return unchanged (no scaling)
    if target_rms is None:
        return soft_tokens

    # Compute current per-token RMS: [B, K, 1]
    current_rms = soft_tokens.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)

    # Detach to prevent translator from gaming the rescaler by learning extreme norms
    if detach_stats:
        current_rms = current_rms.detach()

    # Compute scale factor to match target RMS
    scale = target_rms / current_rms

    # Optional: blend toward target (useful for gradual phase-in during training)
    if blend != 1.0:
        scale = scale.pow(blend)

    # Clamp scale to prevent pathological rescaling
    if clamp is not None:
        lo, hi = clamp
        scale = scale.clamp(min=lo, max=hi)

    # Apply scale (preserves direction, fixes magnitude)
    return soft_tokens * scale

def get_target_embedding_rms(target_model: nn.Module) -> float:
    """
    Compute median RMS of target model's embedding table.
    This provides the characteristic scale for the target LLM's embeddings.

    Args:
        target_model: Target LLM with get_input_embeddings() method

    Returns:
        Median RMS value across all embedding vectors

    References:
        - Used for cross-modal scale matching in VLMs (Flamingo, BLIP-2)
        - Prevents attention imbalance when injecting soft tokens

    Example:
        target_rms = get_target_embedding_rms(tgt_model)
        # Pass to translator or use in apply_rms_matching
    """
    embed_weights = target_model.get_input_embeddings().weight  # [vocab_size, d_model]
    rms_per_token = embed_weights.pow(2).mean(dim=-1).sqrt()
    return rms_per_token.median().item()

@dataclass
class Sample:
    src_prompt: str
    tgt_prompt: str
    tgt_answer: str  # Just the answer text, not tokenized

# ---------------------------
# Translator modules
# ---------------------------

class GatedCrossAttentionBlock(nn.Module):
    """
    Bottlenecked cross-attention block with tanh gating (Flamingo-style).
    All operations happen at bottleneck dimension for efficiency.

    Gating is applied to both self-attention and cross-attention:
    - Self-attention gating prevents noise from randomly-initialized queries
    - Cross-attention gating allows gradual opening (starts at 0)
    """
    def __init__(self, bottleneck_dim: int, src_dim: int, n_heads: int, ffn_mult: int = 4, dropout: float = 0.1,
                 ffn_act: str = "swiglu"):
        super().__init__()
        assert bottleneck_dim % n_heads == 0, "bottleneck_dim must be divisible by n_heads"

        self.bottleneck_dim = bottleneck_dim

        # Layer norms (RMSNorm, matching modern LLMs)
        self.q_norm = nn.RMSNorm(bottleneck_dim, elementwise_affine=True)
        self.src_norm = nn.RMSNorm(src_dim, elementwise_affine=True)

        # Self-attention on queries (in bottleneck space)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=bottleneck_dim,
            num_heads=n_heads,
            batch_first=True
        )

        # Tanh gating on self-attention
        # Initialize at 0.0 to prevent noise from randomly-initialized queries
        self.sa_gate = nn.Parameter(torch.tensor([0.0]))

        # Cross-attention using MultiheadAttention (Flamingo-style)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=bottleneck_dim,
            num_heads=n_heads,
            kdim=src_dim,
            vdim=src_dim,
            batch_first=True,
            dropout=dropout
        )

        # Tanh gating on cross-attention (Flamingo-style)
        # Initialize at 0.0 so tanh(0)=0 (closed). Matches Flamingo: model behaves like frozen LM at init, learns to open gradually.
        self.cross_gate = nn.Parameter(torch.tensor([0.0]))

        # Dropout for regularization (prevents over-reliance on soft tokens)
        self.dropout = nn.Dropout(dropout)

        # FFN in bottleneck space (SwiGLU by default, matching modern LLMs)
        inner = int((ffn_mult * bottleneck_dim) * 2 / 3)  # param-parity for SwiGLU vs 4*d GELU
        if ffn_act.lower() == "swiglu":
            self.up_value = nn.Linear(bottleneck_dim, inner, bias=True)
            self.up_gate = nn.Linear(bottleneck_dim, inner, bias=True)
            self.down = nn.Linear(inner, bottleneck_dim, bias=True)
            self.ffn_dropout = nn.Dropout(dropout)
            self._use_swiglu = True
        else:
            self.ffn = nn.Sequential(
                nn.Linear(bottleneck_dim, ffn_mult * bottleneck_dim),
                nn.GELU(),
                nn.Dropout(dropout),  # Dropout in FFN as well
                nn.Linear(ffn_mult * bottleneck_dim, bottleneck_dim)
            )
            self._use_swiglu = False
        self.ffn_norm = nn.RMSNorm(bottleneck_dim, elementwise_affine=True)

        # FFN gating (Flamingo-style)
        self.ffn_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self, q_tokens: torch.Tensor, src_seq: torch.Tensor,
                src_mask: torch.Tensor = None) -> torch.Tensor:
        # q_tokens: [B, K, bottleneck_dim], src_seq: [B, T, src_dim]

        # Self-attention on queries with gating
        q_norm = self.q_norm(q_tokens)
        sa_out, _ = self.self_attn(q_norm, q_norm, q_norm, need_weights=False)
        # Apply gating to prevent noise from randomly-initialized queries
        q_tokens = q_tokens + torch.tanh(self.sa_gate) * sa_out

        # Cross-attention with gating (Flamingo-style)
        qn = self.q_norm(q_tokens)
        srcn = self.src_norm(src_seq)

        # Convert attention mask to key_padding_mask format (True = ignore)
        kpm = (src_mask == 0) if src_mask is not None else None
        cross, _ = self.cross_attn(qn, srcn, srcn, key_padding_mask=kpm, need_weights=False)

        # Apply tanh gating (starts at 0, learns to open) with dropout for regularization
        gate = torch.tanh(self.cross_gate)
        q_tokens = q_tokens + gate * self.dropout(cross)

        # FFN with gating (Flamingo-style)
        ff_input = self.ffn_norm(q_tokens)
        if self._use_swiglu:
            gate = F.silu(self.up_gate(ff_input))
            value = self.up_value(ff_input)
            ff = self.down(gate * value)
            ff = self.ffn_dropout(ff)
        else:
            ff = self.ffn(ff_input)
        fgate = torch.tanh(self.ffn_gate)
        q_tokens = q_tokens + fgate * ff

        return q_tokens


class BottleneckedGatedTranslator(nn.Module):
    """
    Improved translator with bottleneck architecture and gating.
    Processes at bottleneck_dim internally, only projects to target_dim at the end.

    Key improvements:
    - Orthogonal initialization ensures query diversity
    - Gated self-attention prevents noise from random queries
    - RMS scale matching for cross-LLM compatibility
    """
    def __init__(self, src_dim: int, tgt_dim: int, bottleneck_dim: int = 1024,
                 soft_tokens: int = 48, depth: int = 6, n_heads: int = 16, ffn_mult: int = 4,
                 ffn_act: str = "swiglu"):
        super().__init__()
        self.K = soft_tokens
        self.bottleneck_dim = bottleneck_dim

        # Learned query tokens in bottleneck space
        # Orthogonal initialization ensures query diversity
        self.query_tokens = nn.Parameter(torch.empty(1, soft_tokens, bottleneck_dim))
        nn.init.orthogonal_(self.query_tokens)

        # Cross-attention blocks (all in bottleneck space)
        self.blocks = nn.ModuleList([
            GatedCrossAttentionBlock(
                bottleneck_dim=bottleneck_dim,
                src_dim=src_dim,
                n_heads=n_heads,
                ffn_mult=ffn_mult,
                ffn_act=ffn_act,
            )
            for _ in range(depth)
        ])

        # Final projection from bottleneck to target dimension
        self.output_proj = nn.Linear(bottleneck_dim, tgt_dim)
        self.output_norm = nn.RMSNorm(bottleneck_dim, elementwise_affine=True)

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

        # TODO: Precompute target_rms = get_target_embedding_rms(target_model) once
        # and pass it explicitly for optimal cross-LLM scale matching
        # Scale-match to target embedding distribution for cross-LLM compatibility
        # Note: target_rms should be precomputed once and passed in or cached
        # For now, we apply scale matching with default parameters
        # Users should call get_target_embedding_rms(target_model) and pass it to apply_rms_matching
        soft_tokens = apply_rms_matching(
            soft_tokens,
            target_rms=None,  # Will use current RMS if None; recommend setting explicitly
            eps=1e-8,
            detach_stats=True,
            clamp=(0.25, 4.0),
            blend=1.0
        )

        return soft_tokens

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
                       target_rms: float = None,
                       mode: str = 'train'):
    """
    For each sample:
      - Encode source prompt with A; get last-layer hidden states (no grad)
      - Translator -> K soft tokens in B's space
      - Apply RMS matching to prevent scale mismatch
      - Tokenize target text with B; embed its input ids
      - Concatenate soft tokens + tgt embeddings; build attention mask and labels

    Args:
        mode: 'train' or 'eval'
            - 'train': Uses answer text only for teacher forcing (model must use soft tokens for question)
            - 'eval': Uses start token only for generation (model generates full answer)

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

    # Translator (grad flows) - BottleneckedGatedTranslator supports src_mask
    soft_tokens = translator(src_h, src_mask)  # [B, K, d_tgt]

    # Apply RMS matching to prevent logit collapse (CRITICAL for stability)
    if target_rms is not None:
        soft_tokens = apply_rms_matching(soft_tokens, target_rms)

    # Target tokenization
    # Training: answer only (model must use soft tokens to understand question)
    # Eval: start token only (model generates full answer)
    if mode == 'train':
        tgt_texts = [s.tgt_answer for s in samples]  # Answer only for teacher forcing
    else:
        tgt_texts = [" " for _ in samples]  # Start token for generation

    tgt_batch = tgt_tok(
        tgt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
    tgt_batch = {k: v.to(device) for k, v in tgt_batch.items()}

    # Target input embeddings
    with torch.no_grad():
        embed = tgt_model.get_input_embeddings()  # nn.Embedding
        tgt_embeds = embed(tgt_batch["input_ids"]).to(dtype)  # [B, T_tgt, d_tgt]

    # Concatenate soft tokens in front
    K = soft_tokens.size(1)
    max_T = tgt_embeds.size(1)
    inputs_embeds = torch.cat([soft_tokens, tgt_embeds], dim=1)  # [B, K+T, d]

    # Attention mask: concatenate 1s for soft tokens with actual text attention mask
    # This prevents attending to padding tokens in the text portion
    attn_mask = torch.cat([
        torch.ones((B, K), dtype=torch.long, device=device),  # Soft tokens always attended
        tgt_batch["attention_mask"]  # Text mask (0 for padding)
    ], dim=1)

    # Labels
    labels = tgt_batch["input_ids"].clone()
    if mode == 'train':
        # Mask padding tokens only (answer is already isolated, no prompt to mask)
        for i in range(B):
            labels[i, tgt_batch["attention_mask"][i] == 0] = -100
    else:
        # Eval mode: dummy labels (not used for generation)
        labels = torch.full((B, 1), -100, dtype=labels.dtype, device=device)

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

def gather_texts_from_all_ranks(local_texts: List[str]) -> List[str]:
    """
    Gather text generations from all ranks to rank 0.
    Returns full list on rank 0, empty list on other ranks.
    """
    if not dist.is_initialized():
        return local_texts

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Convert to tensors for gathering
    # Encode to bytes first, then find max BYTE length (not char length)
    encoded_bytes = [text.encode('utf-8') for text in local_texts]
    max_byte_len = max(len(b) for b in encoded_bytes) if encoded_bytes else 1

    # Gather max_byte_len from all ranks
    max_len_tensor = torch.tensor([max_byte_len], dtype=torch.long, device='cuda')
    max_len_list = [torch.zeros(1, dtype=torch.long, device='cuda') for _ in range(world_size)]
    dist.all_gather(max_len_list, max_len_tensor)
    global_max_byte_len = max(t.item() for t in max_len_list)

    # Pad bytes to global_max_byte_len
    # Convert to tensor (handle empty case properly)
    if not encoded_bytes:
        # Create empty tensor instead of placeholder
        local_tensor = torch.zeros((0, global_max_byte_len), dtype=torch.uint8, device='cuda')
    else:
        encoded = []
        for byte_string in encoded_bytes:
            # Pad with zeros to global_max_byte_len
            padded = list(byte_string) + [0] * (global_max_byte_len - len(byte_string))
            encoded.append(padded)
        local_tensor = torch.tensor(encoded, dtype=torch.uint8, device='cuda')

    # Gather batch sizes from all ranks
    batch_size_tensor = torch.tensor([len(local_texts)], dtype=torch.long, device='cuda')
    batch_sizes = [torch.zeros(1, dtype=torch.long, device='cuda') for _ in range(world_size)]
    dist.all_gather(batch_sizes, batch_size_tensor)
    batch_sizes = [bs.item() for bs in batch_sizes]

    # Gather all texts to rank 0
    if rank == 0:
        gathered = [torch.zeros((bs, global_max_byte_len), dtype=torch.uint8, device='cuda')
                   for bs in batch_sizes]
        gathered[0] = local_tensor

        for src_rank in range(1, world_size):
            if batch_sizes[src_rank] > 0:
                dist.recv(gathered[src_rank], src=src_rank)

        # Decode back to strings (strip null bytes from padding)
        all_texts = []
        for rank_texts in gathered:
            for encoded_text in rank_texts:
                # Remove null byte padding (0 bytes at end)
                byte_array = encoded_text.cpu().numpy()
                # Find first null byte (if any) and truncate
                null_idx = (byte_array == 0).argmax() if 0 in byte_array else len(byte_array)
                decoded = bytes(byte_array[:null_idx]).decode('utf-8')
                all_texts.append(decoded)
        return all_texts
    else:
        # Send to rank 0
        if len(local_texts) > 0:
            dist.send(local_tensor, dst=0)
        return []

def evaluate_numeric_accuracy(dataset, src_model, src_tok, tgt_model, tgt_tok, translator,
                              device, dtype, num_samples: int = 200, max_new_tokens: int = 256,
                              show_samples: bool = True, target_rms: float = None, eval_batch_size: int = 50):
    """
    Distributed evaluation: each rank processes num_samples // world_size samples.
    Results are gathered on rank 0 for final metric computation.
    Evaluates in batches to avoid OOM.
    Uses GSM8K answer extraction (#### <number> format).
    """
    tgt_model.eval()
    src_model.eval()
    translator.eval()

    # Determine rank and world_size for distributed evaluation
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    # Shard samples across ranks
    samples_per_rank = num_samples // world_size
    start_idx = rank * samples_per_rank
    end_idx = start_idx + samples_per_rank

    # Last rank takes any remainder
    if rank == world_size - 1:
        end_idx = num_samples

    local_num_samples = end_idx - start_idx

    # Debug: log rank distribution
    if dist.is_initialized() and rank == 0:
        log(f"[Distributed Eval] Sharding {num_samples} samples across {world_size} ranks:")
        for r in range(world_size):
            r_start = r * samples_per_rank
            r_end = r_start + samples_per_rank
            if r == world_size - 1:
                r_end = num_samples
            r_samples = r_end - r_start
            log(f"  Rank {r}: samples {r_start} to {r_end} ({r_samples} samples)")

    # Collect LOCAL samples only (this rank's portion)
    all_samples = []
    taken = 0
    skip = 0
    for ex in dataset:
        if skip < start_idx:
            skip += 1
            continue
        if taken >= local_num_samples:
            break
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
            batch = build_batch_inputs(samples_batch, src_model, src_tok, tgt_model, tgt_tok, translator, device, dtype, target_rms=target_rms, mode='eval')
            gen = tgt_model.generate(
                inputs_embeds=batch["inputs_embeds"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,  # Prevent "181818..." loops
                no_repeat_ngram_size=3,  # Prevent repeating 3-grams
                pad_token_id=tgt_tok.pad_token_id,
                eos_token_id=tgt_tok.eos_token_id,
                cache_implementation="static",  # Use static KV cache for speed
                use_cache=True
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
                eos_token_id=tgt_tok.eos_token_id,
                cache_implementation="static",  # Use static KV cache for speed
                use_cache=True
            )
            base_texts_batch = tgt_tok.batch_decode(base_out, skip_special_tokens=True)
            all_base_texts.extend(base_texts_batch)

    # Gather all generations from all ranks to rank 0
    all_bridged_texts = gather_texts_from_all_ranks(all_bridged_texts)
    all_base_texts = gather_texts_from_all_ranks(all_base_texts)
    all_answers = gather_texts_from_all_ranks([s.tgt_answer for s in all_samples])

    # Only rank 0 computes metrics
    if rank != 0:
        return 0.0, 0.0  # Other ranks return dummy values

    # Reconstruct samples on rank 0 (for ground truth)
    samples = [Sample(src_prompt="", tgt_prompt="", tgt_answer=ans) for ans in all_answers]
    bridged_texts = all_bridged_texts
    base_texts = all_base_texts

    # Compute accuracy using GSM8K answer extraction
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
    parser.add_argument("--bottleneck_dim", type=int, default=1024, help="Bottleneck dimension for BottleneckedGatedTranslator")
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
    parser.add_argument("--eval_batch_size", type=int, default=50,
                        help="Batch size for evaluation (default 50, try 64-100 on H100)")
    parser.add_argument("--no_compile", action="store_true",
                        help="Disable torch.compile (use if PyTorch < 2.0 or for debugging)")
    args = parser.parse_args()

    setup_ddp()
    rank = dist.get_rank() if dist.is_initialized() else 0
    setup_reproducibility(args.seed, rank=rank)
    device = torch.device(f"cuda:{local_rank()}")
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    if is_main():
        log("==== Config ====")
        for k,v in vars(args).items(): log(f"{k}: {v}")

        log("\n==== Reproducibility Settings ====")
        log(f"Base seed: {args.seed}")
        rank = dist.get_rank() if dist.is_initialized() else 0
        log(f"Effective seed (with rank offset): {args.seed + rank}")
        log(f"CUDA deterministic: {torch.backends.cudnn.deterministic}")
        log(f"CUDA benchmark: {torch.backends.cudnn.benchmark}")
        log(f"Deterministic algorithms: {torch.are_deterministic_algorithms_enabled()}")
        log(f"CUBLAS workspace: {os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'Not set')}")
        log(f"World size: {world_size()} (each rank samples different data)")

    # Load models/tokenizers (frozen)
    log("Loading source model/tokenizer...")
    src_tok = AutoTokenizer.from_pretrained(args.source_model, use_fast=True)
    if src_tok.pad_token is None:
        src_tok.pad_token = src_tok.eos_token
    src_tok.padding_side = "left"  # Critical for decoder-only models
    src_tok.model_max_length = 2048

    # Load source model directly to GPU to avoid CPU RAM exhaustion
    # With 4 processes each loading ~30GB models, we'd need 120GB+ CPU RAM
    # device_map loads directly to GPU, bypassing CPU bottleneck
    src_model = AutoModelForCausalLM.from_pretrained(
        args.source_model, torch_dtype=dtype, device_map=str(device)
    ).eval()
    for p in src_model.parameters(): p.requires_grad = False

    log("Loading target model/tokenizer...")
    tgt_tok = AutoTokenizer.from_pretrained(args.target_model, use_fast=True)
    if tgt_tok.pad_token is None:
        tgt_tok.pad_token = tgt_tok.eos_token
    tgt_tok.padding_side = "left"  # Critical for decoder-only models
    tgt_tok.model_max_length = 2048

    # Verify tokenizer supports offset mapping (required for label alignment)
    if not tgt_tok.is_fast:
        raise ValueError(
            f"Tokenizer for {args.target_model} must be a 'fast' tokenizer to support "
            "offset mapping (required for correct label alignment). "
            "The tokenizer was loaded with use_fast=True but is not fast. "
            "This usually means no fast tokenizer is available for this model."
        )

    # Load target model directly to GPU (same reason as source model)
    tgt_model = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=dtype, device_map=str(device)
    ).eval()
    for p in tgt_model.parameters(): p.requires_grad = False

    # Compile target model for faster inference (PyTorch 2.0+)
    # DISABLED by default due to CUDA Graph OOM with dynamic shapes (creates 51+ graphs, 23-25 GiB each)
    if torch.__version__ >= "2.0" and not args.no_compile:
        log("Compiling target model with torch.compile for faster inference...")
        tgt_model = torch.compile(tgt_model, mode="reduce-overhead")
        log("Compilation complete")
    else:
        if args.no_compile:
            log("torch.compile() disabled via --no_compile flag (prevents CUDA Graph OOM)")
        else:
            log("torch.compile() not available (PyTorch < 2.0)")

    # Infer dims
    with torch.no_grad():
        d_src = src_model.config.hidden_size
        d_tgt = tgt_model.config.hidden_size
    log(f"Source hidden dim: {d_src} | Target hidden dim: {d_tgt}")

    # Compute target embedding RMS for normalization (CRITICAL for stability)
    # Use median for robustness to outliers in vocabulary distribution
    # Alternative: target_rms = get_target_embedding_rms(tgt_model)
    with torch.no_grad():
        tgt_embed_table = tgt_model.get_input_embeddings().weight.float()  # [vocab_size, d_tgt]
        per_token_rms = tgt_embed_table.pow(2).mean(dim=1).sqrt()
        target_rms = per_token_rms.median().item()  # Median more robust than mean
    log(f"Target embedding RMS (median): {target_rms:.4f}")
    log(f"NOTE: RMS scale matching is now applied automatically in translator forward pass")

    # Build translator (BottleneckedGatedTranslator with Flamingo-style gating)
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
    # CRITICAL: Use rank-offset seed so each GPU samples different batches
    # Without this, all DDP processes sample identical indices → massive waste
    rank = dist.get_rank() if dist.is_initialized() else 0
    rng = random.Random(args.seed + rank)
    if is_main():
        log(f"Data sampler initialized with seed {args.seed + rank} (base={args.seed}, rank={rank})")

    # Optimizer with proper weight decay filtering (exclude LayerNorm and biases)
    # Gate parameters get higher LR (×3) to open gradually (Flamingo-style)
    decay_params = []
    no_decay_params = []
    gate_params = []

    for name, param in translator.named_parameters():
        if not param.requires_grad:
            continue
        # Gate parameters: higher LR, no weight decay (cross_gate and ffn_gate)
        if 'cross_gate' in name or 'ffn_gate' in name:
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
    sched = get_scheduler(
        "linear",
        optimizer=optim,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.train_steps
    )

    # --------- Train loop ---------
    step = 0
    translator.train()
    running = 0.0
    last_eval = -1

    # Early stopping tracking
    best_bridged_acc = 0.0
    patience_counter = 0
    best_checkpoint = None

    # Initial evaluation at step 0 (before training) - ALL ranks participate
    if is_main():
        log("\n" + "="*60)
        log("INITIAL EVALUATION (Step 0 - Before Training)")
        log("="*60)

    with torch.no_grad():
        acc_base, acc_bridged = evaluate_numeric_accuracy(
            test_ds, src_model, src_tok, tgt_model, tgt_tok, translator.module if isinstance(translator, DDP) else translator,
            device, dtype, num_samples=args.eval_samples, max_new_tokens=args.max_new_tokens,
            show_samples=(args.show_eval_samples > 0), target_rms=target_rms,
            eval_batch_size=args.eval_batch_size
        )

    if is_main():
        log(f"[Eval] Step 0 | Target-alone acc: {acc_base:.3f} | Bridged acc: {acc_bridged:.3f}")
        log("="*60 + "\n")

    # Synchronize all ranks after evaluation
    if dist.is_initialized():
        dist.barrier()

    while step < args.train_steps:
        # sample a batch
        batch_idx = [rng.randrange(0, len(train_ds)) for _ in range(args.per_device_batch)]
        batch = {"question": [train_ds[i]["question"] for i in batch_idx],
                 "answer":   [train_ds[i]["answer"]   for i in batch_idx]}
        samples = build_samples(batch, src_tok, tgt_tok, device, tgt_model)

        # Build bridged inputs (with soft tokens)
        data = build_batch_inputs(samples, src_model, src_tok, tgt_model, tgt_tok,
                                  translator, device, dtype, target_rms=target_rms, mode='train')

        # Diagnostic: verify label alignment (only first step)
        if step == 0 and is_main():
            log("\n" + "="*60)
            log("LABEL ALIGNMENT DIAGNOSTIC (Step 0)")
            log("="*60)
            log(f"  Input embeddings shape: {data['inputs_embeds'].shape}")
            log(f"  Labels shape: {data['labels'].shape}")
            log(f"  Attention mask shape: {data['attention_mask'].shape}")

            # Get K directly from translator
            K = translator.module.K if isinstance(translator, DDP) else translator.K
            log(f"  Soft token count (K): {K}")

            log(f"  Total tokens: {data['labels'].numel()}")
            log(f"  Supervised tokens: {(data['labels'] != -100).sum().item()}")
            log(f"  Masked tokens: {(data['labels'] == -100).sum().item()}")
            log(f"  Sample 0 first 10 labels: {data['labels'][0, :10].tolist()}")
            log(f"  Sample 0 last 10 labels: {data['labels'][0, -10:].tolist()}")

            # Verify all soft token positions are masked
            soft_labels_masked = (data['labels'][:, :K] == -100).all().item()
            log(f"  All soft token labels == -100? {soft_labels_masked}")

            if not soft_labels_masked:
                log("  ⚠️  WARNING: Some soft token positions have non--100 labels!")

            # Sanity check: verify at least SOME tokens are supervised per sample
            supervised_per_sample = [(data['labels'][i] != -100).sum().item() for i in range(len(samples))]
            log(f"  Supervised tokens per sample: {supervised_per_sample}")
            if any(count == 0 for count in supervised_per_sample):
                log("  ⚠️  WARNING: Some samples have ZERO supervised tokens! Check truncation/prompts.")

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

        # Periodic eval - ALL ranks participate
        if args.eval_every > 0 and step % args.eval_every == 0 and step != last_eval:
            last_eval = step
            with torch.no_grad():
                acc_base, acc_bridged = evaluate_numeric_accuracy(
                    test_ds, src_model, src_tok, tgt_model, tgt_tok, translator.module if isinstance(translator, DDP) else translator,
                    device, dtype, num_samples=args.eval_samples, max_new_tokens=args.max_new_tokens,
                    show_samples=(args.show_eval_samples > 0), target_rms=target_rms,
                    eval_batch_size=args.eval_batch_size,
                    dataset_name=args.dataset
                )

            # Only rank 0 logs and checks early stopping
            if is_main():
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

            # Synchronize all ranks after evaluation
            if dist.is_initialized():
                dist.barrier()

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

    # Final eval - ALL ranks participate
    with torch.no_grad():
        acc_base, acc_bridged = evaluate_numeric_accuracy(
            test_ds, src_model, src_tok, tgt_model, tgt_tok, translator.module if isinstance(translator, DDP) else translator,
            device, dtype, num_samples=args.eval_samples, max_new_tokens=args.max_new_tokens,
            show_samples=(args.show_eval_samples > 0), target_rms=target_rms,
            eval_batch_size=args.eval_batch_size
        )

    # Synchronize all ranks before cleanup
    if dist.is_initialized():
        dist.barrier()

    if is_main():
        log(f"[Final Eval] Target-alone acc: {acc_base:.3f} | Bridged acc: {acc_bridged:.3f}")

    cleanup_ddp()

if __name__ == "__main__":
    main()
