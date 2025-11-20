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
from typing import List, Dict, Tuple, Optional, Set, Literal
from datetime import timedelta
from itertools import islice

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
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank_id = int(os.environ["LOCAL_RANK"])

        # Log before DDP to prove rank is alive
        print(f"[Rank {rank}/{world_size}] Starting DDP initialization...", flush=True)

        # Check GPU availability BEFORE init_process_group
        print(f"[Rank {rank}] Checking torch.cuda.is_available()...", flush=True)
        cuda_avail = torch.cuda.is_available()
        print(f"[Rank {rank}]   Result: {cuda_avail}", flush=True)
        if not cuda_avail:
            raise RuntimeError(f"[Rank {rank}] CUDA not available!")

        print(f"[Rank {rank}] Checking torch.cuda.device_count()...", flush=True)
        dev_count = torch.cuda.device_count()
        print(f"[Rank {rank}]   Result: {dev_count} devices", flush=True)
        if local_rank_id >= dev_count:
            raise RuntimeError(
                f"[Rank {rank}] LOCAL_RANK={local_rank_id} but only {dev_count} GPUs available"
            )

        # Log GPU info
        print(f"[Rank {rank}] Getting device name for GPU {local_rank_id}...", flush=True)
        device_name = torch.cuda.get_device_name(local_rank_id)
        print(f"[Rank {rank}] GPU {local_rank_id} available: {device_name}", flush=True)

        # Initialize DDP with TIMEOUT to prevent infinite hangs
        # If any rank fails to connect within 60 seconds, the whole job fails fast
        print(f"[Rank {rank}] Calling init_process_group (timeout=60s)...", flush=True)
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=60)
        )

        torch.cuda.set_device(local_rank_id)
        print(f"[Rank {rank}] DDP initialized successfully on GPU {local_rank_id}", flush=True)

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
    print(f"[Rank {rank}] Setting up reproducibility (base_seed={seed})...", flush=True)

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

    print(f"[Rank {rank}] Reproducibility setup complete (effective_seed={effective_seed})", flush=True)

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
                 use_rmsnorm: bool = True, ffn_act: str = "swiglu"):
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
                 use_rmsnorm: bool = True, ffn_act: str = "swiglu"):
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
                use_rmsnorm=use_rmsnorm,
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

        return soft_tokens

# ---------------------------
# DiT-Bridge Translator (Rectified Flow)
# ---------------------------

class TimestepEmbedding(nn.Module):
    def __init__(self, embed_dim, max_period=10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
    def forward(self, t):  # t: [B] in [0,1]
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.embed_dim % 2:
            emb = torch.cat([emb, emb.new_zeros(emb.size(0), 1)], dim=-1)
        # Cast to MLP dtype (bfloat16 when using --bf16)
        emb = emb.to(self.mlp[0].weight.dtype)
        return self.mlp(emb)

class SourceConditioner(nn.Module):
    def __init__(self, src_dim, d_cond, pool="mean", n_heads=8, dropout=0.1):
        super().__init__()
        self.pool = pool
        self.align = nn.Linear(src_dim, src_dim, bias=False)
        with torch.no_grad():
            eye = torch.eye(src_dim, dtype=self.align.weight.dtype)
            if self.align.weight.shape == eye.shape:
                self.align.weight.copy_(eye)
        if pool == "attn":
            self.query = nn.Parameter(torch.randn(1, 1, src_dim))
            self.attn = nn.MultiheadAttention(src_dim, num_heads=n_heads,
                                              dropout=dropout, batch_first=True)
            self.proj = nn.Sequential(
                nn.Linear(src_dim, 4 * d_cond),
                nn.GELU(approximate="tanh"),
                nn.Dropout(dropout),
                nn.Linear(4 * d_cond, d_cond)
            )
        else:
            self.proj = nn.Sequential(
                nn.Linear(src_dim, 4 * d_cond),
                nn.GELU(approximate="tanh"),
                nn.Dropout(dropout),
                nn.Linear(4 * d_cond, d_cond)
            )
    def forward(self, src_h, src_mask):
        # src_h: [B, T, d_src], src_mask: [B, T] with 1/0 valid/pad
        src_h = self.align(src_h)
        if self.pool == "attn":
            B = src_h.size(0)
            q = self.query.expand(B, -1, -1)  # [B,1,d_src]
            # PyTorch semantics: key_padding_mask True=PAD/ignore
            kpm = (src_mask == 0) if src_mask is not None else None
            pooled = self.attn(q, src_h, src_h, key_padding_mask=kpm, need_weights=False)[0].squeeze(1)
        else:
            if src_mask is None:
                pooled = src_h.mean(dim=1)
            else:
                # Use same dtype as src_h to avoid dtype promotion to float32
                w = src_mask.to(src_h.dtype).unsqueeze(-1)  # [B,T,1]
                pooled = (src_h * w).sum(dim=1) / (w.sum(dim=1).clamp_min(1e-6))
        # Ensure pooled matches proj dtype (handles both mean and attn pooling)
        pooled = pooled.to(self.proj[0].weight.dtype)
        return self.proj(pooled)  # [B, d_cond]

class DiTBlock(nn.Module):
    def __init__(self, model_dim, n_heads, d_cond, dropout=0.1):
        super().__init__()
        self.norm1 = nn.RMSNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(model_dim, n_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.RMSNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(4 * model_dim, model_dim),
            nn.Dropout(dropout)
        )
        # AdaLN-Zero: output 6 * model_dim, zero-init final linear
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(d_cond, 6 * model_dim, bias=True))
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)

    def forward(self, x, full_cond):  # x: [B,K,D], full_cond: [B,D]
        s_attn, b_attn, g_attn, s_mlp, b_mlp, g_mlp = self.mod(full_cond).chunk(6, dim=-1)

        # Sanity check: gates should start near 0 (due to zero-init) and gradually open
        # Monitor gate magnitude during training - should grow from ~0 to ~0.3-1.0
        # Remove this check after verifying training works
        if torch.is_grad_enabled():
            gate_mag = (g_attn.abs().mean() + g_mlp.abs().mean()) / 2
            if hasattr(self, '_gate_mag_ema'):
                self._gate_mag_ema = 0.99 * self._gate_mag_ema + 0.01 * gate_mag.item()
            else:
                self._gate_mag_ema = gate_mag.item()

        # broadcast to tokens
        s_attn = s_attn.unsqueeze(1); b_attn = b_attn.unsqueeze(1); g_attn = g_attn.unsqueeze(1)
        s_mlp  = s_mlp.unsqueeze(1);  b_mlp  = b_mlp .unsqueeze(1);  g_mlp  = g_mlp .unsqueeze(1)
        # attention path
        x_norm = self.norm1(x) * (1 + s_attn) + b_attn
        attn_out = self.attn(x_norm, x_norm, x_norm, need_weights=False)[0]
        x = x + g_attn * attn_out        # NO activation on gates (AdaLN-Zero)
        # mlp path
        x_norm = self.norm2(x) * (1 + s_mlp) + b_mlp
        mlp_out = self.mlp(x_norm)
        x = x + g_mlp * mlp_out          # NO activation on gates (AdaLN-Zero)
        return x

class DiTBridgeTranslator(nn.Module):
    """
    Dimension conventions:
    - tgt_dim: target LLM embedding dim (e.g., 4096)
    - model_dim: internal DiT width (e.g., 512)
    - Flow: tgt_dim -> model_dim (DiT) -> tgt_dim
    """
    def __init__(self, src_dim, tgt_dim, model_dim, soft_tokens,
                 depth=6, n_heads=8, dropout=0.1,
                 steps_train=2, steps_eval=4,
                 cfg_scale=0.0, cfg_dropout=0.1,
                 pool="mean", cond_dim=None):
        super().__init__()
        self.K = soft_tokens
        self.tgt_dim = tgt_dim
        self.model_dim = model_dim
        self.steps_train = steps_train
        self.steps_eval = steps_eval
        self.cfg_scale = cfg_scale
        self.cfg_dropout = cfg_dropout

        d_cond = cond_dim or model_dim
        self.to_model = nn.Linear(tgt_dim, model_dim)
        self.to_tgt   = nn.Linear(model_dim, tgt_dim)

        self.time = TimestepEmbedding(model_dim)
        self.cond = SourceConditioner(src_dim, d_cond=model_dim, pool=pool, n_heads=8, dropout=dropout)

        self.blocks = nn.ModuleList([DiTBlock(model_dim, n_heads, d_cond=model_dim, dropout=dropout)
                                     for _ in range(depth)])
        self.uncond = nn.Parameter(torch.randn(1, model_dim))  # unconditional cond for CFG
        self._last_losses = {}

    # ----- internal helpers -----
    def _forward_step(self, x_m, t, cond_vec):
        # x_m: [B,K,Dm] in model space; cond_vec: [B,Dm]; t: [B] in [0,1]
        t_vec = self.time(t)  # [B,Dm]
        full_cond = cond_vec + t_vec
        for blk in self.blocks:
            x_m = blk(x_m, full_cond)
        # predict a velocity in model space
        return x_m

    def _sample_rf(self, src_h, src_mask, steps):
        B = src_h.size(0); device = src_h.device; dtype = src_h.dtype
        # init: noise in target space, then project to model space
        x_t = torch.randn(B, self.K, self.tgt_dim, device=device, dtype=dtype)
        x_m = self.to_model(x_t)
        cond = self.cond(src_h, src_mask)  # [B,Dm]

        for i in range(steps):
            t = torch.full((B,), (i+1)/steps, device=device, dtype=torch.float32)
            if self.cfg_scale > 0:
                v_u = self._forward_step(x_m, t, self.uncond.expand(B, -1))
                v_c = self._forward_step(x_m, t, cond)
                v   = v_u + self.cfg_scale * (v_c - v_u)
            else:
                v = self._forward_step(x_m, t, cond)
            x_m = x_m + v * (1.0/steps)  # Euler step
        return self.to_tgt(x_m)  # [B,K,tgt_dim]

    def _forward_train_rf(self, src_h, src_mask, teacher_tgt):  # teacher_tgt: [B,K,tgt_dim]
        """
        Rectified Flow training: interpolate x_t linearly from noise z to data x₁;
        predict velocity v = x₁ − z; provide x̂₀ ≈ x_t + v·(1−t) to outer LM path.
        """
        B = src_h.size(0); device = src_h.device; dtype = src_h.dtype
        # project teacher & noise to model space
        x1_m = self.to_model(teacher_tgt)  # data endpoint (answer embeddings)
        z_m  = torch.randn_like(x1_m)      # noise endpoint

        # Rectified Flow: straight-line interpolation from noise (t=0) to data (t=1)
        # Sample random timestep t ~ U(0,1) and build interpolated state
        t = torch.rand(B, device=device, dtype=torch.float32)
        t3d = t.view(B, 1, 1)
        x_t = (1.0 - t3d) * z_m + t3d * x1_m  # linear interpolation

        # Velocity target: constant vector field from noise to data
        # This is the key insight of Rectified Flow - the ODE is dx/dt = v = x1 - x0
        v_target = x1_m - z_m

        # cond with optional classifier-free dropout (training only)
        cond = self.cond(src_h, src_mask)
        if self.cfg_scale > 0 and self.cfg_dropout > 0:
            drop = (torch.rand(B, 1, device=device) < self.cfg_dropout)
            cond = torch.where(drop, self.uncond.expand(B, -1), cond)

        v_pred = self._forward_step(x_t, t, cond)
        flow_loss = F.mse_loss(v_pred, v_target)

        # store side-channel loss
        self._last_losses["dit_flow"] = flow_loss.detach()

        # provide a denoised guess for outer LM path
        x0_m = x_t + v_pred * (1.0 - t3d)   # x0 ≈ x_t + v*(1-t)
        return self.to_tgt(x0_m)

    def pop_last_losses(self):
        out = self._last_losses
        self._last_losses = {}
        return out

    # ----- public API -----
    def forward(self, src_h, src_mask=None, teacher_soft_tokens=None):
        # Use PyTorch's training flag; no extra 'train' argument
        if self.training and teacher_soft_tokens is not None:
            return self._forward_train_rf(src_h, src_mask, teacher_soft_tokens)
        steps = self.steps_train if self.training else self.steps_eval
        return self._sample_rf(src_h, src_mask, steps)

# ---------------------------
# Data prep (GSM8K)
# ---------------------------

@dataclass
class EvalConfig:
    """
    Configuration for evaluation mode.
    If None passed to format_prompts, defaults to training mode (simple CoT).
    If provided, uses prebuilt few-shot prefix for evaluation.
    """
    # If None, training mode (simple CoT). If provided, use the prebuilt few-shot prefix.
    fewshot_prefix: Optional[str] = None
    mode: Literal["train_simple", "eval_8shot"] = "train_simple"

def build_gsm8k_fewshot_prefix(train_ds, k: int = 8, seed: int = 42, avoid_ids: Optional[Set[int]] = None) -> str:
    """
    Prebuild the fixed 8-shot few-shot prefix ONCE for evaluation.

    8-shot Chain-of-Thought is the de-facto GSM8K evaluation style used in many reports
    and matches lm-evaluation-harness `gsm8k_cot` behavior.
    (Kojima et al., 2022; EleutherAI lm-eval harness)

    Args:
        train_ds: GSM8K train dataset
        k: Number of exemplars (default 8)
        seed: Random seed for reproducibility (default 42)
        avoid_ids: Optional set of indices to exclude from sampling

    Returns:
        Prebuilt string prefix with header + 8 exemplars (ready for appending target Q)
    """
    header = "Answer the following questions step by step and end with '#### <number>'.\n\n"
    rng = random.Random(seed)
    pool = [i for i in range(len(train_ds)) if (avoid_ids is None or i not in avoid_ids)]
    idxs = rng.sample(pool, k)
    exemplars = []
    for j in idxs:
        q = train_ds[j]["question"].strip()
        full = train_ds[j]["answer"].strip()
        final = extract_final_answer(full)  # existing #### extractor
        rationale = full.rsplit('####', 1)[0].strip() if '####' in full else full
        exemplars.append((q, rationale, final))
    fewshot = "\n\n".join(f"Q: {q}\nA: {r}\n#### {ans}" for (q, r, ans) in exemplars)
    return header + fewshot


def compute_format_penalty(pred_logits: torch.Tensor, labels: torch.Tensor, tokenizer) -> torch.Tensor:
    """Penalize sequences that fail to emit the #### marker."""
    device = pred_logits.device
    preds = pred_logits.argmax(dim=-1)
    penalties = []
    for pred_row, label_row in zip(preds, labels):
        token_ids = [pid.item() for pid, lid in zip(pred_row, label_row) if lid != -100]
        if not token_ids:
            penalties.append(1.0)
            continue
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        penalties.append(0.0 if "####" in text else 1.0)
    if not penalties:
        return torch.tensor(0.0, device=device)
    return torch.tensor(sum(penalties)/len(penalties), device=device)

def format_prompts(problem: str, cfg: Optional[EvalConfig] = None) -> Tuple[str, str]:
    """
    Build comparable prompts for source and target.
    Pure function - no global state.

    Args:
        problem: The question to solve
        cfg: Optional evaluation config. If None, uses training mode (simple CoT).

    Returns:
        (src_prompt, tgt_prompt) - identical for both models
    """
    if cfg is None or cfg.mode == "train_simple":
        # Training: simple CoT template
        prompt = (
            "Solve the problem step by step, then end your final line with '#### <number>'.\n\n"
            "Problem:\n"
            f"{problem.strip()}\n\n"
            "Answer:"
        )
    else:
        # Evaluation: append target Q after prebuilt 8-shot prefix
        assert cfg.fewshot_prefix is not None and cfg.mode == "eval_8shot", \
            "EvalConfig must have fewshot_prefix set for eval_8shot mode"
        prompt = (
            cfg.fewshot_prefix
            + f"\n\nQ: {problem.strip()}\nA: Let's think step by step.\n\nAnswer:"
        )
    return prompt, prompt  # identical for source/target

def build_samples(batch, src_tok, tgt_tok, device, tgt_model,
                  answer_as_label=True, cfg: Optional[EvalConfig] = None) -> List[Sample]:
    """
    Prepare per-example structures - NO TOKENIZATION HERE.
    Tokenization happens in build_batch_inputs for correct alignment.

    Args:
        batch: Dict with "question" and "answer" keys
        cfg: Optional evaluation config for 8-shot prompts (None = training mode)
    """
    samples: List[Sample] = []
    for prob, sol in zip(batch["question"], batch["answer"]):
        src_prompt, tgt_prompt = format_prompts(prob, cfg)
        samples.append(Sample(
            src_prompt=src_prompt,
            tgt_prompt=tgt_prompt,
            tgt_answer=sol.strip()
        ))
    return samples

# ---------------------------
# Collation building inputs_embeds with soft tokens
# ---------------------------

def _pad_or_truncate_to_k(x, K):
    """
    Pad embeddings to K tokens (truncate if exceeding K).
    x: [B, T, d]
    Returns: [B, K, d]
    """
    B, T, d = x.shape
    if T >= K:
        return x[:, :K, :]
    y = x.new_zeros(B, K, d)
    y[:, :T, :] = x
    return y

def build_batch_inputs(samples: List[Sample],
                       src_model, src_tok,
                       tgt_model, tgt_tok,
                       translator,
                       device, dtype,
                       target_rms: float = None,
                       mode: str = 'train',
                       args = None):
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
        src_enc = src_tok([s.src_prompt for s in samples], return_tensors="pt", padding=True, truncation=True, max_length=8192)
        src_enc = {k: v.to(device) for k, v in src_enc.items()}
        src_out = src_model(**src_enc, output_hidden_states=True)
        # last hidden states: [B, T_src, d_src]
        src_h = src_out.hidden_states[-1].to(dtype)

        # Source mask
        src_mask = src_enc["attention_mask"]

    # Translator (grad flows)
    # For DiT bridge in training mode, provide teacher embeddings
    if args is not None and args.bridge == "dit" and mode == "train":
        K = translator.K if hasattr(translator, 'K') else args.soft_tokens
        with torch.no_grad():
            # Teacher embedding choice: answer vs. prompt
            # - answer (default): DiT learns to denoise towards answer embeddings (output-space alignment)
            #   This is the Transfusion-style approach: teach the diffusion model the target distribution.
            # - prompt: DiT learns to denoise towards prompt embeddings (conditioning alignment)
            #   This may help the DiT learn better representations of the question/context.
            if args.dit_teacher == "answer":
                # Answer teacher: use answer text (what the model generates)
                texts = [s.tgt_answer if s.tgt_answer.strip() else (tgt_tok.bos_token or " ") for s in samples]
            else:
                # Prompt teacher: use prompt text (the question/context)
                texts = [s.tgt_prompt if s.tgt_prompt.strip() else (tgt_tok.bos_token or " ") for s in samples]

            enc = tgt_tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=8192)
            enc = {k: v.to(device) for k, v in enc.items()}
            emb = tgt_model.get_input_embeddings()(enc["input_ids"]).to(dtype)  # [B, T_text, d_tgt]
            teacher_soft = _pad_or_truncate_to_k(emb, K)                         # [B, K, d_tgt]
        soft_tokens = translator(src_h, src_mask, teacher_soft_tokens=teacher_soft)
    else:
        soft_tokens = translator(src_h, src_mask)  # [B, K, d_tgt]

    # Apply RMS matching to prevent logit collapse (CRITICAL for stability)
    if target_rms is not None:
        soft_tokens = apply_rms_matching(soft_tokens, target_rms)

    prompt_alignment_loss = torch.tensor(0.0, device=device, dtype=dtype)
    if mode == 'train':
        prompt_ids = tgt_tok(
            [s.tgt_prompt for s in samples],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=K
        )
        prompt_ids = {k: v.to(device) for k, v in prompt_ids.items()}
        prompt_embeds = tgt_model.get_input_embeddings()(prompt_ids["input_ids"]).to(dtype)
        prompt_embeds = _pad_or_truncate_to_k(prompt_embeds, K)
        prompt_mask = prompt_ids["attention_mask"]
        if prompt_mask.size(1) < K:
            pad = torch.zeros(prompt_mask.size(0), K - prompt_mask.size(1), device=device, dtype=prompt_mask.dtype)
            prompt_mask = torch.cat([prompt_mask, pad], dim=1)
        prompt_mask = prompt_mask.unsqueeze(-1)
        diff = (soft_tokens - prompt_embeds) * prompt_mask
        denom = prompt_mask.sum().clamp_min(1.0)
        prompt_alignment_loss = diff.pow(2).sum() / denom

    # Target tokenization
    # Training: answer only (model must use soft tokens to understand question)
    # Eval: start token only (model generates full answer)
    starter = tgt_tok.bos_token or " "  # BOS token if available, else space
    format_prompt = "\nAnswer the question above and end with '#### <number>'."
    decode_max_len = getattr(args, "decode_max_length", 8192)
    if mode == 'train':
        tgt_texts = [s.tgt_answer for s in samples]  # Answer only for teacher forcing
        decode_prompt_texts = None
    elif mode == 'decode':
        prompt_texts = [samples[i].tgt_prompt + format_prompt for i in range(len(samples))]
        tgt_texts = [
            prompt_texts[i] + ("\n\n" if not samples[i].tgt_answer.startswith("\n") else "") + samples[i].tgt_answer
            for i in range(len(samples))
        ]
        decode_prompt_texts = prompt_texts
    else:
        prompt_mode = getattr(args, "eval_prompt_mode", "soft_plus_text")
        if args is not None:
            curriculum_steps = getattr(args, "soft_only_curriculum_steps", 0)
            curr_step = getattr(args, "_curriculum_eval_step", None)
            if (prompt_mode == "soft_only" and curriculum_steps > 0
                    and isinstance(curr_step, int) and curr_step < curriculum_steps):
                prompt_mode = "soft_plus_text"
        if prompt_mode == "soft_plus_text":
            tgt_texts = [samples[i].tgt_prompt + format_prompt for i in range(len(samples))]
        else:  # soft_only
            tgt_texts = [starter + format_prompt for _ in samples]
        decode_prompt_texts = None

    max_len = decode_max_len if (mode == 'decode') else 8192
    tgt_batch = tgt_tok(
        tgt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len
    )
    tgt_batch = {k: v.to(device) for k, v in tgt_batch.items()}

    prompt_token_lengths = None
    if mode == 'decode' and decode_prompt_texts is not None:
        prompt_batch = tgt_tok(
            decode_prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len
        )
        prompt_batch = {k: v.to(device) for k, v in prompt_batch.items()}
        prompt_token_lengths = prompt_batch["attention_mask"].sum(dim=1)

    # Target input embeddings
    with torch.no_grad():
        embed = tgt_model.get_input_embeddings()  # nn.Embedding
        tgt_embeds = embed(tgt_batch["input_ids"]).to(dtype)  # [B, T_tgt, d_tgt]

    # Concatenate soft tokens in front
    K = soft_tokens.size(1)
    T_tgt = tgt_embeds.size(1)
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
    elif mode == 'decode':
        for i in range(B):
            if prompt_token_lengths is not None:
                keep_from = int(prompt_token_lengths[i].item())
                labels[i, :keep_from] = -100
            labels[i, tgt_batch["attention_mask"][i] == 0] = -100
    else:
        # Eval mode: dummy labels (shape must match T_tgt before K-prepend)
        labels = tgt_batch["input_ids"].new_full((B, T_tgt), -100)

    # Prepend -100 for K soft tokens (no loss on them)
    labels = torch.cat([
        torch.full((B, K), -100, dtype=labels.dtype, device=device),
        labels
    ], dim=1)  # [B, K+T]

    # Shape sanity checks
    assert inputs_embeds.shape[:2] == attn_mask.shape, "mask length mismatch"
    assert inputs_embeds.shape[:2] == labels.shape, "labels length mismatch"

    out = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attn_mask,
        "labels": labels,
        "K": K,
        "prompt_alignment_loss": prompt_alignment_loss,
        "text_lengths": tgt_batch["attention_mask"].sum(dim=1),
        "soft_token_mean": soft_tokens.mean(dim=1),
        "src_prompt_mean": src_h.mean(dim=1),
        "soft_tokens_full": soft_tokens
    }
    if 'teacher_soft' in locals():
        out["teacher_soft"] = teacher_soft.detach()
    if args is not None and getattr(args, "return_soft_tokens", False):
        out["soft_tokens_full"] = soft_tokens
    return out

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
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        gathered: List[List[str]] = [None] * world_size  # type: ignore
        dist.gather_object(local_texts, object_gather_list=gathered)
        # Flatten preserving rank order
        return [t for sub in gathered if sub for t in sub]
    else:
        dist.gather_object(local_texts, object_gather_list=None)
        return []

def evaluate_numeric_accuracy(dataset, src_model, src_tok, tgt_model, tgt_tok, translator,
                              device, dtype, train_ds, num_samples: int = 200, max_new_tokens: int = 256,
                              show_samples: bool = True, target_rms: float = None, eval_batch_size: int = 50,
                              args = None, eval_step: Optional[object] = None):
    """
    Distributed evaluation: each rank processes num_samples // world_size samples.
    Results are gathered on rank 0 for final metric computation.
    Evaluates in batches to avoid OOM.
    Uses GSM8K answer extraction (#### <number> format).

    Evaluation automatically uses 8-shot CoT prompting with fixed seed=42 exemplars.
    """
    tgt_model.eval()
    src_model.eval()
    translator.eval()

    if args is not None:
        if isinstance(eval_step, int):
            args._curriculum_eval_step = eval_step
        else:
            args._curriculum_eval_step = getattr(args, "train_steps", 0)

    # Prebuild 8-shot CoT prefix once for all evaluation samples
    if args is not None and getattr(args, "fewshot_prefix", None):
        fewshot_prefix = args.fewshot_prefix
    else:
        fewshot_prefix = build_gsm8k_fewshot_prefix(train_ds, k=8, seed=42)
    eval_cfg = EvalConfig(fewshot_prefix=fewshot_prefix, mode="eval_8shot")

    # Determine rank and world_size for distributed evaluation
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    # Clamp num_samples to dataset length
    num_samples = min(num_samples, len(dataset))

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
    shard = islice(dataset, start_idx, end_idx)

    all_samples = []
    for ex in shard:
        all_samples.extend(build_samples({"question":[ex["question"]], "answer":[ex["answer"]]},
                                         src_tok, tgt_tok, device, tgt_model, cfg=eval_cfg))

    # Process in batches to avoid OOM
    all_bridged_texts = []
    all_base_texts = []
    all_source_texts = []

    for start_idx in range(0, len(all_samples), eval_batch_size):
        end_idx = min(start_idx + eval_batch_size, len(all_samples))
        samples_batch = all_samples[start_idx:end_idx]

        # Build bridged inputs (uses inputs_embeds)
        with torch.inference_mode():
            batch = build_batch_inputs(samples_batch, src_model, src_tok, tgt_model, tgt_tok, translator, device, dtype, target_rms=target_rms, mode='eval', args=args)
            gen = tgt_model.generate(
                inputs_embeds=batch["inputs_embeds"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tgt_tok.pad_token_id,
                eos_token_id=tgt_tok.eos_token_id,
                use_cache=True
            )
            bridged_texts_batch = tgt_tok.batch_decode(gen, skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=False)
            all_bridged_texts.extend(bridged_texts_batch)

        # Target-alone baseline (Llama only)
        with torch.inference_mode():
            prompts = [s.tgt_prompt for s in samples_batch]
            enc = tgt_tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(device)
            base_out = tgt_model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tgt_tok.pad_token_id,
                eos_token_id=tgt_tok.eos_token_id,
                use_cache=True
            )
            base_texts_batch = tgt_tok.batch_decode(base_out, skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False)
            all_base_texts.extend(base_texts_batch)

        # Source-only baseline (Mistral only)
        with torch.inference_mode():
            prompts = [s.src_prompt for s in samples_batch]
            src_enc = src_tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(device)
            src_out = src_model.generate(
                **src_enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=src_tok.pad_token_id,
                eos_token_id=src_tok.eos_token_id,
                use_cache=True
            )
            source_texts_batch = src_tok.batch_decode(src_out, skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=False)
            all_source_texts.extend(source_texts_batch)

        # Clear CUDA cache after each batch to reduce fragmentation
        torch.cuda.empty_cache()

    # Gather all generations from all ranks to rank 0
    all_bridged_texts = gather_texts_from_all_ranks(all_bridged_texts)
    all_base_texts = gather_texts_from_all_ranks(all_base_texts)
    all_answers = gather_texts_from_all_ranks([s.tgt_answer for s in all_samples])
    all_questions = gather_texts_from_all_ranks([s.tgt_prompt for s in all_samples])
    all_source_texts = gather_texts_from_all_ranks(all_source_texts)

    # Only rank 0 computes metrics
    if rank != 0:
        return 0.0, 0.0, 0.0  # Other ranks return dummy values

    # Reconstruct samples on rank 0 (for ground truth)
    samples = [Sample(src_prompt="", tgt_prompt=prompt, tgt_answer=ans)
               for prompt, ans in zip(all_questions, all_answers)]
    bridged_texts = all_bridged_texts
    base_texts = all_base_texts
    source_texts = all_source_texts

    eval_label = str(eval_step) if eval_step is not None else "unspecified"
    sample_records: List[Dict[str, str]] = []
    correct_source = 0
    correct_baseline = 0
    correct_bridged = 0

    for idx, (sample, base_text, bridged_text, source_text) in enumerate(zip(samples, base_texts, bridged_texts, source_texts)):
        question_text = sample.tgt_prompt.strip()
        gold_full_text = (sample.tgt_prompt + " " + sample.tgt_answer).strip()
        # FIX: Extract gold answer from tgt_answer only, not from prompt+answer
        # (prompt contains 8-shot examples with "####" markers that would be extracted first)
        gold_answer = extract_final_answer(sample.tgt_answer)

        # Truncate outputs at first new "Q:" to prevent continuation behavior
        # Models sometimes generate additional Q&A pairs after answering the test question
        def clean_generation(text: str, prompt_prefix: Optional[str] = None) -> str:
            """Strip the original prompt and drop any follow-up questions."""
            cleaned = text.strip()
            if prompt_prefix:
                prefix = prompt_prefix.strip()
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].lstrip()

            # Find second "\nQ:" occurrence (first is the real question)
            matches = [m.start() for m in re.finditer(r'\nQ:', cleaned)]
            if len(matches) >= 2:
                cleaned = cleaned[:matches[1]]
            return cleaned.strip()

        target_full = clean_generation(base_text, sample.tgt_prompt)
        target_answer = extract_final_answer(target_full)

        source_full = clean_generation(source_text, sample.tgt_prompt)
        source_answer = extract_final_answer(source_full)

        bridged_full = clean_generation(bridged_text)
        bridged_answer = extract_final_answer(bridged_full)

        sample_records.append({
            "eval_step": eval_label,
            "index": idx,
            "question": question_text,
            "gold_full_text": gold_full_text,
            "gold_extracted": gold_answer,
            "source_full": source_full,
            "source_extracted": source_answer,
            "target_full": target_full,
            "target_extracted": target_answer,
            "bridged_full": bridged_full,
            "bridged_extracted": bridged_answer,
        })

        correct_source += int(source_answer == gold_answer)
        correct_baseline += int(target_answer == gold_answer)
        correct_bridged += int(bridged_answer == gold_answer)

    total_samples = max(1, len(sample_records))
    acc_source = correct_source / total_samples
    acc_baseline = correct_baseline / total_samples
    acc_bridged = correct_bridged / total_samples

    # Print sample outputs for inspection (only from main process)
    if show_samples and num_samples >= 3:
        log("\n" + "="*60)
        log("SAMPLE OUTPUTS (first 3 examples):")
        log("="*60)
        for i in range(min(3, len(sample_records))):
            rec = sample_records[i]
            log(f"\n--- Example {i+1} ---")
            log("Question (full prompt):")
            log(rec["question"] if rec["question"] else "[empty prompt]")
            log(f"Gold answer: {rec['gold_extracted']}")

            log(f"Source-alone (extracted): {rec['source_extracted']}")
            log("Source-alone full output:")
            log(rec["source_full"] if rec["source_full"] else "[empty output]")

            log(f"Target-alone (extracted): {rec['target_extracted']}")
            log("Target-alone full output:")
            log(rec["target_full"] if rec["target_full"] else "[empty output]")

            log(f"Bridged (extracted): {rec['bridged_extracted']}")
            log("Bridged full output:")
            log(rec["bridged_full"] if rec["bridged_full"] else "[empty output]")
        log("="*60 + "\n")

    log_dir = getattr(args, "log_dir", "") if args is not None else ""
    if log_dir and len(sample_records) > 0:
        os.makedirs(log_dir, exist_ok=True)
        safe_label = str(eval_label).replace(" ", "_")
        jsonl_path = os.path.join(log_dir, f"eval_samples_step_{safe_label}.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in sample_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log(f"[Eval] Saved sample outputs to {jsonl_path}")

    return acc_source, acc_baseline, acc_bridged

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
        tgt_embed = target_model.get_input_embeddings().weight.float()
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
            torch.ones(soft_tokens.size(0), soft_tokens.size(1),
                      device=prompt_mask.device, dtype=prompt_mask.dtype),
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
    parser.add_argument("--soft_tokens", type=int, default=-1,
                        help="Number of soft tokens to generate (<=0 chooses full prompt length)")
    parser.add_argument("--eval_prompt_mode", type=str,
                        choices=["soft_only", "soft_plus_text"], default="soft_plus_text",
                        help="How much literal text to feed Llama during evaluation.")
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
    parser.add_argument("--log_dir", type=str, default="",
                        help="Directory to write logs/artifacts (e.g., eval sample JSONL)")
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
    parser.add_argument("--decode_loss_weight", type=float, default=0.0,
                        help="Weight for decode-aware supervision (0 disables)")
    parser.add_argument("--decode_interval", type=int, default=50,
                        help="Frequency (in steps) to apply decode-aware supervision")
    parser.add_argument("--decode_samples", type=int, default=1,
                        help="Number of samples per rank for decode-aware supervision (0 = use full batch)")
    parser.add_argument("--decode_max_length", type=int, default=4096,
                        help="Max token length for decode-aware supervision inputs")
    parser.add_argument("--kl_max_length", type=int, default=512,
                        help="Maximum token length when encoding answers for KL baseline")
    parser.add_argument("--kl_tokens", type=int, default=20,
                        help="Number of answer tokens to compare in KL alignment")
    parser.add_argument("--prompt_alignment_weight", type=float, default=0.001,
                        help="Scaling factor for prompt-alignment loss")
    parser.add_argument("--soft_only_curriculum_steps", type=int, default=0,
                        help="Keep literal prompt until this step when eval_prompt_mode=soft_only")
    parser.add_argument("--prompt_contrast_weight", type=float, default=0.0,
                        help="Extra weight for prompt contrastive InfoNCE loss")
    parser.add_argument("--aux_probe_weight", type=float, default=0.0,
                        help="Weight for auxiliary probe aligning pooled soft tokens to prompt embeddings")
    parser.add_argument("--format_loss_weight", type=float, default=0.1,
                        help="Scaling factor for format penalty")
    parser.add_argument("--token_alignment_weight", type=float, default=0.0,
                        help="Align soft tokens to teacher embeddings (addresses vocab/pos mismatch)")
    parser.add_argument("--no_compile", action="store_true",
                        help="Disable torch.compile (use if PyTorch < 2.0 or for debugging)")
    # DiT-Bridge options
    parser.add_argument("--bridge", type=str, choices=["cross", "dit"], default="cross")
    parser.add_argument("--dit_dim", type=int, default=512, help="DiT internal width (project to/from d_tgt)")
    parser.add_argument("--dit_depth", type=int, default=6)
    parser.add_argument("--dit_heads", type=int, default=8, help="512/8 = 64 dims per head")
    parser.add_argument("--dit_steps_train", type=int, default=2, help="Small for first ablation")
    parser.add_argument("--dit_steps_eval", type=int, default=4)
    parser.add_argument("--dit_dropout", type=float, default=0.1)
    parser.add_argument("--dit_cfg", type=float, default=0.0, help="CFG scale (off by default)")
    parser.add_argument("--dit_drop_cond", type=float, default=0.1, help="p(cond drop) when training if CFG>0")
    parser.add_argument("--dit_cfg_dropout", type=float, default=None, help="Alias for --dit_drop_cond (deprecated)")
    parser.add_argument("--dit_pool", type=str, choices=["mean", "attn"], default="mean")
    parser.add_argument("--dit_cond_dim", type=int, default=512, help="Conditioner MLP width")
    parser.add_argument("--dit_loss_weight", type=float, default=0.1)
    parser.add_argument("--dit_loss_warmup", type=int, default=0,
                        help="Warm up dit_loss_weight linearly over the first N steps (0=off).")
    parser.add_argument("--dit_teacher", type=str,
                        choices=["answer", "prompt"],
                        default="answer",
                        help="Supervision for teacher_tgt: 'answer' (current) or 'prompt' (prefix).")
    args = parser.parse_args()

    # Handle deprecated alias
    if args.dit_cfg_dropout is not None:
        args.dit_drop_cond = args.dit_cfg_dropout

    # Early CUDA diagnostics (before DDP)
    print("=" * 60, flush=True)
    print("EARLY CUDA DIAGNOSTICS (before DDP setup)", flush=True)
    print("=" * 60, flush=True)
    print(f"torch.cuda.is_available() check...", flush=True)
    cuda_available = torch.cuda.is_available()
    print(f"  Result: {cuda_available}", flush=True)
    if cuda_available:
        print(f"torch.cuda.device_count() check...", flush=True)
        device_count = torch.cuda.device_count()
        print(f"  Result: {device_count} GPUs", flush=True)
        for i in range(device_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}", flush=True)
    else:
        print("  WARNING: CUDA not available!", flush=True)
    print("=" * 60, flush=True)
    print("", flush=True)

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
    src_tok.model_max_length = 8192

    # Load source model directly to GPU to avoid CPU RAM exhaustion
    # With 4 processes each loading ~30GB models, we'd need 120GB+ CPU RAM
    # device_map loads directly to GPU, bypassing CPU bottleneck
    src_model = AutoModelForCausalLM.from_pretrained(
        args.source_model,
        torch_dtype=dtype,
        device_map={"": local_rank()},
        low_cpu_mem_usage=True,
    ).eval()
    for p in src_model.parameters(): p.requires_grad = False

    log("Loading target model/tokenizer...")
    tgt_tok = AutoTokenizer.from_pretrained(args.target_model, use_fast=True)
    if tgt_tok.pad_token_id is None and tgt_tok.eos_token_id is not None:
        tgt_tok.pad_token = tgt_tok.eos_token
    tgt_tok.padding_side = "left"  # Critical for decoder-only models
    tgt_tok.model_max_length = 8192

    if args.soft_tokens <= 0:
        # Use full prompt length but cap to 2048 to prevent OOM
        args.soft_tokens = min(src_tok.model_max_length, 2048)
        log(f"Soft tokens not specified; capping to {args.soft_tokens} tokens")

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
        args.target_model,
        torch_dtype=dtype,
        device_map={"": local_rank()},
        low_cpu_mem_usage=True,
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

    # Set generation config once for consistency
    gen_cfg = tgt_model.generation_config
    gen_cfg.pad_token_id = tgt_tok.pad_token_id
    gen_cfg.eos_token_id = tgt_tok.eos_token_id

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
    log(f"NOTE: RMS scale matching is applied during batch collation (before concat)")

    # Build translator
    if args.bridge == "cross":
        translator = BottleneckedGatedTranslator(
            src_dim=d_src,
            tgt_dim=d_tgt,
            bottleneck_dim=args.bottleneck_dim,
            soft_tokens=args.soft_tokens,
            depth=args.depth,
            n_heads=args.heads
        ).to(device=device, dtype=dtype)
        log(f"Bridge: {args.bridge} (BottleneckedGatedTranslator)")
    else:
        translator = DiTBridgeTranslator(
            src_dim=d_src,
            tgt_dim=d_tgt,
            model_dim=args.dit_dim,
            soft_tokens=args.soft_tokens,
            depth=args.dit_depth,
            n_heads=args.dit_heads,
            dropout=args.dit_dropout,
            steps_train=args.dit_steps_train,
            steps_eval=args.dit_steps_eval,
            cfg_scale=args.dit_cfg,
            cfg_dropout=args.dit_drop_cond,
            pool=args.dit_pool,
            cond_dim=args.dit_cond_dim
        ).to(device=device, dtype=dtype)
        log(f"Bridge: {args.bridge} (DiTBridgeTranslator)")
        log(f"  Using DiTBridge: dim={args.dit_dim}, depth={args.dit_depth}, heads={args.dit_heads}, "
            f"steps(train/eval)={args.dit_steps_train}/{args.dit_steps_eval}, cfg={args.dit_cfg}, "
            f"pool={args.dit_pool}, cond_dim={args.dit_cond_dim}")

    # Count parameters
    param_count = sum(p.numel() for p in translator.parameters())
    log(f"Translator parameters: {param_count / 1e6:.1f}M")

    # DDP only on translator (models are frozen)
    # find_unused_parameters=True needed for DiT: training uses _forward_train_rf (no CFG),
    # while eval uses _sample_rf (may use CFG/uncond params)
    if dist.is_initialized():
        translator = DDP(translator, device_ids=[local_rank()], output_device=local_rank(), find_unused_parameters=True)

    # Data
    log(f"Loading {args.dataset.upper()}...")
    train_prompt_cfg = None
    if args.dataset == "gsm8k":
        ds = load_dataset("gsm8k", "main")
        train_ds = ds["train"]
        test_ds  = ds["test"]
        fewshot_prefix = build_gsm8k_fewshot_prefix(train_ds, k=8, seed=42)
        args.fewshot_prefix = fewshot_prefix
        train_prompt_cfg = EvalConfig(fewshot_prefix=fewshot_prefix, mode="eval_8shot")
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
        # Gate parameters: higher LR, no weight decay (cross_gate, ffn_gate, sa_gate)
        if 'cross_gate' in name or 'ffn_gate' in name or 'sa_gate' in name:
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
    optim = torch.optim.AdamW(optim_groups, betas=(0.9, 0.98), eps=1e-8, fused=True)
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
        acc_source, acc_base, acc_bridged = evaluate_numeric_accuracy(
            test_ds, src_model, src_tok, tgt_model, tgt_tok, translator.module if isinstance(translator, DDP) else translator,
            device, dtype, train_ds, num_samples=args.eval_samples, max_new_tokens=args.max_new_tokens,
            show_samples=(args.show_eval_samples > 0), target_rms=target_rms,
            eval_batch_size=args.eval_batch_size, args=args, eval_step=0
        )

    if is_main():
        log(f"[Eval] Step 0 | Source-alone acc: {acc_source:.3f} | Target-alone acc: {acc_base:.3f} | Bridged acc: {acc_bridged:.3f}")
        log("="*60 + "\n")

    # Synchronize all ranks after evaluation
    if dist.is_initialized():
        dist.barrier()

    while step < args.train_steps:
        args._current_step = step
        # sample a batch
        batch_idx = [rng.randrange(0, len(train_ds)) for _ in range(args.per_device_batch)]
        batch = {"question": [train_ds[i]["question"] for i in batch_idx],
                 "answer":   [train_ds[i]["answer"]   for i in batch_idx]}
        samples = build_samples(batch, src_tok, tgt_tok, device, tgt_model, cfg=train_prompt_cfg)

        # Build bridged inputs (with soft tokens)
        data = build_batch_inputs(samples, src_model, src_tok, tgt_model, tgt_tok,
                                  translator, device, dtype, target_rms=target_rms, mode='train', args=args)

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

        # Extract metadata from data dict (not valid model forward args)
        K_soft = data.pop("K", 0)
        prompt_alignment_loss = data.pop("prompt_alignment_loss", torch.tensor(0.0, device=device, dtype=dtype))
        text_lengths = data.pop("text_lengths", None)

        # Forward (compute NLL loss on target)
        out = tgt_model(**data)
        nll_loss = out.loss

        # KL consistency loss (align first textual tokens between bridged and baseline)
        kl_loss = torch.tensor(0.0, device=device, dtype=dtype)
        if step > args.warmup_steps:
            with torch.no_grad():
                tgt_answers = [s.tgt_answer for s in samples]
                kl_enc = tgt_tok(
                    tgt_answers,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.kl_max_length
                ).to(device)
                baseline_out = tgt_model(**kl_enc)
                baseline_logits = baseline_out.logits
            available = out.logits.size(1) - K_soft
            num_compare = min(args.kl_tokens, baseline_logits.size(1), available)
            if num_compare > 0:
                bridged_slice = out.logits[:, K_soft:K_soft + num_compare, :]
                baseline_slice = baseline_logits[:, :num_compare, :]
                kl_loss = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(bridged_slice.float(), dim=-1),
                    torch.nn.functional.softmax(baseline_slice.float(), dim=-1),
                    reduction="batchmean"
                )

        contrastive_loss = torch.tensor(0.0, device=device, dtype=dtype)
        aux_probe_loss = torch.tensor(0.0, device=device, dtype=dtype)
        if (step > args.warmup_steps // 2
                and (args.info_nce_weight > 0 or args.prompt_contrast_weight > 0 or args.aux_probe_weight > 0)):
            with torch.no_grad():
                tgt_prompts = [s.tgt_prompt for s in samples]
                tgt_enc = tgt_tok(tgt_prompts, return_tensors="pt", padding=True, truncation=True, max_length=K_soft).to(device)
                tgt_embeds_full = tgt_model.get_input_embeddings()(tgt_enc["input_ids"])
                tgt_pooled = tgt_embeds_full.mean(dim=1)

            soft_pooled = data["soft_tokens_full"].mean(dim=1)
            soft_norm = torch.nn.functional.normalize(soft_pooled.float(), dim=-1)
            tgt_norm = torch.nn.functional.normalize(tgt_pooled.float(), dim=-1)
            temperature = 0.07
            logits_contrastive = soft_norm @ tgt_norm.T / temperature
            labels_contrastive = torch.arange(logits_contrastive.size(0), device=device)
            contrastive_loss = torch.nn.functional.cross_entropy(logits_contrastive, labels_contrastive)

            if args.aux_probe_weight > 0:
                aux_probe_loss = torch.nn.functional.mse_loss(soft_pooled, tgt_pooled.float())

        format_loss = compute_format_penalty(out.logits.detach(), data['labels'], tgt_tok)
        # prompt_alignment_loss already extracted at line 1741

        decode_loss = torch.tensor(0.0, device=device, dtype=dtype)
        if (args.decode_loss_weight > 0 and args.decode_interval > 0
                and step > args.warmup_steps
                and (step % args.decode_interval == 0)):
            run_decode = True
            decode_world_size = 1
            rank = 0
            if dist.is_initialized():
                decode_world_size = dist.get_world_size()
                rank = dist.get_rank()
                run_decode = (rank == 0)
            if run_decode:
                decode_subset = samples
                if args.decode_samples > 0 and len(samples) > args.decode_samples:
                    idxs = rng.sample(range(len(samples)), args.decode_samples)
                    decode_subset = [samples[i] for i in idxs]
                decode_data = build_batch_inputs(
                    decode_subset, src_model, src_tok, tgt_model, tgt_tok,
                    translator, device, dtype, target_rms=target_rms, mode='decode', args=args
                )
                decode_out = tgt_model(**decode_data)
                decode_loss = decode_out.loss * decode_world_size

        loss = nll_loss \
            + 0.03 * kl_loss \
            + args.info_nce_weight * contrastive_loss \
            + args.prompt_contrast_weight * contrastive_loss \
            + args.prompt_alignment_weight * prompt_alignment_loss \
            + args.format_loss_weight * format_loss \
            + args.decode_loss_weight * decode_loss \
            + args.aux_probe_weight * aux_probe_loss

        token_alignment_loss = torch.tensor(0.0, device=device, dtype=dtype)
        if args.token_alignment_weight > 0 and "teacher_soft" in data:
            token_alignment_loss = torch.nn.functional.mse_loss(
                data["soft_tokens_full"], data["teacher_soft"]
            )
            loss = loss + args.token_alignment_weight * token_alignment_loss

        # Add DiT flow loss if using DiT bridge
        # The args.dit_loss_weight balances flow loss vs LM loss (default 0.1)
        # Too high: DiT dominates, model ignores LM signal
        # Too low: DiT doesn't learn proper denoising
        # Optional linear warmup over first N steps via args.dit_loss_warmup
        if args.bridge == "dit":
            # Get the module (unwrap DDP if needed)
            module = translator.module if isinstance(translator, DDP) else translator
            aux = getattr(module, "pop_last_losses", lambda: {})()
            dit_flow_loss = aux.get("dit_flow", torch.tensor(0.0, device=loss.device, dtype=loss.dtype))

            # Apply warmup if configured
            weight = args.dit_loss_weight
            if args.dit_loss_warmup and step <= args.dit_loss_warmup:
                weight = weight * float(step) / float(max(1, args.dit_loss_warmup))

            loss = loss + weight * dit_flow_loss

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
            log_msg = f"Step {step}/{args.train_steps} | Loss (avg over last 20): {running/20:.4f}"

            # Add DiT-specific metrics if using DiT bridge
            if args.bridge == "dit":
                module = translator.module if isinstance(translator, DDP) else translator
                aux_losses = getattr(module, '_last_losses', {})
                if 'dit_flow' in aux_losses:
                    log_msg += f" | DiT_flow: {aux_losses['dit_flow']:.4f}"

            log(log_msg)
            running = 0.0

        # Periodic eval - ALL ranks participate
        if args.eval_every > 0 and step % args.eval_every == 0 and step != last_eval:
            last_eval = step
            with torch.no_grad():
                acc_source, acc_base, acc_bridged = evaluate_numeric_accuracy(
                    test_ds, src_model, src_tok, tgt_model, tgt_tok, translator.module if isinstance(translator, DDP) else translator,
                    device, dtype, train_ds, num_samples=args.eval_samples, max_new_tokens=args.max_new_tokens,
                    show_samples=(args.show_eval_samples > 0), target_rms=target_rms,
                    eval_batch_size=args.eval_batch_size, args=args, eval_step=step
                )

            # Only rank 0 logs and checks early stopping
            stop_training = torch.tensor(0, device=device)
            if is_main():
                log(f"[Eval] Step {step} | Source-alone acc: {acc_source:.3f} | Target-alone acc: {acc_base:.3f} | Bridged acc: {acc_bridged:.3f}")

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
                        stop_training.fill_(1)

            if dist.is_initialized():
                dist.broadcast(stop_training, src=0)
            if stop_training.item():
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
        acc_source, acc_base, acc_bridged = evaluate_numeric_accuracy(
            test_ds, src_model, src_tok, tgt_model, tgt_tok, translator.module if isinstance(translator, DDP) else translator,
            device, dtype, train_ds, num_samples=args.eval_samples, max_new_tokens=args.max_new_tokens,
            show_samples=(args.show_eval_samples > 0), target_rms=target_rms,
            eval_batch_size=args.eval_batch_size, args=args, eval_step="final"
        )

    # Synchronize all ranks before cleanup
    if dist.is_initialized():
        dist.barrier()

    if is_main():
        log(f"[Final Eval] Source-alone acc: {acc_source:.3f} | Target-alone acc: {acc_base:.3f} | Bridged acc: {acc_bridged:.3f}")

    cleanup_ddp()

if __name__ == "__main__":
    main()
