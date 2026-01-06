# latentwire/loss_bundles.py
"""
Helper functions for auxiliary loss computation.

These utilities were originally defined inside `latentwire.train`. They
are extracted here to keep the training loop lean while preserving
behaviour.
"""
import os
from typing import Optional, Tuple, Any

import torch

from latentwire.core_utils import tensor_rms_d


def loss_with_text_prompt_chunked(wrapper, scaffold_ids, target_ids):
    """Compute teacher-forced loss in small chunks to save memory."""
    chunk_env = os.getenv("TEXT_TEACHER_CHUNK", "4")
    try:
        chunk_size = max(1, int(chunk_env))
    except ValueError:
        chunk_size = 4

    batch_size = scaffold_ids.size(0)
    total_loss = torch.zeros((), device=scaffold_ids.device)
    count = 0
    for start in range(0, batch_size, chunk_size):
        end = min(batch_size, start + chunk_size)
        loss, _, _ = wrapper.loss_with_text_prompt(
            scaffold_ids[start:end], target_ids[start:end]
        )
        total_loss = total_loss + loss * (end - start)
        count += end - start
    avg_loss = total_loss / max(count, 1)
    return avg_loss, None, None


def alignment_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Mean-squared error with optional masking."""
    if pred.numel() == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    diff = (pred - target).pow(2)
    if mask is not None:
        mask_base = mask.to(pred.device, dtype=pred.dtype)
        diff = diff * mask_base.unsqueeze(-1)
        denom = (mask_base.sum().clamp_min(1.0)) * pred.size(-1)
    else:
        denom = torch.tensor(float(diff.numel()), device=pred.device, dtype=pred.dtype)
    return diff.sum() / denom


def manifold_stat_loss(
    prefix: torch.Tensor,
    embed_stats: Tuple[torch.Tensor, torch.Tensor],
    weight: float,
) -> torch.Tensor:
    """Match latent prefix statistics (mean/std) to reference embeddings."""
    if weight <= 0.0:
        return torch.zeros((), device=prefix.device)
    mu, sd = embed_stats
    mu = mu.to(prefix.device, dtype=prefix.dtype)
    sd = sd.to(prefix.device, dtype=prefix.dtype)
    cur_mu = prefix.float().mean(dim=[0, 1])
    cur_sd = prefix.float().std(dim=[0, 1]).clamp_min(1e-6)
    return (cur_mu - mu).pow(2).mean() + (cur_sd - sd).pow(2).mean()


def scale_penalty(adapter: Any, weight: float, device: torch.device) -> torch.Tensor:
    """L2 penalty on adapter scale parameters."""
    if weight <= 0.0 or getattr(adapter, "scale", None) is None:
        return torch.zeros((), device=device)
    scale_param = adapter.scale
    if not getattr(scale_param, "requires_grad", False):
        return torch.zeros((), device=device)
    return (scale_param - 1.0).pow(2).mean()


def rms_raw_penalty(
    prefix_raw: torch.Tensor,
    wrapper,
    weight: float,
) -> torch.Tensor:
    """Match latent RMS to embedding RMS when requested."""
    if weight <= 0.0:
        return torch.zeros((), device=prefix_raw.device)
    tgt = prefix_raw.new_tensor(wrapper.input_embedding_rms())
    cur = tensor_rms_d(prefix_raw)
    return (cur - tgt).pow(2)

