# -*- coding: utf-8 -*-
"""Prefix utilities for LatentWire.

This module provides utilities for:
- Prefix calibration
- BOS token handling
- Anchor text processing
- Embedding normalization
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


def calibrate_to_embed_rms(
    latents: torch.Tensor,
    text_embeds: torch.Tensor,
    epsilon: float = 1e-6
) -> torch.Tensor:
    """Calibrate latents to match text embedding RMS statistics.

    Args:
        latents: Latent representations [batch_size, seq_len, hidden_dim]
        text_embeds: Text embeddings for reference [batch_size, seq_len, hidden_dim]
        epsilon: Small value for numerical stability

    Returns:
        Calibrated latents with matched RMS
    """
    # Compute RMS for text embeddings
    text_rms = torch.sqrt(torch.mean(text_embeds ** 2, dim=-1, keepdim=True) + epsilon)

    # Compute RMS for latents
    latent_rms = torch.sqrt(torch.mean(latents ** 2, dim=-1, keepdim=True) + epsilon)

    # Scale latents to match text RMS
    scale_factor = text_rms / (latent_rms + epsilon)
    calibrated_latents = latents * scale_factor

    return calibrated_latents


def apply_bos_policy(
    input_ids: torch.Tensor,
    tokenizer: Any,
    policy: str = "auto"
) -> torch.Tensor:
    """Apply BOS token policy to input IDs.

    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        tokenizer: Tokenizer with BOS token
        policy: BOS policy - "always", "never", or "auto"

    Returns:
        Input IDs with BOS policy applied
    """
    if policy == "never":
        # Remove BOS if present
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            if (input_ids[:, 0] == tokenizer.bos_token_id).all():
                return input_ids[:, 1:]
        return input_ids

    elif policy == "always":
        # Add BOS if not present
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            if not (input_ids[:, 0] == tokenizer.bos_token_id).all():
                batch_size = input_ids.shape[0]
                bos_tokens = torch.full(
                    (batch_size, 1),
                    tokenizer.bos_token_id,
                    dtype=input_ids.dtype,
                    device=input_ids.device
                )
                return torch.cat([bos_tokens, input_ids], dim=1)
        return input_ids

    else:  # auto
        # Let tokenizer decide
        return input_ids


def get_anchor_token_ids(
    anchor_text: str,
    tokenizer: Any,
    add_space: bool = True
) -> List[int]:
    """Get token IDs for anchor text.

    Args:
        anchor_text: Anchor text string
        tokenizer: Tokenizer instance
        add_space: Whether to add leading space

    Returns:
        List of token IDs for anchor text
    """
    if add_space and not anchor_text.startswith(" "):
        anchor_text = " " + anchor_text

    # Encode without special tokens
    anchor_ids = tokenizer.encode(anchor_text, add_special_tokens=False)

    return anchor_ids


def create_position_ids(
    seq_length: int,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    start_pos: int = 0
) -> torch.Tensor:
    """Create position IDs for sequence.

    Args:
        seq_length: Sequence length
        batch_size: Batch size
        device: Device to create tensor on
        start_pos: Starting position

    Returns:
        Position IDs tensor [batch_size, seq_length]
    """
    position_ids = torch.arange(
        start_pos,
        start_pos + seq_length,
        dtype=torch.long,
        device=device
    )
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

    return position_ids


def normalize_embeddings(
    embeddings: torch.Tensor,
    norm_type: str = "layer",
    epsilon: float = 1e-6
) -> torch.Tensor:
    """Normalize embeddings.

    Args:
        embeddings: Embedding tensor [batch_size, seq_len, hidden_dim]
        norm_type: Type of normalization - "layer", "rms", "none"
        epsilon: Small value for stability

    Returns:
        Normalized embeddings
    """
    if norm_type == "layer":
        # Layer normalization
        mean = embeddings.mean(dim=-1, keepdim=True)
        var = embeddings.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (embeddings - mean) / torch.sqrt(var + epsilon)

    elif norm_type == "rms":
        # RMS normalization
        rms = torch.sqrt(torch.mean(embeddings ** 2, dim=-1, keepdim=True) + epsilon)
        normalized = embeddings / rms

    else:  # none
        normalized = embeddings

    return normalized


def compute_embedding_stats(
    embeddings: torch.Tensor
) -> Dict[str, float]:
    """Compute statistics for embeddings.

    Args:
        embeddings: Embedding tensor

    Returns:
        Dictionary of statistics
    """
    stats = {}

    # Basic statistics
    stats["mean"] = embeddings.mean().item()
    stats["std"] = embeddings.std().item()
    stats["min"] = embeddings.min().item()
    stats["max"] = embeddings.max().item()

    # RMS
    stats["rms"] = torch.sqrt(torch.mean(embeddings ** 2)).item()

    # L2 norm
    stats["l2_norm"] = torch.norm(embeddings, p=2, dim=-1).mean().item()

    # Sparsity (fraction of near-zero values)
    near_zero = (torch.abs(embeddings) < 1e-6).float()
    stats["sparsity"] = near_zero.mean().item()

    return stats


def align_prefix_lengths(
    prefix1: torch.Tensor,
    prefix2: torch.Tensor,
    pad_value: float = 0.0,
    side: str = "right"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Align two prefix tensors to same length.

    Args:
        prefix1: First prefix tensor
        prefix2: Second prefix tensor
        pad_value: Value to use for padding
        side: Side to pad - "left" or "right"

    Returns:
        Tuple of aligned prefixes
    """
    len1 = prefix1.shape[1]
    len2 = prefix2.shape[1]

    if len1 == len2:
        return prefix1, prefix2

    max_len = max(len1, len2)

    # Pad first prefix if needed
    if len1 < max_len:
        pad_size = max_len - len1
        if side == "left":
            padding = (0, 0, pad_size, 0)  # (left, right, top, bottom)
        else:
            padding = (0, 0, 0, pad_size)
        prefix1 = F.pad(prefix1, padding, value=pad_value)

    # Pad second prefix if needed
    if len2 < max_len:
        pad_size = max_len - len2
        if side == "left":
            padding = (0, 0, pad_size, 0)
        else:
            padding = (0, 0, 0, pad_size)
        prefix2 = F.pad(prefix2, padding, value=pad_value)

    return prefix1, prefix2


def extract_first_token_logits(
    logits: torch.Tensor,
    prefix_length: int
) -> torch.Tensor:
    """Extract logits for first generated token after prefix.

    Args:
        logits: Full sequence logits [batch_size, seq_len, vocab_size]
        prefix_length: Length of prefix

    Returns:
        First token logits [batch_size, vocab_size]
    """
    if prefix_length >= logits.shape[1]:
        # Prefix is entire sequence
        return logits[:, -1, :]
    else:
        # Get logits at prefix boundary
        return logits[:, prefix_length - 1, :]


def create_causal_mask(
    seq_length: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Create causal attention mask.

    Args:
        seq_length: Sequence length
        device: Device to create tensor on
        dtype: Data type for mask

    Returns:
        Causal mask tensor [seq_length, seq_length]
    """
    mask = torch.triu(
        torch.ones(seq_length, seq_length, dtype=dtype, device=device),
        diagonal=1
    )
    mask = mask.masked_fill(mask == 1, float('-inf'))

    return mask