"""CPU reference for fixed sink-token attention decomposition.

This Phase 2 primitive tests whether fixed sink-token logits/values can be
combined exactly with non-sink logits/values for a single query row. It is a
scalar-value reference used for kernel correctness scaffolding, not a full
FlashAttention replacement.
"""

from __future__ import annotations

import torch


def sink_decomposed_scalar_attention_reference(
    sink_logits: torch.Tensor,
    tail_logits: torch.Tensor,
    sink_values: torch.Tensor,
    tail_values: torch.Tensor,
) -> torch.Tensor:
    """Return exact softmax-weighted scalar output for sink + tail tokens."""

    if sink_logits.ndim != 1 or tail_logits.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if sink_values.shape != sink_logits.shape:
        raise ValueError("sink_values must match sink_logits")
    if tail_values.shape != tail_logits.shape:
        raise ValueError("tail_values must match tail_logits")

    logits = torch.cat([sink_logits, tail_logits]).to(torch.float32)
    values = torch.cat([sink_values, tail_values]).to(torch.float32)
    weights = torch.softmax(logits, dim=0)
    return torch.sum(weights * values)
