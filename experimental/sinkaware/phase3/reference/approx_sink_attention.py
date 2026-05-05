"""CPU reference for approximate fixed-sink attention.

The live SinkAware branch does not skip fixed-sink logits exactly. It replaces
only the sink-token logits with a predictor and keeps the non-sink tail logits
exact. This reference defines the operator that a future Triton/CUDA kernel must
match before any performance claim is allowed.
"""

from __future__ import annotations

import math

import torch


def exact_attention_reference(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact per-head attention output and probabilities.

    Shapes:
    - query: [heads, head_dim]
    - keys: [heads, seq, head_dim]
    - values: [heads, seq, value_dim]
    """

    if query.ndim != 2 or keys.ndim != 3 or values.ndim != 3:
        raise ValueError("expected query [H,D], keys [H,L,D], values [H,L,V]")
    if query.shape[0] != keys.shape[0] or keys.shape[:2] != values.shape[:2]:
        raise ValueError("head and sequence dimensions must match")
    logits = torch.einsum("hd,hld->hl", query.float(), keys.float()) / math.sqrt(query.shape[-1])
    probs = torch.softmax(logits, dim=-1)
    out = torch.einsum("hl,hld->hd", probs, values.float())
    return out, probs


def approx_sink_attention_reference(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    predicted_sink_logits: torch.Tensor,
    *,
    sink_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return attention after replacing only the fixed-sink logits.

    The tail logits for positions `sink_tokens:` are still exact QK products.
    """

    if sink_tokens <= 0:
        raise ValueError("sink_tokens must be positive")
    if predicted_sink_logits.shape != (query.shape[0], sink_tokens):
        raise ValueError("predicted_sink_logits must have shape [heads, sink_tokens]")
    if keys.shape[1] < sink_tokens:
        raise ValueError("sequence length must include sink tokens")
    logits = torch.einsum("hd,hld->hl", query.float(), keys.float()) / math.sqrt(query.shape[-1])
    logits[:, :sink_tokens] = predicted_sink_logits.float()
    probs = torch.softmax(logits, dim=-1)
    out = torch.einsum("hl,hld->hd", probs, values.float())
    return out, probs


def attention_drift_metrics(
    exact_out: torch.Tensor,
    exact_probs: torch.Tensor,
    approx_out: torch.Tensor,
    approx_probs: torch.Tensor,
    *,
    sink_tokens: int,
) -> dict[str, float]:
    return {
        "sink_mass_mae": float(
            torch.mean(
                torch.abs(
                    exact_probs[:, :sink_tokens].sum(dim=-1)
                    - approx_probs[:, :sink_tokens].sum(dim=-1)
                )
            )
        ),
        "attention_l1": float(torch.mean(torch.sum(torch.abs(exact_probs - approx_probs), dim=-1))),
        "output_rel_l2": float(
            torch.linalg.norm(exact_out - approx_out) / torch.linalg.norm(exact_out).clamp_min(1e-8)
        ),
    }
