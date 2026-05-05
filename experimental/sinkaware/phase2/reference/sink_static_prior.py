"""Reference helpers for the SinkAware Phase 2 exactness gate."""

from __future__ import annotations

import torch


def full_scalar_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Compute exact scalar-value attention for each query."""

    logits = queries.to(torch.float32) @ keys.to(torch.float32).T
    weights = torch.softmax(logits, dim=-1)
    return weights @ values.to(torch.float32)


def static_sink_prior_attention(
    queries: torch.Tensor,
    sink_logit_prior: torch.Tensor,
    tail_keys: torch.Tensor,
    sink_values: torch.Tensor,
    tail_values: torch.Tensor,
) -> torch.Tensor:
    """Approximate attention that reuses a query-independent sink-logit prior."""

    tail_logits = queries.to(torch.float32) @ tail_keys.to(torch.float32).T
    sink_logits = sink_logit_prior.to(torch.float32).expand(queries.shape[0], -1)
    logits = torch.cat([sink_logits, tail_logits], dim=-1)
    values = torch.cat([sink_values.to(torch.float32), tail_values.to(torch.float32)])
    return torch.softmax(logits, dim=-1) @ values


def sink_logits_are_query_dependent(queries: torch.Tensor, sink_keys: torch.Tensor) -> bool:
    """Return true when fixed sink keys still require per-query dot products."""

    logits = queries.to(torch.float32) @ sink_keys.to(torch.float32).T
    return bool(torch.max(torch.abs(logits - logits[:1])).item() > 1e-6)
