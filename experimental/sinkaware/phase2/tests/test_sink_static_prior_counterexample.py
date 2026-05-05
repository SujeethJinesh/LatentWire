from __future__ import annotations

import torch

from experimental.sinkaware.phase2.reference.sink_static_prior import (
    full_scalar_attention,
    sink_logits_are_query_dependent,
    static_sink_prior_attention,
)


def test_static_sink_prior_is_not_exact_without_query_sink_logits() -> None:
    queries = torch.tensor([[2.0, 0.0], [-2.0, 0.0], [0.0, 2.0]])
    sink_keys = torch.tensor([[1.0, 0.0]])
    tail_keys = torch.tensor([[0.0, 1.0], [0.0, -1.0]])
    sink_values = torch.tensor([10.0])
    tail_values = torch.tensor([1.0, -1.0])
    keys = torch.cat([sink_keys, tail_keys], dim=0)
    values = torch.cat([sink_values, tail_values])

    exact = full_scalar_attention(queries, keys, values)
    static_prior = torch.tensor([0.0])
    approx = static_sink_prior_attention(
        queries, static_prior, tail_keys, sink_values, tail_values
    )

    assert sink_logits_are_query_dependent(queries, sink_keys)
    assert torch.max(torch.abs(exact - approx)).item() > 1.0
