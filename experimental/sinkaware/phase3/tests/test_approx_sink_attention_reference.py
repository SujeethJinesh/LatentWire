from __future__ import annotations

import math

import pytest
import torch

from experimental.sinkaware.phase3.reference.approx_sink_attention import (
    approx_sink_attention_reference,
    attention_drift_metrics,
    exact_attention_reference,
)


def test_exact_sink_logits_reproduce_exact_attention() -> None:
    generator = torch.Generator().manual_seed(20260505)
    query = torch.randn(3, 8, generator=generator)
    keys = torch.randn(3, 13, 8, generator=generator)
    values = torch.randn(3, 13, 5, generator=generator)
    sink_tokens = 4

    exact_out, exact_probs = exact_attention_reference(query, keys, values)
    exact_logits = torch.einsum("hd,hld->hl", query, keys) / math.sqrt(query.shape[-1])
    approx_out, approx_probs = approx_sink_attention_reference(
        query,
        keys,
        values,
        exact_logits[:, :sink_tokens],
        sink_tokens=sink_tokens,
    )

    torch.testing.assert_close(approx_out, exact_out, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(approx_probs, exact_probs, rtol=1e-6, atol=1e-6)


def test_approx_sink_attention_changes_only_sink_logits() -> None:
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(2, 4, generator=generator)
    keys = torch.randn(2, 9, 4, generator=generator)
    values = torch.randn(2, 9, 4, generator=generator)
    sink_tokens = 2

    exact_out, exact_probs = exact_attention_reference(query, keys, values)
    exact_logits = torch.einsum("hd,hld->hl", query, keys) / math.sqrt(query.shape[-1])
    predicted = exact_logits[:, :sink_tokens] + 0.1
    approx_out, approx_probs = approx_sink_attention_reference(
        query,
        keys,
        values,
        predicted,
        sink_tokens=sink_tokens,
    )
    metrics = attention_drift_metrics(
        exact_out,
        exact_probs,
        approx_out,
        approx_probs,
        sink_tokens=sink_tokens,
    )

    assert metrics["sink_mass_mae"] > 0.0
    assert metrics["attention_l1"] > 0.0
    assert metrics["output_rel_l2"] > 0.0


def test_shape_checks() -> None:
    query = torch.zeros(2, 4)
    keys = torch.zeros(2, 5, 4)
    values = torch.zeros(2, 5, 4)

    with pytest.raises(ValueError):
        approx_sink_attention_reference(query, keys, values, torch.zeros(2, 3), sink_tokens=2)
