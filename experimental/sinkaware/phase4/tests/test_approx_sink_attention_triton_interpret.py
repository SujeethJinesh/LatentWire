from __future__ import annotations

import math

import pytest
import torch

from experimental.sinkaware.phase4.kernel.approx_sink_attention_triton import (
    approx_sink_attention_scalar_triton_interpret,
    exact_scalar_attention_reference,
)


def test_approx_sink_attention_triton_matches_exact_when_prediction_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("triton")
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    generator = torch.Generator().manual_seed(20260505)
    query = torch.randn(8, generator=generator)
    keys = torch.randn(11, 8, generator=generator)
    values = torch.randn(11, generator=generator)
    sink_tokens = 3
    exact_logits = keys @ query / math.sqrt(query.numel())

    expected = exact_scalar_attention_reference(query, keys, values)
    actual = approx_sink_attention_scalar_triton_interpret(
        query,
        keys,
        values,
        exact_logits[:sink_tokens],
        sink_tokens=sink_tokens,
        block_seq=16,
        block_dim=16,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_approx_sink_attention_triton_requires_interpret(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("triton")
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    with pytest.raises(RuntimeError):
        approx_sink_attention_scalar_triton_interpret(
            torch.zeros(4),
            torch.zeros(6, 4),
            torch.zeros(6),
            torch.zeros(2),
            sink_tokens=2,
        )
