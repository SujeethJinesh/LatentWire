from __future__ import annotations

import pytest
import torch

from experimental.sinkaware.phase2.reference.sink_decomposition import (
    sink_decomposed_scalar_attention_reference,
)
from experimental.sinkaware.phase4.kernel.sink_decomposition_triton import (
    sink_decomposed_scalar_attention_triton_interpret,
)


@pytest.mark.parametrize("n_tail", [7, 65])
def test_sink_decomposition_triton_matches_reference(
    monkeypatch: pytest.MonkeyPatch, n_tail: int
) -> None:
    pytest.importorskip("triton")
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    generator = torch.Generator().manual_seed(103 + n_tail)
    sink_logits = torch.randn(3, generator=generator)
    tail_logits = torch.randn(n_tail, generator=generator)
    sink_values = torch.randn(3, generator=generator)
    tail_values = torch.randn(n_tail, generator=generator)

    expected = sink_decomposed_scalar_attention_reference(
        sink_logits, tail_logits, sink_values, tail_values
    )
    actual = sink_decomposed_scalar_attention_triton_interpret(
        sink_logits,
        tail_logits,
        sink_values,
        tail_values,
        block_sink=8,
        block_tail=128,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
