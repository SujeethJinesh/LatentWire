from __future__ import annotations

import torch

from experimental.sinkaware.phase2.reference.sink_decomposition import (
    sink_decomposed_scalar_attention_reference,
)


def test_sink_decomposition_reference_matches_direct_softmax() -> None:
    sink_logits = torch.tensor([2.0, -0.5])
    tail_logits = torch.tensor([0.0, 1.0, -1.0])
    sink_values = torch.tensor([10.0, -2.0])
    tail_values = torch.tensor([1.0, 4.0, 0.5])

    actual = sink_decomposed_scalar_attention_reference(
        sink_logits, tail_logits, sink_values, tail_values
    )

    logits = torch.cat([sink_logits, tail_logits])
    values = torch.cat([sink_values, tail_values])
    expected = torch.sum(torch.softmax(logits, dim=0) * values)
    torch.testing.assert_close(actual, expected)
