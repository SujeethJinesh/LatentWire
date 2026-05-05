from __future__ import annotations

import pytest
import torch

from experimental.thoughtflow_fp8.phase2.reference.anchor_phase_quant import (
    anchor_phase_quantize_reference,
)
from experimental.thoughtflow_fp8.phase4.kernel.anchor_phase_quant_triton import (
    anchor_phase_quantize_triton_interpret,
)


@pytest.mark.parametrize("n_elements", [33, 257])
def test_anchor_phase_quant_triton_matches_reference(
    monkeypatch: pytest.MonkeyPatch, n_elements: int
) -> None:
    pytest.importorskip("triton")
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    generator = torch.Generator().manual_seed(211 + n_elements)
    values = torch.randn(n_elements, generator=generator)
    importance = torch.rand(n_elements, generator=generator)
    anchor_mask = (torch.arange(n_elements) % 17 == 0).to(torch.uint8)
    phase_mask = (torch.arange(n_elements) % 29 == 0).to(torch.uint8)

    expected_q, expected_keep = anchor_phase_quantize_reference(
        values,
        importance,
        anchor_mask,
        phase_mask,
        threshold=0.72,
        scale=0.125,
    )
    actual_q, actual_keep = anchor_phase_quantize_triton_interpret(
        values,
        importance,
        anchor_mask,
        phase_mask,
        threshold=0.72,
        scale=0.125,
        block_size=128,
    )

    torch.testing.assert_close(actual_keep, expected_keep)
    torch.testing.assert_close(actual_q, expected_q)
