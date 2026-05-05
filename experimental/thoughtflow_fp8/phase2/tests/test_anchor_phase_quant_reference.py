from __future__ import annotations

import torch

from experimental.thoughtflow_fp8.phase2.reference.anchor_phase_quant import (
    anchor_phase_quantize_reference,
)


def test_anchor_phase_quantize_reference_protects_anchor_and_phase_tokens() -> None:
    values = torch.tensor([0.1, 0.2, -0.3, 1.3])
    importance = torch.tensor([0.0, 0.1, 0.9, 0.2])
    anchor_mask = torch.tensor([1, 0, 0, 0], dtype=torch.uint8)
    phase_mask = torch.tensor([0, 1, 0, 0], dtype=torch.uint8)

    quantized, keep = anchor_phase_quantize_reference(
        values,
        importance,
        anchor_mask,
        phase_mask,
        threshold=0.5,
        scale=0.1,
    )

    torch.testing.assert_close(keep, torch.tensor([True, True, True, False]))
    torch.testing.assert_close(quantized, torch.tensor([1, 2, -3, 0], dtype=torch.int8))
