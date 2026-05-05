from __future__ import annotations

import torch

from experimental.hybridkernel.phase3.reference.boundary import (
    hybrid_boundary_blend_reference,
)


def test_hybrid_boundary_blend_reference_matches_manual_formula() -> None:
    attention_state = torch.tensor([1.0, 2.0, 3.0])
    ssm_state = torch.tensor([10.0, 20.0, 30.0])
    gate = torch.tensor([0.0, 0.5, 1.0])
    bias = torch.tensor([0.25, -0.25, 0.0])

    out = hybrid_boundary_blend_reference(attention_state, ssm_state, gate, bias)

    expected = torch.tensor([10.25, 10.75, 3.0])
    torch.testing.assert_close(out, expected)
