from __future__ import annotations

import pytest
import torch

from experimental.hybridkernel.phase3.reference.boundary import (
    hybrid_boundary_blend_reference,
)
from experimental.hybridkernel.phase4.kernel.boundary_triton import (
    hybrid_boundary_blend_triton_interpret,
)


@pytest.mark.parametrize("shape", [(2, 64), (3, 257)])
def test_hybrid_boundary_blend_triton_matches_reference(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, int]
) -> None:
    pytest.importorskip("triton")
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    generator = torch.Generator().manual_seed(17 + shape[-1])
    attention_state = torch.randn(shape, generator=generator)
    ssm_state = torch.randn(shape, generator=generator)
    gate = torch.rand(shape, generator=generator)
    bias = torch.randn(shape, generator=generator) * 0.01

    expected = hybrid_boundary_blend_reference(attention_state, ssm_state, gate, bias)
    actual = hybrid_boundary_blend_triton_interpret(
        attention_state, ssm_state, gate, bias, block_size=128
    )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
