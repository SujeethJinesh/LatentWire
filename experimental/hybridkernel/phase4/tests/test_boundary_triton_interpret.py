from __future__ import annotations

import pytest
import torch

from experimental.hybridkernel.phase3.reference.boundary import (
    hybrid_boundary_blend_reference,
)
from experimental.hybridkernel.phase4.kernel.boundary_triton import (
    hybrid_boundary_blend_triton_interpret,
)


@pytest.mark.parametrize("shape", [(1,), (2, 64), (3, 257), (2, 5, 129)])
def test_hybrid_boundary_blend_triton_matches_reference(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...]
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


def test_hybrid_boundary_blend_triton_accepts_noncontiguous_and_fp16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("triton")
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    generator = torch.Generator().manual_seed(2907)
    base_shape = (4, 19, 2)
    attention_state = torch.randn(base_shape, generator=generator, dtype=torch.float16)[:, :, 0]
    ssm_state = torch.randn(base_shape, generator=generator, dtype=torch.float16)[:, :, 0]
    gate = torch.rand(base_shape, generator=generator, dtype=torch.float16)[:, :, 0]
    bias = (torch.randn(base_shape, generator=generator, dtype=torch.float16) * 0.01)[:, :, 0]

    assert not attention_state.is_contiguous()
    expected = hybrid_boundary_blend_reference(attention_state, ssm_state, gate, bias)
    actual = hybrid_boundary_blend_triton_interpret(
        attention_state, ssm_state, gate, bias, block_size=32
    )

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def test_hybrid_boundary_blend_triton_rejects_shape_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("triton")
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    good = torch.zeros((2, 3))
    bad = torch.zeros((2, 4))

    with pytest.raises(ValueError, match="same shape"):
        hybrid_boundary_blend_triton_interpret(good, bad, good, good, block_size=16)
