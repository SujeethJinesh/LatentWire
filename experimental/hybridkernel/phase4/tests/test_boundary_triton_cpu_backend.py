from __future__ import annotations

import os

import pytest
import torch

from experimental.hybridkernel.phase3.reference.boundary import (
    hybrid_boundary_blend_reference,
)
from experimental.hybridkernel.phase4.kernel.boundary_triton import (
    hybrid_boundary_blend_triton_cpu_backend,
)


def test_hybrid_boundary_blend_triton_cpu_backend_matches_reference() -> None:
    if os.environ.get("HYBRIDKERNEL_RUN_TRITON_CPU_BACKEND") != "1":
        pytest.skip("set HYBRIDKERNEL_RUN_TRITON_CPU_BACKEND=1 to run this opt-in gate")
    if os.environ.get("TRITON_CPU_BACKEND") != "1":
        pytest.skip("set TRITON_CPU_BACKEND=1 to run the CPU-backend gate")
    if os.environ.get("TRITON_INTERPRET") == "1":
        pytest.skip("run CPU-backend gate from a process without TRITON_INTERPRET=1")
    pytest.importorskip("triton")

    generator = torch.Generator().manual_seed(2906)
    shape = (2, 17)
    attention_state = torch.randn(shape, generator=generator)
    ssm_state = torch.randn(shape, generator=generator)
    gate = torch.rand(shape, generator=generator)
    bias = torch.randn(shape, generator=generator) * 0.01

    expected = hybrid_boundary_blend_reference(attention_state, ssm_state, gate, bias)
    actual = hybrid_boundary_blend_triton_cpu_backend(
        attention_state, ssm_state, gate, bias, block_size=64
    )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
