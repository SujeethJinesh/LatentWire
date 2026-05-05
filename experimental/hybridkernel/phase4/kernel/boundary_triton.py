"""Triton interpreter kernel for the HybridKernel boundary primitive.

Run only with TRITON_INTERPRET=1 during Macbook phases. This checks the kernel
math against the CPU reference but does not measure GPU performance.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - exercised by local dependency gate
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _hybrid_boundary_blend_kernel(
        attention_ptr,
        ssm_ptr,
        gate_ptr,
        bias_ptr,
        out_ptr,
        n_elements: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        attention = tl.load(attention_ptr + offsets, mask=mask, other=0.0)
        ssm = tl.load(ssm_ptr + offsets, mask=mask, other=0.0)
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
        bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        out = gate * attention + (1.0 - gate) * ssm + bias
        tl.store(out_ptr + offsets, out, mask=mask)


def _require_interpreter_mode() -> None:
    if os.environ.get("TRITON_INTERPRET") != "1":
        raise RuntimeError("Set TRITON_INTERPRET=1 for Macbook kernel correctness gates")
    if triton is None:
        raise ModuleNotFoundError("triton is not importable in this environment")


def hybrid_boundary_blend_triton_interpret(
    attention_state: torch.Tensor,
    ssm_state: torch.Tensor,
    gate: torch.Tensor,
    bias: torch.Tensor,
    *,
    block_size: int = 128,
) -> torch.Tensor:
    """Run the boundary blend kernel in Triton interpreter mode."""

    _require_interpreter_mode()
    if attention_state.shape != ssm_state.shape:
        raise ValueError("attention_state and ssm_state must have the same shape")
    if gate.shape != attention_state.shape:
        raise ValueError("gate must have the same shape as attention_state")
    if bias.shape != attention_state.shape:
        raise ValueError("bias must have the same shape as attention_state")

    attention_flat = attention_state.contiguous().to(torch.float32).flatten()
    ssm_flat = ssm_state.contiguous().to(torch.float32).flatten()
    gate_flat = gate.contiguous().to(torch.float32).flatten()
    bias_flat = bias.contiguous().to(torch.float32).flatten()
    out = torch.empty_like(attention_flat)
    n_elements = attention_flat.numel()
    grid = (triton.cdiv(n_elements, block_size),)
    _hybrid_boundary_blend_kernel[grid](
        attention_flat,
        ssm_flat,
        gate_flat,
        bias_flat,
        out,
        n_elements,
        BLOCK=block_size,
    )
    return out.reshape_as(attention_state)
