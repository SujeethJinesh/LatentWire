"""Triton interpreter kernel for anchor/phase protected int8 quantization."""

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
    def _anchor_phase_quant_kernel(
        values_ptr,
        importance_ptr,
        anchor_mask_ptr,
        phase_mask_ptr,
        quantized_ptr,
        keep_ptr,
        threshold: tl.constexpr,
        scale: tl.constexpr,
        n_elements: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        values = tl.load(values_ptr + offsets, mask=mask, other=0.0)
        importance = tl.load(importance_ptr + offsets, mask=mask, other=0.0)
        anchor = tl.load(anchor_mask_ptr + offsets, mask=mask, other=0).to(tl.int1)
        phase = tl.load(phase_mask_ptr + offsets, mask=mask, other=0).to(tl.int1)
        keep = anchor | phase | (importance >= threshold)

        scaled = values / scale
        abs_scaled = tl.abs(scaled)
        rounded_abs = tl.floor(abs_scaled + 0.5)
        rounded = tl.where(scaled >= 0.0, rounded_abs, -rounded_abs)
        clipped = tl.minimum(tl.maximum(rounded, -127.0), 127.0).to(tl.int8)
        quantized = tl.where(keep, clipped, 0)
        tl.store(quantized_ptr + offsets, quantized, mask=mask)
        tl.store(keep_ptr + offsets, keep.to(tl.int8), mask=mask)


def _require_interpreter_mode() -> None:
    if os.environ.get("TRITON_INTERPRET") != "1":
        raise RuntimeError("Set TRITON_INTERPRET=1 for Macbook kernel correctness gates")
    if triton is None:
        raise ModuleNotFoundError("triton is not importable in this environment")


def anchor_phase_quantize_triton_interpret(
    values: torch.Tensor,
    importance: torch.Tensor,
    anchor_mask: torch.Tensor,
    phase_mask: torch.Tensor,
    *,
    threshold: float,
    scale: float,
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run anchor/phase protected int8 quantization in interpreter mode."""

    _require_interpreter_mode()
    if values.shape != importance.shape:
        raise ValueError("values and importance must have the same shape")
    if anchor_mask.shape != values.shape:
        raise ValueError("anchor_mask must match values")
    if phase_mask.shape != values.shape:
        raise ValueError("phase_mask must match values")
    if scale <= 0.0:
        raise ValueError("scale must be positive")

    values_flat = values.contiguous().to(torch.float32).flatten()
    importance_flat = importance.contiguous().to(torch.float32).flatten()
    anchor_flat = anchor_mask.contiguous().to(torch.uint8).flatten()
    phase_flat = phase_mask.contiguous().to(torch.uint8).flatten()
    quantized = torch.empty(values_flat.shape, dtype=torch.int8)
    keep = torch.empty(values_flat.shape, dtype=torch.uint8)
    n_elements = values_flat.numel()
    grid = (triton.cdiv(n_elements, block_size),)
    _anchor_phase_quant_kernel[grid](
        values_flat,
        importance_flat,
        anchor_flat,
        phase_flat,
        quantized,
        keep,
        threshold,
        scale,
        n_elements,
        BLOCK=block_size,
    )
    return quantized.reshape_as(values), keep.bool().reshape_as(values)
