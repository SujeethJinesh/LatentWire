"""Triton interpreter kernel for a fixed sink-token decomposition primitive."""

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
    def _sink_decomposed_scalar_kernel(
        sink_logits_ptr,
        tail_logits_ptr,
        sink_values_ptr,
        tail_values_ptr,
        out_ptr,
        n_sink: tl.constexpr,
        n_tail: tl.constexpr,
        BLOCK_SINK: tl.constexpr,
        BLOCK_TAIL: tl.constexpr,
    ):
        sink_offsets = tl.arange(0, BLOCK_SINK)
        tail_offsets = tl.arange(0, BLOCK_TAIL)
        sink_mask = sink_offsets < n_sink
        tail_mask = tail_offsets < n_tail

        sink_logits = tl.load(sink_logits_ptr + sink_offsets, mask=sink_mask, other=-float("inf"))
        tail_logits = tl.load(tail_logits_ptr + tail_offsets, mask=tail_mask, other=-float("inf"))
        max_logit = tl.maximum(tl.max(sink_logits, axis=0), tl.max(tail_logits, axis=0))

        sink_weights = tl.exp(sink_logits - max_logit)
        tail_weights = tl.exp(tail_logits - max_logit)
        sink_values = tl.load(sink_values_ptr + sink_offsets, mask=sink_mask, other=0.0)
        tail_values = tl.load(tail_values_ptr + tail_offsets, mask=tail_mask, other=0.0)

        numerator = tl.sum(sink_weights * sink_values, axis=0) + tl.sum(
            tail_weights * tail_values, axis=0
        )
        denominator = tl.sum(sink_weights, axis=0) + tl.sum(tail_weights, axis=0)
        tl.store(out_ptr, numerator / denominator)


def _require_interpreter_mode() -> None:
    if os.environ.get("TRITON_INTERPRET") != "1":
        raise RuntimeError("Set TRITON_INTERPRET=1 for Macbook kernel correctness gates")
    if triton is None:
        raise ModuleNotFoundError("triton is not importable in this environment")


def sink_decomposed_scalar_attention_triton_interpret(
    sink_logits: torch.Tensor,
    tail_logits: torch.Tensor,
    sink_values: torch.Tensor,
    tail_values: torch.Tensor,
    *,
    block_sink: int = 16,
    block_tail: int = 128,
) -> torch.Tensor:
    """Run exact scalar sink+tail softmax output in Triton interpreter mode."""

    _require_interpreter_mode()
    if sink_logits.ndim != 1 or tail_logits.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if sink_values.shape != sink_logits.shape:
        raise ValueError("sink_values must match sink_logits")
    if tail_values.shape != tail_logits.shape:
        raise ValueError("tail_values must match tail_logits")
    if sink_logits.numel() > block_sink or tail_logits.numel() > block_tail:
        raise ValueError("increase block sizes for this synthetic gate")

    out = torch.empty((), dtype=torch.float32)
    _sink_decomposed_scalar_kernel[(1,)](
        sink_logits.contiguous().to(torch.float32),
        tail_logits.contiguous().to(torch.float32),
        sink_values.contiguous().to(torch.float32),
        tail_values.contiguous().to(torch.float32),
        out,
        sink_logits.numel(),
        tail_logits.numel(),
        BLOCK_SINK=block_sink,
        BLOCK_TAIL=block_tail,
    )
    return out
