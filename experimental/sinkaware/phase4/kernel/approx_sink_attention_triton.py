"""Triton interpreter scaffold for approximate fixed-sink attention.

This is a scalar-output correctness kernel for the live SinkAware operator. It
computes exact tail QK logits, uses predicted fixed-sink logits for the sink
positions, and returns one attention output dimension for one head.
"""

from __future__ import annotations

import math
import os

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - local dependency gate
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _approx_sink_attention_scalar_kernel(
        query_ptr,
        keys_ptr,
        values_ptr,
        predicted_sink_logits_ptr,
        out_ptr,
        seq_len: tl.constexpr,
        head_dim: tl.constexpr,
        sink_tokens: tl.constexpr,
        BLOCK_SEQ: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
    ):
        seq_offsets = tl.arange(0, BLOCK_SEQ)
        dim_offsets = tl.arange(0, BLOCK_DIM)
        seq_mask = seq_offsets < seq_len
        dim_mask = dim_offsets < head_dim

        query = tl.load(query_ptr + dim_offsets, mask=dim_mask, other=0.0)
        keys = tl.load(
            keys_ptr + seq_offsets[:, None] * head_dim + dim_offsets[None, :],
            mask=seq_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        logits = tl.sum(keys * query[None, :], axis=1) / tl.sqrt(head_dim + 0.0)

        sink_offsets = tl.arange(0, BLOCK_SEQ)
        predicted = tl.load(
            predicted_sink_logits_ptr + sink_offsets,
            mask=sink_offsets < sink_tokens,
            other=0.0,
        )
        logits = tl.where(seq_offsets < sink_tokens, predicted, logits)
        logits = tl.where(seq_mask, logits, -float("inf"))
        max_logit = tl.max(logits, axis=0)
        weights = tl.exp(logits - max_logit)
        vals = tl.load(values_ptr + seq_offsets, mask=seq_mask, other=0.0)
        numerator = tl.sum(weights * vals, axis=0)
        denominator = tl.sum(weights, axis=0)
        tl.store(out_ptr, numerator / denominator)


def _require_interpreter_mode() -> None:
    if os.environ.get("TRITON_INTERPRET") != "1":
        raise RuntimeError("Set TRITON_INTERPRET=1 for Macbook kernel correctness gates")
    if triton is None:
        raise ModuleNotFoundError("triton is not importable in this environment")


def triton_interpreter_readiness() -> dict[str, object]:
    """Report whether the local environment can run the interpreter gate."""

    triton_importable = triton is not None
    interpret_enabled = os.environ.get("TRITON_INTERPRET") == "1"
    ready = triton_importable and interpret_enabled
    if not triton_importable:
        reason = "triton is not importable"
    elif not interpret_enabled:
        reason = "TRITON_INTERPRET is not set to 1"
    else:
        reason = "ready for interpreter correctness tests"
    return {
        "ready": ready,
        "reason": reason,
        "triton_importable": triton_importable,
        "triton_version": getattr(triton, "__version__", None) if triton_importable else None,
        "triton_interpret_enabled": interpret_enabled,
        "torch_cuda_available": torch.cuda.is_available(),
    }


def approx_sink_attention_scalar_triton_interpret(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    predicted_sink_logits: torch.Tensor,
    *,
    sink_tokens: int,
    block_seq: int = 128,
    block_dim: int = 128,
) -> torch.Tensor:
    """Run scalar approximate sink attention in Triton interpreter mode.

    Shapes:
    - query: [head_dim]
    - keys: [seq_len, head_dim]
    - values: [seq_len]
    - predicted_sink_logits: [sink_tokens]
    """

    _require_interpreter_mode()
    if query.ndim != 1 or keys.ndim != 2 or values.ndim != 1:
        raise ValueError("expected query [D], keys [L,D], values [L]")
    if keys.shape[0] != values.shape[0] or keys.shape[1] != query.shape[0]:
        raise ValueError("key/value/query shapes must match")
    if predicted_sink_logits.shape != (sink_tokens,):
        raise ValueError("predicted_sink_logits must match sink_tokens")
    if keys.shape[0] > block_seq or keys.shape[1] > block_dim:
        raise ValueError("increase block sizes for this synthetic gate")

    out = torch.empty((), dtype=torch.float32)
    _approx_sink_attention_scalar_kernel[(1,)](
        query.contiguous().to(torch.float32),
        keys.contiguous().to(torch.float32),
        values.contiguous().to(torch.float32),
        predicted_sink_logits.contiguous().to(torch.float32),
        out,
        keys.shape[0],
        keys.shape[1],
        sink_tokens,
        BLOCK_SEQ=block_seq,
        BLOCK_DIM=block_dim,
    )
    return out


def exact_scalar_attention_reference(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    logits = keys.float() @ query.float() / math.sqrt(query.numel())
    return torch.softmax(logits, dim=0) @ values.float()
