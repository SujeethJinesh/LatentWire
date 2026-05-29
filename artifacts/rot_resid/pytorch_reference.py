#!/usr/bin/env python3
"""CPU reference for ParoQuant residual protected-column correction.

This file defines semantics only. It does not implement a CUDA/Triton kernel
and does not run GPU work.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ResidualSidecar:
    """Residual columns selected for protected-column correction."""

    protected_idx: torch.Tensor
    delta_columns: torch.Tensor


@dataclass(frozen=True)
class OverheadEstimate:
    """Simple analytical overhead estimate for residual correction."""

    base_flops: int
    correction_flops: int
    relative_flop_overhead: float
    sidecar_bytes: int
    activation_gather_bytes: int


def _validate_weight_pair(weight_fp: torch.Tensor, weight_pq: torch.Tensor) -> tuple[int, int]:
    if weight_fp.ndim != 2 or weight_pq.ndim != 2:
        raise ValueError("weight_fp and weight_pq must have shape [out_features, in_features]")
    if weight_fp.shape != weight_pq.shape:
        raise ValueError("weight_fp and weight_pq must have identical shapes")
    out_features, in_features = weight_fp.shape
    return int(out_features), int(in_features)


def _normalize_indices(protected_idx: torch.Tensor, in_features: int, *, device: torch.device) -> torch.Tensor:
    idx = protected_idx.to(device=device, dtype=torch.long).flatten()
    if idx.numel() == 0:
        return idx
    if torch.any(idx < 0) or torch.any(idx >= in_features):
        raise ValueError("protected_idx contains an out-of-range input channel")
    if torch.unique(idx).numel() != idx.numel():
        raise ValueError("protected_idx contains duplicate channels")
    return idx


def select_residual_columns(
    activation_second_moment: torch.Tensor,
    delta_weight: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Select columns by E[x_i^2] * ||DeltaW[:, i]||_2^2.

    Returns indices sorted by descending score. Stable sorting keeps lower
    channel indices first on ties for deterministic auditability.
    """

    if delta_weight.ndim != 2:
        raise ValueError("delta_weight must have shape [out_features, in_features]")
    if activation_second_moment.ndim != 1:
        raise ValueError("activation_second_moment must have shape [in_features]")
    in_features = delta_weight.shape[1]
    if activation_second_moment.shape[0] != in_features:
        raise ValueError("activation_second_moment length must match weight input dimension")
    if k < 0:
        raise ValueError("k must be non-negative")

    k = min(int(k), int(in_features))
    if k == 0:
        return torch.empty(0, dtype=torch.long, device=delta_weight.device)

    scores = activation_second_moment.to(torch.float64) * delta_weight.to(torch.float64).pow(2).sum(dim=0)
    order = torch.argsort(scores, descending=True, stable=True)
    return order[:k].to(torch.long)


def build_residual_sidecar(
    weight_fp: torch.Tensor,
    weight_pq: torch.Tensor,
    protected_idx: torch.Tensor,
) -> ResidualSidecar:
    """Build DeltaW sidecar columns for protected input channels."""

    _, in_features = _validate_weight_pair(weight_fp, weight_pq)
    idx = _normalize_indices(protected_idx, in_features, device=weight_fp.device)
    delta = (weight_fp - weight_pq).index_select(1, idx)
    return ResidualSidecar(protected_idx=idx, delta_columns=delta.contiguous())


def base_linear_reference(
    x: torch.Tensor,
    weight_pq: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    accumulation_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reference ParoQuant/base output using already-dequantized weights."""

    if x.ndim < 2:
        raise ValueError("x must have shape [..., in_features]")
    if weight_pq.ndim != 2:
        raise ValueError("weight_pq must have shape [out_features, in_features]")
    out_features, in_features = weight_pq.shape
    if x.shape[-1] != in_features:
        raise ValueError("x last dimension does not match weight input dimension")
    if bias is not None and bias.shape != (out_features,):
        raise ValueError("bias must have shape [out_features]")

    flat_x = x.reshape(-1, in_features).to(accumulation_dtype)
    out = flat_x @ weight_pq.to(accumulation_dtype).t()
    if bias is not None:
        out = out + bias.to(accumulation_dtype)
    return out.reshape(*x.shape[:-1], out_features)


def protected_column_correction(
    x: torch.Tensor,
    weight_pq: torch.Tensor,
    sidecar: ResidualSidecar,
    bias: torch.Tensor | None = None,
    *,
    base_out: torch.Tensor | None = None,
    accumulation_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute y = x W_pq^T + x_P DeltaW_P^T."""

    if x.ndim < 2:
        raise ValueError("x must have shape [..., in_features]")
    if weight_pq.ndim != 2:
        raise ValueError("weight_pq must have shape [out_features, in_features]")
    out_features, in_features = weight_pq.shape
    if x.shape[-1] != in_features:
        raise ValueError("x last dimension does not match weight input dimension")
    if sidecar.delta_columns.ndim != 2 or sidecar.delta_columns.shape[0] != out_features:
        raise ValueError("sidecar.delta_columns must have shape [out_features, protected_count]")
    idx = _normalize_indices(sidecar.protected_idx, in_features, device=x.device)
    if sidecar.delta_columns.shape[1] != idx.numel():
        raise ValueError("sidecar index count and delta column count differ")
    if output_dtype is None:
        output_dtype = x.dtype

    if base_out is None:
        base = base_linear_reference(
            x,
            weight_pq,
            bias=bias,
            accumulation_dtype=accumulation_dtype,
        )
    else:
        expected = (*x.shape[:-1], out_features)
        if base_out.shape != expected:
            raise ValueError(f"base_out shape {tuple(base_out.shape)} does not match {expected}")
        base = base_out.to(accumulation_dtype)

    if idx.numel() == 0:
        return base.to(output_dtype)

    flat_x = x.reshape(-1, in_features)
    x_p = flat_x.index_select(-1, idx).to(accumulation_dtype)
    delta_p = sidecar.delta_columns.to(device=x.device, dtype=accumulation_dtype)
    correction = x_p @ delta_p.t()
    out = base.reshape(-1, out_features).to(accumulation_dtype) + correction
    return out.reshape(*x.shape[:-1], out_features).to(output_dtype)


def explicit_mixed_weight_reference(
    x: torch.Tensor,
    weight_fp: torch.Tensor,
    weight_pq: torch.Tensor,
    protected_idx: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    accumulation_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Slow oracle that explicitly restores protected columns."""

    _, in_features = _validate_weight_pair(weight_fp, weight_pq)
    idx = _normalize_indices(protected_idx, in_features, device=weight_pq.device)
    mixed = weight_pq.clone()
    if idx.numel() > 0:
        mixed[:, idx] = weight_fp[:, idx]
    if output_dtype is None:
        output_dtype = x.dtype
    return base_linear_reference(
        x,
        mixed,
        bias=bias,
        accumulation_dtype=accumulation_dtype,
    ).to(output_dtype)


def estimate_overhead(
    *,
    rows: int,
    in_features: int,
    out_features: int,
    protected_count: int,
    activation_bytes: int = 2,
    delta_bytes: int = 2,
    index_bytes: int = 4,
) -> OverheadEstimate:
    """Estimate correction compute and sidecar memory."""

    if min(rows, in_features, out_features, protected_count) < 0:
        raise ValueError("shape/count arguments must be non-negative")
    base_flops = 2 * int(rows) * int(out_features) * int(in_features)
    correction_flops = 2 * int(rows) * int(out_features) * int(protected_count)
    relative = 0.0 if in_features == 0 else float(protected_count) / float(in_features)
    sidecar_bytes = int(out_features) * int(protected_count) * int(delta_bytes)
    sidecar_bytes += int(protected_count) * int(index_bytes)
    activation_gather_bytes = int(rows) * int(protected_count) * int(activation_bytes)
    return OverheadEstimate(
        base_flops=base_flops,
        correction_flops=correction_flops,
        relative_flop_overhead=relative,
        sidecar_bytes=sidecar_bytes,
        activation_gather_bytes=activation_gather_bytes,
    )


def _self_test() -> None:
    torch.manual_seed(20260528)

    x = torch.randn(4, 3, 16, dtype=torch.float16)
    weight_fp = torch.randn(24, 16, dtype=torch.float16)
    weight_pq = weight_fp + 0.03 * torch.randn(24, 16, dtype=torch.float16)
    bias = torch.randn(24, dtype=torch.float16)
    delta = weight_fp - weight_pq
    activation_second_moment = torch.linspace(0.5, 2.0, 16)

    selected = select_residual_columns(activation_second_moment, delta, k=4)
    assert selected.shape == (4,)
    assert torch.unique(selected).numel() == 4

    sidecar = build_residual_sidecar(weight_fp, weight_pq, selected)
    corrected = protected_column_correction(
        x,
        weight_pq,
        sidecar,
        bias=bias,
        output_dtype=torch.float32,
    )
    oracle = explicit_mixed_weight_reference(
        x,
        weight_fp,
        weight_pq,
        selected,
        bias=bias,
        output_dtype=torch.float32,
    )
    torch.testing.assert_close(corrected, oracle, atol=1e-4, rtol=1e-4)

    base = base_linear_reference(x, weight_pq, bias=bias)
    corrected_from_base = protected_column_correction(
        x,
        weight_pq,
        sidecar,
        base_out=base,
        output_dtype=torch.float32,
    )
    torch.testing.assert_close(corrected_from_base, oracle, atol=1e-4, rtol=1e-4)

    small = estimate_overhead(rows=8, in_features=16, out_features=24, protected_count=2)
    large = estimate_overhead(rows=8, in_features=16, out_features=24, protected_count=4)
    assert small.relative_flop_overhead == 0.125
    assert large.relative_flop_overhead == 0.25
    assert large.correction_flops > small.correction_flops
    assert large.sidecar_bytes > small.sidecar_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run CPU reference self-test")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        print("kernel_spec pytorch_reference self-test passed")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
