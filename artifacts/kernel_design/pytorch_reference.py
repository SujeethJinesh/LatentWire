#!/usr/bin/env python3
"""PyTorch references for future LatentWire kernel candidates.

These functions define semantics only. They do not implement or benchmark a
real GPU kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PolicyUpdateConfig:
    """Configuration for EMA plus hysteretic protected-mask update."""

    alpha: float
    enter_k: int
    exit_k: int
    max_k: int | None = None
    wjac_weight: float = 1.0


def _as_2d(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be rank-2 [rows, channels], got {tuple(tensor.shape)}")
    return tensor


def _validate_same_shape(name: str, tensor: torch.Tensor, expected: torch.Size) -> None:
    if tensor.shape != expected:
        raise ValueError(f"{name} shape {tuple(tensor.shape)} does not match {tuple(expected)}")


def _topk_mask(values: torch.Tensor, k: int) -> torch.Tensor:
    """Return a deterministic top-k mask along the last dimension.

    Ties keep the lower channel index first because stable argsort preserves
    the original channel order.
    """

    values = _as_2d("values", values)
    rows, channels = values.shape
    if k <= 0:
        return torch.zeros((rows, channels), dtype=torch.bool, device=values.device)
    k = min(int(k), channels)
    order = torch.argsort(values, dim=-1, descending=True, stable=True)
    selected = order[:, :k]
    mask = torch.zeros((rows, channels), dtype=torch.bool, device=values.device)
    return mask.scatter(-1, selected, True)


def ema_update_reference(
    ema_scores: torch.Tensor,
    current_scores: torch.Tensor,
    *,
    alpha: float,
    wjac_scores: torch.Tensor | None = None,
    wjac_weight: float = 1.0,
) -> torch.Tensor:
    """Update per-row channel EMA scores.

    The optional WJAC score is an additive input. Callers are responsible for
    calibrating it onto the same scale as ``current_scores``.
    """

    ema_scores = _as_2d("ema_scores", ema_scores)
    current_scores = _as_2d("current_scores", current_scores)
    _validate_same_shape("current_scores", current_scores, ema_scores.shape)
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    combined = current_scores
    if wjac_scores is not None:
        wjac_scores = _as_2d("wjac_scores", wjac_scores)
        _validate_same_shape("wjac_scores", wjac_scores, ema_scores.shape)
        combined = combined + float(wjac_weight) * wjac_scores

    work_dtype = torch.promote_types(torch.float32, torch.promote_types(ema_scores.dtype, combined.dtype))
    ema_work = ema_scores.to(work_dtype)
    combined_work = combined.to(work_dtype)
    return float(alpha) * combined_work + (1.0 - float(alpha)) * ema_work


def hysteresis_mask_update_reference(
    scores: torch.Tensor,
    prev_mask: torch.Tensor,
    *,
    enter_k: int,
    exit_k: int,
    max_k: int | None = None,
) -> torch.Tensor:
    """Update a protected mask with enter/retain hysteresis.

    A channel enters if it is in ``topk(scores, enter_k)``. A previously
    protected channel remains protected while it is in ``topk(scores, exit_k)``.
    If ``max_k`` is set, the final candidate mask is capped by score rank.
    """

    scores = _as_2d("scores", scores)
    prev_mask = _as_2d("prev_mask", prev_mask)
    _validate_same_shape("prev_mask", prev_mask, scores.shape)
    prev_mask = prev_mask.to(dtype=torch.bool)

    enter = _topk_mask(scores, enter_k)
    retain_window = _topk_mask(scores, exit_k)
    candidate = enter | (prev_mask & retain_window)
    if max_k is None:
        return candidate
    if int(max_k) < 0:
        raise ValueError("max_k must be non-negative")
    restricted_scores = scores.masked_fill(~candidate, -torch.inf)
    cap = _topk_mask(restricted_scores, int(max_k))
    return candidate & cap


def policy_update_reference(
    ema_scores: torch.Tensor,
    current_scores: torch.Tensor,
    prev_mask: torch.Tensor,
    config: PolicyUpdateConfig,
    *,
    wjac_scores: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference for the policy kernel candidate."""

    next_ema = ema_update_reference(
        ema_scores,
        current_scores,
        alpha=config.alpha,
        wjac_scores=wjac_scores,
        wjac_weight=config.wjac_weight,
    )
    next_mask = hysteresis_mask_update_reference(
        next_ema,
        prev_mask,
        enter_k=config.enter_k,
        exit_k=config.exit_k,
        max_k=config.max_k,
    )
    return next_ema, next_mask


def base_w4a16_dequant_reference(
    x: torch.Tensor,
    weight_dequant: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    accumulation_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reference for base W4A16 output using already-dequantized weights."""

    if x.ndim < 2:
        raise ValueError(f"x must have shape [..., in_features], got {tuple(x.shape)}")
    if weight_dequant.ndim != 2:
        raise ValueError("weight_dequant must have shape [out_features, in_features]")
    in_features = x.shape[-1]
    out_features, weight_in = weight_dequant.shape
    if int(in_features) != int(weight_in):
        raise ValueError(f"x last dim {in_features} does not match weight input dim {weight_in}")

    flat_x = x.reshape(-1, in_features).to(accumulation_dtype)
    y = flat_x @ weight_dequant.to(accumulation_dtype).t()
    if bias is not None:
        if bias.shape != (out_features,):
            raise ValueError(f"bias shape {tuple(bias.shape)} does not match ({out_features},)")
        y = y + bias.to(accumulation_dtype)
    return y.reshape(*x.shape[:-1], out_features)


def protected_column_correction_reference(
    x: torch.Tensor,
    weight_fp16: torch.Tensor,
    weight_dequant: torch.Tensor,
    protected_idx: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    base_out: torch.Tensor | None = None,
    accumulation_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Reference for y = base_W4A16(x) + x_P @ (W_fp16_P - W_dequant_P)^T."""

    if weight_fp16.ndim != 2 or weight_dequant.ndim != 2:
        raise ValueError("weights must have shape [out_features, in_features]")
    if weight_fp16.shape != weight_dequant.shape:
        raise ValueError("weight_fp16 and weight_dequant shapes must match")
    if x.ndim < 2:
        raise ValueError(f"x must have shape [..., in_features], got {tuple(x.shape)}")

    out_features, in_features = weight_fp16.shape
    if x.shape[-1] != in_features:
        raise ValueError(f"x last dim {x.shape[-1]} does not match weight input dim {in_features}")

    if output_dtype is None:
        output_dtype = x.dtype

    if base_out is None:
        base = base_w4a16_dequant_reference(
            x,
            weight_dequant,
            bias=bias,
            accumulation_dtype=accumulation_dtype,
        )
    else:
        if base_out.shape != (*x.shape[:-1], out_features):
            raise ValueError(
                f"base_out shape {tuple(base_out.shape)} does not match expected {(*x.shape[:-1], out_features)}"
            )
        base = base_out.to(accumulation_dtype)

    protected_idx = protected_idx.to(device=x.device, dtype=torch.long).flatten()
    if protected_idx.numel() == 0:
        return base.to(output_dtype)
    if torch.any(protected_idx < 0) or torch.any(protected_idx >= in_features):
        raise ValueError("protected_idx contains an out-of-range input channel")

    flat_x = x.reshape(-1, in_features)
    x_p = flat_x.index_select(-1, protected_idx).to(accumulation_dtype)
    w_fp16_p = weight_fp16.index_select(1, protected_idx).to(accumulation_dtype)
    w_dequant_p = weight_dequant.index_select(1, protected_idx).to(accumulation_dtype)
    delta_p = w_fp16_p - w_dequant_p
    correction = x_p @ delta_p.t()
    y = base.reshape(-1, out_features).to(accumulation_dtype) + correction
    return y.reshape(*x.shape[:-1], out_features).to(output_dtype)


def explicit_mixed_weight_reference(
    x: torch.Tensor,
    weight_fp16: torch.Tensor,
    weight_dequant: torch.Tensor,
    protected_idx: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    accumulation_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Slow oracle that explicitly restores protected columns before matmul."""

    mixed = weight_dequant.clone()
    protected_idx = protected_idx.to(device=mixed.device, dtype=torch.long).flatten()
    if protected_idx.numel() > 0:
        mixed[:, protected_idx] = weight_fp16[:, protected_idx]
    if output_dtype is None:
        output_dtype = x.dtype
    return base_w4a16_dequant_reference(
        x,
        mixed,
        bias=bias,
        accumulation_dtype=accumulation_dtype,
    ).to(output_dtype)


def _self_test() -> None:
    torch.manual_seed(20260528)

    ema = torch.zeros(2, 8)
    current = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        ],
        dtype=torch.float32,
    )
    prev = torch.tensor(
        [
            [False, False, False, False, False, True, False, False],
            [False, False, True, False, False, False, False, False],
        ]
    )
    _, mask = policy_update_reference(
        ema,
        current,
        prev,
        PolicyUpdateConfig(alpha=1.0, enter_k=2, exit_k=4, max_k=3),
    )
    assert mask.sum(dim=-1).tolist() == [3, 3]
    assert mask[0, 6] and mask[0, 7]
    assert mask[1, 0] and mask[1, 1]

    x = torch.randn(3, 5, 16, dtype=torch.float16)
    weight_fp16 = torch.randn(24, 16, dtype=torch.float16)
    noise = 0.02 * torch.randn(24, 16, dtype=torch.float16)
    weight_dequant = weight_fp16 + noise
    bias = torch.randn(24, dtype=torch.float16)
    protected_idx = torch.tensor([0, 3, 7, 15])
    corrected = protected_column_correction_reference(
        x,
        weight_fp16,
        weight_dequant,
        protected_idx,
        bias=bias,
        output_dtype=torch.float32,
    )
    oracle = explicit_mixed_weight_reference(
        x,
        weight_fp16,
        weight_dequant,
        protected_idx,
        bias=bias,
        output_dtype=torch.float32,
    )
    torch.testing.assert_close(corrected, oracle, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    _self_test()
    print("pytorch_reference self-test passed")
