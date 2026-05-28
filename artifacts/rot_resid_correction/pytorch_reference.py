"""PyTorch reference for rotated residual correction.

This is intentionally a reference implementation, not a kernel. It assumes the
caller has already produced a ParoQuant-dequantized weight and a column pool.
"""

from __future__ import annotations

import torch


def residual_column_scores(
    activation_ema_sq: torch.Tensor,
    delta_weight: torch.Tensor,
) -> torch.Tensor:
    """Return s_i = EMA(x_i^2) * ||DeltaW[:, i]||_2^2.

    Args:
        activation_ema_sq: shape [in_features].
        delta_weight: shape [out_features, in_features].
    """

    if activation_ema_sq.ndim != 1:
        raise ValueError("activation_ema_sq must be 1D")
    if delta_weight.ndim != 2:
        raise ValueError("delta_weight must be 2D")
    if delta_weight.shape[1] != activation_ema_sq.numel():
        raise ValueError("in_features mismatch")
    delta_col_norm_sq = delta_weight.float().pow(2).sum(dim=0)
    return activation_ema_sq.float() * delta_col_norm_sq


def select_residual_columns(
    activation_ema_sq: torch.Tensor,
    delta_weight: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Select protected input columns by residual score."""

    if k <= 0:
        raise ValueError("k must be positive")
    scores = residual_column_scores(activation_ema_sq, delta_weight)
    k = min(k, scores.numel())
    return torch.topk(scores, k=k, largest=True, sorted=True).indices


def residual_corrected_linear(
    x: torch.Tensor,
    weight_pq: torch.Tensor,
    delta_weight: torch.Tensor,
    protected_columns: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute y = x W_pq^T + x_P DeltaW_P^T.

    Args:
        x: shape [..., in_features].
        weight_pq: ParoQuant-dequantized weight, shape [out_features, in_features].
        delta_weight: W_fp - W_pq, same shape as weight_pq.
        protected_columns: 1D long tensor of input-column indices.
        bias: optional shape [out_features].
    """

    if weight_pq.shape != delta_weight.shape:
        raise ValueError("weight_pq and delta_weight shapes must match")
    if weight_pq.shape[-1] != x.shape[-1]:
        raise ValueError("input feature mismatch")
    protected_columns = protected_columns.to(device=x.device, dtype=torch.long)
    base = torch.nn.functional.linear(x, weight_pq, bias)
    x_p = x.index_select(dim=-1, index=protected_columns)
    delta_p = delta_weight.index_select(dim=-1, index=protected_columns)
    correction = torch.nn.functional.linear(x_p, delta_p, None)
    return base + correction
