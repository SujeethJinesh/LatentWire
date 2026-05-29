"""Reference protected-column correction for the systems envelope.

This is not used as a claimed positive method in the paper. It documents the
operation whose cost is modeled in cost_table.csv.
"""
from __future__ import annotations

import torch


def protected_column_correction(y_base: torch.Tensor, x: torch.Tensor, delta_w: torch.Tensor, columns: torch.Tensor) -> torch.Tensor:
    """Return y_base + x_P @ DeltaW_P.

    Args:
        y_base: [..., D_out] base W4A16/dequantized output.
        x: [..., D_in] activation vector in the rotated basis.
        delta_w: [D_in, D_out] residual weight matrix W_fp - W_q.
        columns: [K] input column indices selected by a future quality-passing selector.
    """
    x_p = x.index_select(-1, columns)
    delta_p = delta_w.index_select(0, columns)
    return y_base + x_p @ delta_p
