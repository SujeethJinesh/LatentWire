"""Minimal pairwise rotation baseline utilities."""

from __future__ import annotations

import numpy as np


def givens_pair_rotation(x: np.ndarray, i: int, j: int, theta: float) -> np.ndarray:
    """Apply a Givens rotation to two columns of a matrix."""

    arr = np.array(x, copy=True, dtype=np.float32)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    xi = arr[:, i].copy()
    xj = arr[:, j].copy()
    arr[:, i] = c * xi - s * xj
    arr[:, j] = s * xi + c * xj
    return arr
