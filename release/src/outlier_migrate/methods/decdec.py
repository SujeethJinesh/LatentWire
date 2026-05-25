"""Algorithmic DecDEC-style reactive top-k channel selection."""

from __future__ import annotations

import numpy as np

from outlier_migrate.methods import register_method
from outlier_migrate.methods.static import static_topk


def reactive_topk(current_magnitudes: np.ndarray, budget: int) -> np.ndarray:
    """Protect the current-step top-k channels without temporal smoothing."""

    return static_topk(current_magnitudes, budget)


register_method("decdec", reactive_topk)
