"""Budget-tuned EMA channel protection."""

from __future__ import annotations

import numpy as np

from outlier_migrate.methods import register_method
from outlier_migrate.methods.static import static_topk


def ema_scores(previous: np.ndarray, current: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Update EMA channel scores."""

    if previous.shape != current.shape:
        raise ValueError("previous and current scores must have the same shape")
    return alpha * current + (1.0 - alpha) * previous


def m11b_topk(scores: np.ndarray, budget: int) -> np.ndarray:
    """Protect top-k EMA scores; scripts choose the budget."""

    return static_topk(scores, budget)


register_method("m11b", m11b_topk)
