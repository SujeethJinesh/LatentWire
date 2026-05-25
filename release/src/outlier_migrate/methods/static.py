"""Static top-k channel protection."""

from __future__ import annotations

import numpy as np

from outlier_migrate.methods import register_method


def static_topk(scores: np.ndarray, budget: int) -> np.ndarray:
    """Protect the top-k channels by calibration score."""

    if budget <= 0 or budget > scores.size:
        raise ValueError("budget must be in [1, channel_count]")
    winners = np.argpartition(np.asarray(scores), -budget)[-budget:]
    mask = np.zeros(scores.size, dtype=bool)
    mask[winners] = True
    return mask


register_method("static", static_topk)
