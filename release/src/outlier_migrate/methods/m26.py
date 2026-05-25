"""Stable-core channel protection."""

from __future__ import annotations

import numpy as np

from outlier_migrate.methods import register_method


def stable_core_mask(stability_counts: np.ndarray, budget: int) -> np.ndarray:
    """Protect channels with highest all-position stability counts."""

    if budget <= 0 or budget > stability_counts.size:
        raise ValueError("budget must be in [1, channel_count]")
    winners = np.argpartition(np.asarray(stability_counts), -budget)[-budget:]
    mask = np.zeros(stability_counts.size, dtype=bool)
    mask[winners] = True
    return mask


register_method("m26", stable_core_mask)
