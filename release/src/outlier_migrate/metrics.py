"""Metrics used by the release reproduction scripts."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def recovery_fraction(bf16: float, static: float, candidate: float) -> float:
    """Compute recovery of the BF16-vs-static gap."""

    gap = static - bf16
    if math.isclose(gap, 0.0, abs_tol=1e-12):
        raise ValueError("Recovery is undefined when static and BF16 match")
    return 1.0 - ((candidate - bf16) / gap)


def set_leaving_rate(base_topk: set[int], later_topk: set[int]) -> float:
    """Return fraction of base channels absent from the later top-k set."""

    if not base_topk:
        raise ValueError("base_topk must be non-empty")
    return len(base_topk - later_topk) / len(base_topk)


def kl_divergence(p: Sequence[float], q: Sequence[float], eps: float = 1e-12) -> float:
    """Compute KL(P||Q) for probability vectors."""

    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    if p_arr.shape != q_arr.shape:
        raise ValueError("p and q must have the same shape")
    p_arr = np.clip(p_arr, eps, None)
    q_arr = np.clip(q_arr, eps, None)
    p_arr = p_arr / p_arr.sum()
    q_arr = q_arr / q_arr.sum()
    return float(np.sum(p_arr * np.log(p_arr / q_arr)))


def bootstrap_ci(values: Sequence[float], seed: int, samples: int = 1000) -> tuple[float, float]:
    """Compute a percentile bootstrap CI for the sample median."""

    if not values:
        raise ValueError("values must be non-empty")
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    medians = [float(np.median(rng.choice(arr, size=arr.size, replace=True))) for _ in range(samples)]
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))
