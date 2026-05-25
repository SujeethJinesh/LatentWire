"""Metrics used by release analyses."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import mean, median


@dataclass(frozen=True)
class RecoverySummary:
    """Summary statistics for per-trace recovery values."""

    median_recovery: float
    mean_recovery: float
    ci95_low: float
    ci95_high: float
    trace_count: int


def perplexity(mean_nll: float) -> float:
    """Convert mean negative log likelihood to perplexity."""

    return float(math.exp(min(80.0, mean_nll)))


def recovery(bf16_perplexity: float, static_perplexity: float, candidate_perplexity: float) -> float:
    """Compute recovery relative to the BF16 versus static gap."""

    static_gap = static_perplexity - bf16_perplexity
    if static_gap <= 0.0:
        return 0.0
    return 1.0 - (candidate_perplexity - bf16_perplexity) / static_gap


def bootstrap_median_ci(values: list[float], *, samples: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Bootstrap a 95 percent interval for the median."""

    if not values:
        raise ValueError("values must not be empty")
    rng = random.Random(seed)
    medians: list[float] = []
    for _ in range(samples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        medians.append(float(median(sample)))
    medians.sort()
    low_index = int(0.025 * (len(medians) - 1))
    high_index = int(0.975 * (len(medians) - 1))
    return medians[low_index], medians[high_index]


def summarize_recovery(values: list[float], *, seed: int = 20260510) -> RecoverySummary:
    """Summarize per-trace recovery values."""

    if not values:
        raise ValueError("values must not be empty")
    ci_low, ci_high = bootstrap_median_ci(values, seed=seed)
    return RecoverySummary(
        median_recovery=float(median(values)),
        mean_recovery=float(mean(values)),
        ci95_low=ci_low,
        ci95_high=ci_high,
        trace_count=len(values),
    )
