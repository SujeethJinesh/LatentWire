"""Decode-decayed magnitude selector placeholder."""

from __future__ import annotations

from outlier_migrate.registry import MethodSpec, ScoreTable


def make_decdec() -> MethodSpec:
    """Create a recency-weighted average selector."""

    return MethodSpec(
        name="decdec",
        description="Recency-weighted average over available decode positions.",
        selector=select_decayed_average,
    )


def select_decayed_average(scores_by_position: ScoreTable, budget: int) -> set[int]:
    """Select channels by recency-weighted mean score."""

    positions = sorted(scores_by_position)
    if not positions:
        return set()
    channel_count = len(scores_by_position[positions[0]])
    totals = [0.0] * channel_count
    weight_total = 0.0
    for rank, position in enumerate(positions, start=1):
        weight = float(rank)
        weight_total += weight
        for index, value in enumerate(scores_by_position[position]):
            totals[index] += weight * float(value)
    averaged = [value / weight_total for value in totals]
    return set(sorted(range(channel_count), key=lambda index: (-averaged[index], index))[:budget])
