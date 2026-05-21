"""M11B static union selector placeholder."""

from __future__ import annotations

from outlier_migrate.registry import MethodSpec, ScoreTable


PRIMARY_GRID = (100, 1000, 5000, 10000)


def make_m11b() -> MethodSpec:
    """Create the primary migration-aware union selector."""

    return MethodSpec(
        name="m11b",
        description="Union of top channels over the primary migration grid.",
        selector=lambda scores, budget: select_union(scores, budget, positions=PRIMARY_GRID),
    )


def select_union(scores_by_position: ScoreTable, budget: int, *, positions: tuple[int, ...]) -> set[int]:
    """Select the union of per-position top-k channel sets."""

    selected: set[int] = set()
    for position in positions:
        values = scores_by_position[position]
        selected.update(sorted(range(len(values)), key=lambda index: (-float(values[index]), index))[:budget])
    return selected
