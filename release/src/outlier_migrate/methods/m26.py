"""M26 dense-grid union selector placeholder."""

from __future__ import annotations

from outlier_migrate.methods.m11b import select_union
from outlier_migrate.registry import MethodSpec


DENSE_GRID = (100, 500, 1000, 2000, 5000, 7500, 10000)


def make_m26() -> MethodSpec:
    """Create the dense-grid migration-aware selector."""

    return MethodSpec(
        name="m26",
        description="Union of top channels over a denser decode-position grid.",
        selector=lambda scores, budget: select_union(scores, budget, positions=DENSE_GRID),
        required_positions=DENSE_GRID,
        parameters={"positions": DENSE_GRID},
    )
