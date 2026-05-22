"""Static position-based channel selectors."""

from __future__ import annotations

from outlier_migrate.registry import MethodSpec, ScoreTable, validate_budget, validate_score_table


def make_static_topk(name: str, *, position: int) -> MethodSpec:
    """Create a top-k selector for one decode position."""

    return MethodSpec(
        name=name,
        description=f"Top channels at decode position {position}.",
        selector=lambda scores, budget: select_static_topk(scores, budget, position=position),
        required_positions=(position,),
        parameters={"position": position},
    )


def select_static_topk(scores_by_position: ScoreTable, budget: int, *, position: int) -> set[int]:
    """Select top channels at a fixed decode position."""

    validate_budget(budget)
    validate_score_table(scores_by_position, required_positions=(position,))
    values = scores_by_position[position]
    return set(_top_indices(values, budget))


def _top_indices(values: list[float], budget: int) -> list[int]:
    return sorted(range(len(values)), key=lambda index: (-float(values[index]), index))[:budget]
