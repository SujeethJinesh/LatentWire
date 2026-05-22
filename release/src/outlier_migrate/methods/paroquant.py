"""Position-aware rotation and budget-composition selector placeholder."""

from __future__ import annotations

from outlier_migrate.registry import MethodSpec, ScoreTable, validate_budget, validate_score_table


def make_paroquant() -> MethodSpec:
    """Create a simple position-rotation selector for extension tests."""

    return MethodSpec(
        name="paroquant",
        description="Round-robin budget composition across decode positions.",
        selector=select_round_robin,
    )


def select_round_robin(scores_by_position: ScoreTable, budget: int) -> set[int]:
    """Select channels by round-robin top ranks across positions."""

    validate_budget(budget)
    validate_score_table(scores_by_position)
    rankings = {
        position: sorted(range(len(values)), key=lambda index: (-float(values[index]), index))
        for position, values in scores_by_position.items()
    }
    selected: set[int] = set()
    rank = 0
    positions = sorted(rankings)
    while len(selected) < budget and positions:
        progressed = False
        for position in positions:
            ranking = rankings[position]
            if rank < len(ranking):
                selected.add(ranking[rank])
                progressed = True
                if len(selected) >= budget:
                    break
        if not progressed:
            break
        rank += 1
    return selected
