from experimental.outlier_migrate.phase9 import check_om_phase9_m11b_budget_scaling as checker


def test_m11b_passes_when_high_budget_beats_static_top10() -> None:
    decision, _reasons, pass_regime = checker.decision_from_summaries(
        {
            "m11b_top1": {"median_recovery": 0.05},
            "m11b_top5": {"median_recovery": 0.32},
            "m11b_top10": {"median_recovery": 0.20},
            "static_top10": {"median_recovery": 0.10},
        }
    )

    assert decision == checker.PASS_DECISION
    assert pass_regime == "m11b_top5"


def test_m11b_kills_when_high_budgets_match_top1() -> None:
    decision, _reasons, pass_regime = checker.decision_from_summaries(
        {
            "m11b_top1": {"median_recovery": 0.04},
            "m11b_top5": {"median_recovery": 0.07},
            "m11b_top10": {"median_recovery": 0.08},
            "static_top10": {"median_recovery": 0.02},
        }
    )

    assert decision == checker.KILL_BUDGET_INSUFFICIENT
    assert pass_regime is None


def test_m11b_ambiguous_for_intermediate_budget_gain() -> None:
    decision, _reasons, _pass_regime = checker.decision_from_summaries(
        {
            "m11b_top1": {"median_recovery": 0.04},
            "m11b_top5": {"median_recovery": 0.20},
            "m11b_top10": {"median_recovery": 0.25},
            "static_top10": {"median_recovery": 0.15},
        }
    )

    assert decision == checker.AMBIGUOUS
