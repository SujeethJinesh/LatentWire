from experimental.outlier_migrate.phase9 import check_om_phase9_m10_position_binned_scales as checker


def test_m10_decision_passes_when_all_thresholds_separate() -> None:
    decision, _reasons = checker.decision_from_summaries(
        {
            "median_recovery": 0.55,
            "bootstrap_ci95": {"ci95_low": 0.25, "ci95_high": 0.70},
        },
        {"median_recovery": 0.35},
        {"median_recovery": 0.30},
    )

    assert decision == checker.PASS_DECISION


def test_m10_decision_kills_if_random_control_beats() -> None:
    decision, reasons = checker.decision_from_summaries(
        {
            "median_recovery": 0.20,
            "bootstrap_ci95": {"ci95_low": 0.05, "ci95_high": 0.35},
        },
        {"median_recovery": 0.10},
        {"median_recovery": 0.35},
    )

    assert decision == checker.KILL_RANDOM_CONTROL
    assert "random-bin scale control" in reasons[0]


def test_m10_decision_kills_no_improvement() -> None:
    decision, _reasons = checker.decision_from_summaries(
        {
            "median_recovery": 0.03,
            "bootstrap_ci95": {"ci95_low": 0.0, "ci95_high": 0.08},
        },
        {"median_recovery": 0.02},
        {"median_recovery": 0.01},
    )

    assert decision == checker.KILL_NO_IMPROVEMENT
