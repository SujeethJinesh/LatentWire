from experimental.outlier_migrate.phase9 import check_om_phase9_m11_ema_drift as checker


def test_m11_decision_passes_if_any_alpha_passes() -> None:
    decision, reasons, best_alpha = checker.decision_from_summaries(
        {
            "m11_alpha_0_1": {
                "median_recovery": 0.20,
                "bootstrap_ci95": {"ci95_low": 0.05, "ci95_high": 0.30},
            },
            "m11_alpha_0_3": {
                "median_recovery": 0.42,
                "bootstrap_ci95": {"ci95_low": 0.18, "ci95_high": 0.60},
            },
            "m11_alpha_0_5": {
                "median_recovery": 0.10,
                "bootstrap_ci95": {"ci95_low": 0.0, "ci95_high": 0.20},
            },
        },
        {"median_recovery": 0.18},
    )

    assert decision == checker.PASS_DECISION
    assert best_alpha == "m11_alpha_0_3"
    assert "m11_alpha_0_3" in reasons[0]


def test_m11_decision_kills_if_random_walk_beats_all_alphas() -> None:
    decision, _reasons, best_alpha = checker.decision_from_summaries(
        {
            "m11_alpha_0_1": {
                "median_recovery": 0.05,
                "bootstrap_ci95": {"ci95_low": 0.0, "ci95_high": 0.12},
            },
            "m11_alpha_0_3": {
                "median_recovery": 0.10,
                "bootstrap_ci95": {"ci95_low": 0.0, "ci95_high": 0.20},
            },
            "m11_alpha_0_5": {
                "median_recovery": -0.05,
                "bootstrap_ci95": {"ci95_low": -0.10, "ci95_high": 0.05},
            },
        },
        {"median_recovery": 0.25},
    )

    assert decision == checker.KILL_RANDOM_CONTROL
    assert best_alpha == "m11_alpha_0_3"


def test_m11_decision_kills_no_improvement() -> None:
    decision, _reasons, best_alpha = checker.decision_from_summaries(
        {
            "m11_alpha_0_1": {
                "median_recovery": 0.01,
                "bootstrap_ci95": {"ci95_low": 0.0, "ci95_high": 0.05},
            },
            "m11_alpha_0_3": {
                "median_recovery": 0.04,
                "bootstrap_ci95": {"ci95_low": 0.0, "ci95_high": 0.08},
            },
            "m11_alpha_0_5": {
                "median_recovery": -0.02,
                "bootstrap_ci95": {"ci95_low": -0.05, "ci95_high": 0.02},
            },
        },
        {"median_recovery": -0.10},
    )

    assert decision == checker.KILL_NO_IMPROVEMENT
    assert best_alpha == "m11_alpha_0_3"
