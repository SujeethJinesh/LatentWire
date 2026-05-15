from experimental.outlier_migrate.phase9 import check_om_phase9_m11_ema_drift as checker
from experimental.outlier_migrate.phase9 import run_om_phase9_m11_ema_drift as runner


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


def test_m11_ema_trajectory_selects_final_drifted_channel() -> None:
    rows = []
    for position in runner.UPDATE_POSITIONS:
        values = [0.0] * 100
        if position == 100:
            values[0] = 10.0
        else:
            values[1] = 10.0
        rows.append(
            {
                "layer_index": 0,
                "layer_name": "layer.0",
                "decode_position": position,
                "channel_magnitudes": values,
            }
        )

    protected_sets, trajectories = runner.build_ema_trajectories(rows)

    alpha_05 = protected_sets["regimes"]["m11_alpha_0_5"]["layers"]["0"]
    assert alpha_05["protected_channels"] == [1]
    assert protected_sets["regimes"]["static_1pct"]["layers"]["0"]["protected_channels"] == [0]
    assert protected_sets["regimes"]["random_walk_protection"]["layers"]["0"]["protected_count"] == 1
    assert trajectories["strict_set_leaving_summary"]["0"]["strict_set_leaving_fraction"] == 1.0


def test_m11_ema_trajectory_requires_100_position_evidence() -> None:
    rows = [
        {
            "layer_index": 0,
            "layer_name": "layer.0",
            "decode_position": 100,
            "channel_magnitudes": [1.0] * 100,
        }
    ]

    try:
        runner.build_ema_trajectories(rows)
    except RuntimeError as exc:
        assert "requires 100-position activation evidence" in str(exc)
    else:
        raise AssertionError("missing update positions should fail")
