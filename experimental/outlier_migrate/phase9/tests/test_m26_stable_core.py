from experimental.outlier_migrate.phase9 import check_om_phase9_m26_stable_core as checker


def test_m26_passes_when_core_beats_static_and_random() -> None:
    decision, _reasons = checker.decision_from_summaries(
        {
            "m26_core": {"median_recovery": 0.42, "bootstrap_ci95": {"ci95_low": 0.12}},
            "core_current_top1_union": {"median_recovery": 0.40, "bootstrap_ci95": {"ci95_low": 0.10}},
            "random_matched_core": {"median_recovery": 0.10, "bootstrap_ci95": {"ci95_low": -0.1}},
        }
    )

    assert decision == checker.PASS_DECISION


def test_m26_kills_when_random_control_beats() -> None:
    decision, _reasons = checker.decision_from_summaries(
        {
            "m26_core": {"median_recovery": 0.20, "bootstrap_ci95": {"ci95_low": -0.1}},
            "core_current_top1_union": {"median_recovery": 0.18, "bootstrap_ci95": {"ci95_low": -0.1}},
            "random_matched_core": {"median_recovery": 0.35, "bootstrap_ci95": {"ci95_low": 0.0}},
        }
    )

    assert decision == checker.KILL_RANDOM_CONTROL_BEATS


def test_m26_kills_no_improvement_near_static() -> None:
    decision, _reasons = checker.decision_from_summaries(
        {
            "m26_core": {"median_recovery": 0.04, "bootstrap_ci95": {"ci95_low": -0.1}},
            "core_current_top1_union": {"median_recovery": 0.12, "bootstrap_ci95": {"ci95_low": -0.1}},
            "random_matched_core": {"median_recovery": 0.01, "bootstrap_ci95": {"ci95_low": -0.1}},
        }
    )

    assert decision == checker.KILL_NO_IMPROVEMENT


def test_m26_ambiguous_for_partial_nonpassing_signal() -> None:
    decision, _reasons = checker.decision_from_summaries(
        {
            "m26_core": {"median_recovery": 0.22, "bootstrap_ci95": {"ci95_low": -0.1}},
            "core_current_top1_union": {"median_recovery": 0.28, "bootstrap_ci95": {"ci95_low": -0.1}},
            "random_matched_core": {"median_recovery": 0.11, "bootstrap_ci95": {"ci95_low": -0.1}},
        }
    )

    assert decision == checker.AMBIGUOUS
