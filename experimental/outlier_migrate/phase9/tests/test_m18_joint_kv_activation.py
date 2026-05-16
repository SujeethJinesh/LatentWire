from experimental.outlier_migrate.phase9 import check_om_phase9_m18_joint_kv_activation as checker


def test_m18_decision_passes_when_primary_beats_controls() -> None:
    decision, _reasons = checker.decision_from_summaries(
        {
            "median_recovery": 0.45,
            "bootstrap_ci95": {"ci95_low": 0.20, "ci95_high": 0.60},
        },
        {"median_recovery": 0.30},
        {"median_recovery": 0.20},
        {"key_cache_attention_layer_coverage": 1.0},
    )

    assert decision == checker.PASS_DECISION


def test_m18_decision_kills_when_kv_only_not_beaten() -> None:
    decision, reasons = checker.decision_from_summaries(
        {
            "median_recovery": 0.25,
            "bootstrap_ci95": {"ci95_low": 0.05, "ci95_high": 0.40},
        },
        {"median_recovery": 0.23},
        {"median_recovery": 0.05},
        {"key_cache_attention_layer_coverage": 1.0},
    )

    assert decision == checker.KILL_KV_ONLY_NOT_BEATEN
    assert "KIVI-style" in reasons[0]


def test_m18_decision_kills_when_random_control_beats() -> None:
    decision, reasons = checker.decision_from_summaries(
        {
            "median_recovery": 0.20,
            "bootstrap_ci95": {"ci95_low": 0.02, "ci95_high": 0.35},
        },
        {"median_recovery": 0.05},
        {"median_recovery": 0.35},
        {"key_cache_attention_layer_coverage": 1.0},
    )

    assert decision == checker.KILL_RANDOM_CONTROL
    assert "random-coupled" in reasons[0]


def test_m18_decision_kills_no_improvement() -> None:
    decision, _reasons = checker.decision_from_summaries(
        {
            "median_recovery": 0.03,
            "bootstrap_ci95": {"ci95_low": 0.0, "ci95_high": 0.10},
        },
        {"median_recovery": -0.10},
        {"median_recovery": -0.20},
        {"key_cache_attention_layer_coverage": 1.0},
    )

    assert decision == checker.KILL_NO_IMPROVEMENT
