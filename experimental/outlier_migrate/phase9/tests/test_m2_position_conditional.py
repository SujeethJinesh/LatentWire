import json
from pathlib import Path

from experimental.outlier_migrate.phase9 import check_om_phase9_m2_position_conditional as checker


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_m2_decision_passes_when_all_thresholds_separate() -> None:
    decision, _reasons = checker.decision_from_summaries(
        {
            "median_recovery": 0.55,
            "bootstrap_ci95": {"ci95_low": 0.25, "ci95_high": 0.70},
        },
        {"median_recovery": 0.30},
        {"median_recovery": 0.35},
    )

    assert decision == checker.PASS_DECISION


def test_m2_decision_kills_if_random_control_beats() -> None:
    decision, reasons = checker.decision_from_summaries(
        {
            "median_recovery": 0.30,
            "bootstrap_ci95": {"ci95_low": 0.10, "ci95_high": 0.50},
        },
        {"median_recovery": 0.20},
        {"median_recovery": 0.45},
    )

    assert decision == checker.KILL_RANDOM_CONTROL
    assert "random-bin control" in reasons[0]


def test_m2_decision_kills_no_improvement() -> None:
    decision, _reasons = checker.decision_from_summaries(
        {
            "median_recovery": 0.03,
            "bootstrap_ci95": {"ci95_low": 0.0, "ci95_high": 0.08},
        },
        {"median_recovery": 0.02},
        {"median_recovery": 0.01},
    )

    assert decision == checker.KILL_NO_IMPROVEMENT
