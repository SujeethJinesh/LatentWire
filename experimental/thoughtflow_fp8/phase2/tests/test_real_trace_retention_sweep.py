from experimental.thoughtflow_fp8.phase2.real_trace_retention_sweep import _status


def test_retention_sweep_status_alive_only_with_positive_phase_band() -> None:
    rows = [
        {"phase_margin_vs_best_other": 0.02, "math_margin_vs_best_other": 0.0},
        {"phase_margin_vs_best_other": 0.06, "math_margin_vs_best_other": -0.01},
    ]

    assert _status(rows).startswith("ALIVE")


def test_retention_sweep_status_weakened_without_phase_margin() -> None:
    rows = [
        {"phase_margin_vs_best_other": 0.0, "math_margin_vs_best_other": 0.0},
        {"phase_margin_vs_best_other": -0.01, "math_margin_vs_best_other": 0.1},
    ]

    assert _status(rows).startswith("WEAKENED")
