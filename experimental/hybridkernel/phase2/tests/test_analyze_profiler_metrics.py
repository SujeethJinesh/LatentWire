from __future__ import annotations

from experimental.hybridkernel.phase2.analyze_profiler_metrics import analyze


def test_pending_without_native_rows() -> None:
    result = analyze({"rows": [{"total_step_ms": None, "attention_ssm_boundary_ms": None}]})

    assert result["status"].startswith("PENDING")
    assert result["rows"] == []


def test_promotes_only_when_all_repeated_runs_clear_gate() -> None:
    result = analyze(
        {
            "rows": [
                {
                    "model": "granite",
                    "run_id": idx,
                    "total_step_ms": 100.0,
                    "attention_ssm_boundary_ms": 8.0,
                    "matched_non_boundary_ms": 2.0,
                    "recoverable_fraction": 0.60,
                }
                for idx in range(3)
            ]
        }
    )

    assert result["status"].startswith("PROMOTE")
    assert result["summary"]["granite"]["clears_3pct_gate_all_runs"] is True


def test_kills_when_recoverable_gain_is_tiny() -> None:
    result = analyze(
        {
            "rows": [
                {
                    "model": "granite",
                    "run_id": idx,
                    "total_step_ms": 100.0,
                    "attention_ssm_boundary_ms": 2.0,
                    "matched_non_boundary_ms": 1.0,
                    "recoverable_fraction": 0.30,
                }
                for idx in range(3)
            ]
        }
    )

    assert result["status"].startswith("KILL")


def test_rejects_invalid_recoverable_fraction() -> None:
    try:
        analyze(
            {
                "rows": [
                    {
                        "model": "granite",
                        "run_id": 0,
                        "total_step_ms": 100.0,
                        "attention_ssm_boundary_ms": 8.0,
                        "matched_non_boundary_ms": 2.0,
                        "recoverable_fraction": 1.5,
                    }
                ]
            }
        )
    except ValueError as exc:
        assert "recoverable_fraction" in str(exc)
    else:
        raise AssertionError("expected invalid recoverable_fraction to be rejected")
