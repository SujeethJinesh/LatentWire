from __future__ import annotations

from experimental.hybridkernel.phase2.analyze_profiler_metrics import analyze


def _native_row(idx: int, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "model": "granite",
        "run_id": idx,
        "total_step_ms": 100.0,
        "attention_ssm_boundary_ms": 8.0,
        "matched_non_boundary_ms": 2.0,
        "recoverable_fraction": 0.60,
        "dtype": "bfloat16",
        "cuda_graph_enabled": True,
        "batch_shape": {
            "batch_size": 1,
            "prefill_tokens": 128,
            "decode_tokens": 64,
            "requests": 16,
        },
        "control_model_or_segment": "matched_transformer_block",
    }
    row.update(overrides)
    return row


def test_pending_without_native_rows() -> None:
    result = analyze({"rows": [{"total_step_ms": None, "attention_ssm_boundary_ms": None}]})

    assert result["status"].startswith("PENDING")
    assert result["rows"] == []


def test_promotes_only_when_all_repeated_runs_clear_gate() -> None:
    result = analyze(
        {
            "rows": [
                _native_row(idx)
                for idx in range(3)
            ]
        }
    )

    assert result["status"].startswith("PROMOTE")
    summary_row = next(iter(result["summary"].values()))
    assert summary_row["clears_3pct_gate_all_runs"] is True
    assert summary_row["median_recoverable_gain_upper_bound"] == 0.036
    assert summary_row["bootstrap_ci95_recoverable_gain_upper_bound"]["low"] == 0.036


def test_kills_when_recoverable_gain_is_tiny() -> None:
    result = analyze(
        {
            "rows": [
                _native_row(
                    idx,
                    attention_ssm_boundary_ms=2.0,
                    matched_non_boundary_ms=1.0,
                    recoverable_fraction=0.30,
                )
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
                    _native_row(0, recoverable_fraction=1.5)
                ]
            }
        )
    except ValueError as exc:
        assert "recoverable_fraction" in str(exc)
    else:
        raise AssertionError("expected invalid recoverable_fraction to be rejected")


def test_rejects_missing_matched_control_and_run_config() -> None:
    for field in ["matched_non_boundary_ms", "recoverable_fraction", "batch_shape"]:
        row = _native_row(0)
        row[field] = None
        try:
            analyze({"rows": [row]})
        except ValueError as exc:
            assert field in str(exc)
        else:
            raise AssertionError(f"expected missing {field} to be rejected")
