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
        "row_role": "primary_hybrid",
        "control_family": "same_family_matched_segment",
        "boundary_direction": "mixed_attention_ssm",
        "nsys_artifact": "nsys/granite_tiny_b1_decode64.nsys-rep",
        "ncu_artifact": "ncu/suspicious_boundary_kernel.ncu-rep",
        "kernel_names": ["synthetic_boundary_kernel"],
        "boundary_indices": [0],
        "time_window_ms": {"start": float(idx), "end": float(idx) + 1.0},
        "reduction_notes": "Reduced from synthetic fixture row.",
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


def test_rejects_string_cuda_graph_state() -> None:
    try:
        analyze({"rows": [_native_row(0, cuda_graph_enabled="False")]})
    except ValueError as exc:
        assert "cuda_graph_enabled" in str(exc)
    else:
        raise AssertionError("expected string cuda_graph_enabled to be rejected")


def test_rejects_empty_control_segment() -> None:
    try:
        analyze({"rows": [_native_row(0, control_model_or_segment="  ")]})
    except ValueError as exc:
        assert "control_model_or_segment" in str(exc)
    else:
        raise AssertionError("expected empty control segment to be rejected")


def test_rejects_non_positive_batch_shape_fields() -> None:
    row = _native_row(0)
    row["batch_shape"] = {
        "batch_size": 0,
        "prefill_tokens": 128,
        "decode_tokens": 64,
        "requests": 16,
    }
    try:
        analyze({"rows": [row]})
    except ValueError as exc:
        assert "batch_shape.batch_size" in str(exc)
    else:
        raise AssertionError("expected non-positive batch size to be rejected")


def test_rejects_missing_or_empty_run_id() -> None:
    for run_id in [None, "  "]:
        row = _native_row(0)
        row["run_id"] = run_id
        try:
            analyze({"rows": [row]})
        except ValueError as exc:
            assert "run_id" in str(exc)
        else:
            raise AssertionError("expected missing or empty run_id to be rejected")


def test_rejects_missing_or_empty_model() -> None:
    for model in [None, "  "]:
        row = _native_row(0)
        row["model"] = model
        try:
            analyze({"rows": [row]})
        except ValueError as exc:
            assert "model" in str(exc)
        else:
            raise AssertionError("expected missing or empty model to be rejected")


def test_rejects_non_integer_batch_shape_fields() -> None:
    row = _native_row(0)
    row["batch_shape"] = {
        "batch_size": 1.5,
        "prefill_tokens": 128,
        "decode_tokens": 64,
        "requests": 16,
    }
    try:
        analyze({"rows": [row]})
    except ValueError as exc:
        assert "batch_shape.batch_size" in str(exc)
    else:
        raise AssertionError("expected non-integer batch size to be rejected")
