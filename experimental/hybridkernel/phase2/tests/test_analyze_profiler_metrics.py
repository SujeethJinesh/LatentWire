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


def _review_control_rows() -> list[dict[str, object]]:
    return [
        _native_row(
            100,
            model="granite-transformer-control",
            row_role="same_family_control",
            control_family="same_family_transformer_heavy_control",
            control_model_or_segment="same_family_transformer_heavy_control",
            attention_ssm_boundary_ms=2.0,
            matched_non_boundary_ms=2.0,
        ),
        _native_row(
            101,
            model="qwen-cross-family-falsification",
            row_role="cross_family_falsification",
            control_model_or_segment="cross_family_hybrid_control",
            attention_ssm_boundary_ms=2.0,
            matched_non_boundary_ms=2.0,
        ),
    ]


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
            + _review_control_rows()
        }
    )

    assert result["status"].startswith("PROMOTE")
    summary_row = next(iter(result["summary"].values()))
    assert summary_row["clears_3pct_gate_all_runs"] is True
    assert summary_row["same_family_control_rows"] == 1
    assert summary_row["cross_family_falsification_rows"] == 1
    assert summary_row["median_recoverable_gain_upper_bound"] == 0.036
    assert summary_row["bootstrap_ci95_recoverable_gain_upper_bound"]["low"] == 0.036


def test_unmatched_review_controls_do_not_promote_clearing_primary() -> None:
    unrelated_controls = [
        _native_row(
            100,
            row_role="same_family_control",
            control_model_or_segment="same_family_transformer_heavy_control",
            attention_ssm_boundary_ms=2.0,
            matched_non_boundary_ms=2.0,
            batch_shape={
                "batch_size": 1,
                "prefill_tokens": 128,
                "decode_tokens": 65,
                "requests": 16,
            },
        ),
        _native_row(
            101,
            model="qwen-cross-family-falsification",
            row_role="cross_family_falsification",
            control_model_or_segment="cross_family_hybrid_control",
            attention_ssm_boundary_ms=2.0,
            matched_non_boundary_ms=2.0,
            batch_shape={
                "batch_size": 1,
                "prefill_tokens": 128,
                "decode_tokens": 65,
                "requests": 16,
            },
        ),
    ]
    result = analyze({"rows": [_native_row(idx) for idx in range(3)] + unrelated_controls})

    assert result["status"].startswith("WEAKLY ALIVE")
    summary_row = next(
        row for row in result["summary"].values() if row["model"] == "granite"
    )
    assert summary_row["clears_3pct_gate_all_runs"] is True
    assert summary_row["same_family_control_rows"] == 0
    assert summary_row["cross_family_falsification_rows"] == 0


def test_primary_gate_clear_without_controls_is_not_promoted() -> None:
    result = analyze(
        {
            "rows": [
                _native_row(idx)
                for idx in range(3)
            ]
        }
    )

    assert result["status"].startswith("WEAKLY ALIVE")
    assert "same-family controls" in result["status"]
    summary_row = next(iter(result["summary"].values()))
    assert summary_row["clears_3pct_gate_all_runs"] is True


def test_duplicate_run_ids_do_not_clear_gate() -> None:
    result = analyze(
        {
            "rows": [
                _native_row(idx, run_id="same_trace")
                for idx in range(3)
            ]
            + _review_control_rows()
        }
    )

    summary_row = next(iter(result["summary"].values()))
    assert summary_row["distinct_run_ids"] == 1
    assert summary_row["clears_3pct_gate_all_runs"] is False
    assert not result["status"].startswith("PROMOTE")


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


def test_rejects_impossible_local_timings() -> None:
    invalid_cases = [
        (
            _native_row(0, attention_ssm_boundary_ms=101.0),
            "attention_ssm_boundary_ms cannot exceed total_step_ms",
        ),
        (
            _native_row(0, matched_non_boundary_ms=101.0),
            "matched_non_boundary_ms cannot exceed total_step_ms",
        ),
        (
            _native_row(0, time_window_ms={"start": 10.0, "end": 9.0}),
            "time_window_ms.end must exceed start",
        ),
        (
            _native_row(
                0,
                total_step_ms=1.0,
                attention_ssm_boundary_ms=0.5,
                matched_non_boundary_ms=0.2,
                time_window_ms={"start": 0.0, "end": 2.0},
            ),
            "time_window_ms duration cannot exceed total_step_ms",
        ),
    ]
    for row, expected in invalid_cases:
        try:
            analyze({"rows": [row]})
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"expected {expected} to be rejected")


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
