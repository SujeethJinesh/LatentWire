from __future__ import annotations

import json

from scripts.build_source_private_arc_challenge_systems_trace import build_arc_challenge_systems_trace


def _write(path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _gate_payload(*, n: int, matched: float, target: float, text: float) -> dict:
    return {
        "eval_rows": n,
        "pass_gate": True,
        "headline": {
            "matched_accuracy": matched,
            "target_accuracy": target,
            "same_byte_structured_text_accuracy": text,
            "best_destructive_control_accuracy": target,
            "paired_ci95_vs_target": {"ci95_low": 0.04},
        },
        "condition_metrics": {
            "matched_source_private_packet": {
                "p50_latency_ms": 0.03,
                "p95_latency_ms": 0.10,
            },
            "same_byte_structured_text": {
                "p50_latency_ms": 0.01,
                "p95_latency_ms": 0.02,
            },
        },
        "systems_trace": {
            "source_private": True,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_payload_bytes_per_request": 12,
            "record_bytes_with_header_crc": 15,
            "single_request_cacheline_bytes": 64.0,
            "single_request_dma_bytes": 128.0,
            "batch64_cacheline_bytes_per_request": 15.0,
            "batch64_dma_bytes_per_request": 16.0,
            "eval_candidate_pairs": n * 4,
            "feature_cache_bytes_eval_float64": n * 4 * 384 * 8,
            "feature_cache_bytes_eval_float32_floor": n * 4 * 384 * 4,
            "projection_matrix_bytes_float64": 384 * 96 * 8,
            "peak_rss_mib": 7000.0,
            "phase_timings_s": {
                "source_scoring": n * 0.25,
                "packet_encode_decode_all_conditions": 0.50,
            },
        },
    }


def test_arc_challenge_systems_trace_marks_native_rows_pending(tmp_path) -> None:
    validation = tmp_path / "validation.json"
    test = tmp_path / "test.json"
    seed_validation = tmp_path / "seed_validation.json"
    seed_test = tmp_path / "seed_test.json"
    _write(validation, _gate_payload(n=299, matched=0.388, target=0.244, text=0.348))
    _write(test, _gate_payload(n=1172, matched=0.344, target=0.265, text=0.311))
    _write(seed_validation, {"aggregate": {"pass_count": 5}})
    _write(seed_test, {"aggregate": {"pass_count": 5}})

    payload = build_arc_challenge_systems_trace(
        validation_artifact=validation,
        test_artifact=test,
        seed_validation_artifact=seed_validation,
        seed_test_artifact=seed_test,
        output_dir=tmp_path / "out",
    )

    assert payload["pass_gate"] is True
    assert payload["headline"]["colm_systems_trace_ready"] is True
    assert payload["headline"]["iclr_native_systems_complete"] is False
    assert payload["headline"]["test_source_scoring_ms_per_question"] == 250.0
    rows = {row["row_id"]: row for row in payload["rows"]}
    assert rows["arc_shared_basis_test"]["source_private"] is True
    assert rows["arc_shared_basis_test"]["record_bytes"] == 15.0
    assert rows["arc_shared_basis_test"]["batch64_dma_bytes_per_request"] == 16.0
    assert rows["pending_c2c_native"]["source_kv_exposed"] is True
    assert rows["pending_vllm_serving"]["native_kernel_status"] == "pending_native_required"
    assert (tmp_path / "out" / "arc_challenge_systems_trace.json").exists()
    assert (tmp_path / "out" / "arc_challenge_systems_trace.csv").exists()
