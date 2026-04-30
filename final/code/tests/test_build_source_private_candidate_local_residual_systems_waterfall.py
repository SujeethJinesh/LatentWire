from __future__ import annotations

import json

from scripts.build_source_private_candidate_local_residual_systems_waterfall import (
    _packet_accounting,
    build_candidate_local_residual_systems_waterfall,
)


def _metric(accuracy: float, payload_bytes: float, payload_tokens: float) -> dict[str, float | int]:
    return {
        "accuracy": accuracy,
        "correct": int(accuracy * 16),
        "mean_payload_bytes": payload_bytes,
        "mean_payload_tokens": payload_tokens,
        "n": 16,
        "p50_latency_ms": 0.12 if payload_bytes else 0.01,
        "p95_latency_ms": 0.34 if payload_bytes else 0.02,
        "strict_accuracy": accuracy,
    }


def test_packet_accounting_reports_record_line_dma_and_batch_bytes() -> None:
    zero = _packet_accounting(0, batch_size=64, header_bytes=3)
    assert zero["record_bytes"] == 0.0
    packet = _packet_accounting(8, batch_size=64, header_bytes=3)
    assert packet["record_bytes"] == 11.0
    assert packet["single_request_cacheline_bytes"] == 64.0
    assert packet["single_request_dma_bytes"] == 128.0
    assert packet["batch_line_bytes_per_request"] == 11.0
    assert packet["batch_dma_bytes_per_request"] == 12.0


def test_candidate_local_residual_systems_waterfall_from_existing_summaries(tmp_path) -> None:
    run_dir = tmp_path / "run_n512"
    direction_dir = run_dir / "core_to_holdout"
    direction_dir.mkdir(parents=True)
    direction_summary = {
        "direction": "core_to_holdout",
        "surface_overlap_audit": {
            "calibration_eval_exact_id_overlap_count": 0,
            "exact_transformed_eval_surface_overlap_count": 0,
        },
        "atom_dictionary": ["empty", "guard", "default", "round"],
        "budget_summaries": [
            {
                "budget_bytes": 8,
                "target_accuracy": 0.25,
                "best_control_accuracy": 0.25,
                "learned_minus_target": 0.375,
                "learned_minus_best_control": 0.375,
                "paired_bootstrap_vs_target": {"ci95_low": 0.31, "ci95_high": 0.44, "mean": 0.375},
                "controls_ok": True,
                "pass_gate": True,
                "metrics": {
                    "target_only": _metric(0.25, 0, 0),
                    "learned_synonym_dictionary_packet": _metric(0.625, 8, 4),
                    "zero_source": _metric(0.25, 0, 0),
                    "random_same_byte": _metric(0.25, 8, 4),
                    "private_random_source_atoms": _metric(0.25, 8, 4),
                    "permuted_teacher_receiver": _metric(0.25, 8, 4),
                    "answer_only_text": _metric(0.25, 8, 1),
                    "structured_text_matched": _metric(0.25, 8, 1),
                },
            }
        ],
    }
    (direction_dir / "summary.json").write_text(json.dumps(direction_summary), encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "summary": {
                    "pass_gate": True,
                    "runs": [{"run_dir": str(run_dir)}],
                }
            }
        ),
        encoding="utf-8",
    )

    payload = build_candidate_local_residual_systems_waterfall(
        summary_path=summary_path,
        output_dir=tmp_path / "out",
        run_dirs=[run_dir],
        bench_run_dir=None,
    )

    assert payload["gate"] == "source_private_candidate_local_residual_systems_waterfall"
    assert payload["pass_gate"] is True
    assert payload["headline"]["n512_passing_packet_rows"] == 1
    assert payload["headline"]["packet_record_bytes"] == 11
    assert payload["headline"]["packet_batch_line_bytes_per_request"] == 11.0
    packet = next(row for row in payload["rows"] if row["condition"] == "learned_synonym_dictionary_packet")
    text = next(row for row in payload["rows"] if row["condition"] == "answer_only_text")
    assert packet["source_text_exposed"] is False
    assert packet["source_kv_exposed"] is False
    assert packet["resident_feature_bytes_per_example"] == 4 * 4 * 8
    assert packet["sparse_decode_read_bytes_per_request"] == 4 * 4 * 4 + 11.0
    assert text["source_text_exposed"] is True
    assert (tmp_path / "out" / "candidate_local_residual_systems_waterfall.json").exists()
    assert (tmp_path / "out" / "manifest.json").exists()
