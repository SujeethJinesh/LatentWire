from __future__ import annotations

import csv
import json

from scripts import build_source_private_hellaswag_repair_systems_acceptance_card as card


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _fixture(tmp_path):
    fixed = tmp_path / "fixed.json"
    control = tmp_path / "control.json"
    score = tmp_path / "score.json"
    public = tmp_path / "public.json"
    train_score = tmp_path / "train_score.json"
    hidden = tmp_path / "hidden.json"
    top2 = tmp_path / "top2.json"
    hidden_innovation = tmp_path / "hidden_innovation.json"
    ring = tmp_path / "ring.json"
    slo = tmp_path / "slo.json"
    native = tmp_path / "native.json"
    cross = tmp_path / "cross.json"

    fixed_trace = {
        "batch64_cacheline_bytes_per_request": 5,
        "batch64_dma_bytes_per_request": 6,
        "peak_rss_mib": 123.0,
        "phase_timings_s": {"source_scoring": 10.0, "total_before_artifact_write": 11.0},
        "raw_payload_bytes_per_request": 2,
        "record_bytes_with_header_crc": 5,
        "single_request_cacheline_bytes": 64,
        "single_request_dma_bytes": 128,
        "source_kv_exposed": False,
        "source_text_exposed": False,
    }
    _write_json(
        fixed,
        {
            "eval_rows": 1024,
            "headline": {
                "matched_accuracy": 0.46,
                "matched_minus_same_byte_text": 0.07,
            },
            "systems_trace": fixed_trace,
        },
    )
    _write_json(
        control,
        {
            "headline": {
                "label_copy_threat_present": True,
                "matched_minus_source_label_text_copy": -0.002,
                "paired_ci95_vs_source_label_text_copy": {"ci95_low": -0.005, "ci95_high": 0.0},
                "source_label_text_copy_accuracy": 0.462,
            }
        },
    )
    _write_json(
        score,
        {
            "calibration_rows": 512,
            "heldout_eval_rows": 512,
            "headline": {
                "score_packet_heldout_accuracy": 0.466,
                "score_packet_minus_source_label_text_heldout": 0.001,
                "source_label_text_heldout_accuracy": 0.465,
                "top2_oracle_accuracy": 0.716,
                "top2_oracle_heldout_accuracy": 0.734,
            },
            "packet_contract": {"payload_bytes": 2},
            "source_model": {"latency_s": 10.0},
        },
    )
    _write_json(
        public,
        {
            "eval_rows": 1024,
            "headline": {
                "best_repair_accuracy": 0.413,
                "best_repair_minus_source_label_copy": -0.049,
                "source_label_copy_accuracy": 0.462,
                "source_top2_oracle_accuracy": 0.716,
            },
            "packet_contract": {"payload_bytes": 1, "source_kv_exposed": False, "source_text_exposed": False},
            "source_model": {"latency_s": 10.0},
            "timing": {"total_seconds": 2.0},
            "train_rows": 100,
        },
    )
    source_model = {
        "eval": {"latency_s": 10.0},
        "train": {"latency_s": 9.0},
    }
    _write_json(
        train_score,
        {
            "eval_rows": 1024,
            "headline": {
                "best_label_copy_eval_accuracy": 0.462,
                "selected_eval_accuracy": 0.447,
                "selected_minus_best_label_copy": -0.015,
                "selected_minus_trained_choice_bias_label_copy": -0.012,
                "source_top2_oracle_accuracy": 0.716,
                "trained_choice_bias_label_copy_eval_accuracy": 0.459,
            },
            "packet_contract": {
                "framed_record_bytes": 5,
                "raw_payload_bytes": 2,
                "source_kv_exposed": False,
                "source_text_exposed": False,
            },
            "scored_train_rows": 512,
            "source_model": source_model,
            "timing": {"total_seconds": 12.0},
        },
    )
    _write_json(
        hidden,
        {
            "eval_rows": 1024,
            "headline": {
                "hidden_packet_eval_accuracy": 0.413,
                "hidden_packet_minus_same_byte_text": 0.055,
                "hidden_packet_minus_source_label_copy": -0.049,
                "paired_ci95_hidden_packet_vs_source_label_copy": {"ci95_low": -0.086, "ci95_high": -0.013},
                "source_label_copy_eval_accuracy": 0.462,
                "source_top2_oracle_accuracy": 0.716,
            },
            "packet_contract": {
                "framed_record_bytes": 5,
                "raw_hidden_vector_transmitted": False,
                "raw_payload_bytes": 2,
                "raw_scores_transmitted": False,
                "source_kv_exposed": False,
                "source_text_exposed": False,
            },
            "scored_train_rows": 512,
            "source_model": {
                "hidden_eval": {"latency_s": 8.0},
                "hidden_train": {"latency_s": 7.0},
                "score_eval": {"latency_s": 10.0},
                "score_train": {"latency_s": 9.0},
            },
            "timing": {"total_seconds": 15.0},
        },
    )
    _write_json(
        top2,
        {
            "eval_rows": 1024,
            "headline": {
                "paired_ci95_selected_vs_best_label_copy": {"ci95_low": -0.042, "ci95_high": 0.019},
                "selected_eval_accuracy": 0.449,
                "selected_minus_source_label_copy": -0.013,
                "selected_minus_trained_choice_bias_label_copy": -0.010,
                "source_label_copy_eval_accuracy": 0.462,
                "source_top2_oracle_accuracy": 0.716,
                "trained_choice_bias_label_copy_eval_accuracy": 0.459,
            },
            "packet_contract": {
                "framed_record_bytes": 5,
                "raw_hidden_vector_transmitted": False,
                "raw_payload_bytes": 2,
                "raw_scores_transmitted": False,
                "source_kv_exposed": False,
                "source_text_exposed": False,
            },
            "scored_train_rows": 512,
            "source_model": {
                "hidden_eval": {"latency_s": 8.0},
                "hidden_train": {"latency_s": 7.0},
                "score_eval": {"latency_s": 10.0},
                "score_train": {"latency_s": 9.0},
            },
            "timing": {"total_seconds": 1.0},
        },
    )
    _write_json(
        hidden_innovation,
        {
            "eval_rows": 1024,
            "headline": {
                "best_label_copy_eval_accuracy": 0.462,
                "paired_ci95_selected_vs_best_label_copy": {"ci95_low": 0.010, "ci95_high": 0.061},
                "selected_eval_accuracy": 0.499,
                "selected_minus_best_label_copy": 0.037,
                "selected_minus_trained_choice_bias_label_copy": 0.040,
                "source_top2_oracle_accuracy": 0.716,
                "trained_choice_bias_label_copy_eval_accuracy": 0.459,
            },
            "packet_contract": {
                "framed_record_bytes": 5,
                "raw_hidden_vector_transmitted": False,
                "raw_payload_bytes": 2,
                "raw_scores_transmitted": False,
                "source_kv_exposed": False,
                "source_text_exposed": False,
            },
            "scored_train_rows": 512,
            "source_model": {
                "hidden_eval": {"latency_s": 8.0},
                "hidden_train": {"latency_s": 7.0},
                "score_eval": {"latency_s": 10.0},
                "score_train": {"latency_s": 9.0},
            },
            "timing": {"total_seconds": 1.2},
        },
    )
    _write_json(
        ring,
        {
            "headline": {
                "packet_batch64_dma_bytes_per_request": 6,
                "packet_batch64_line_bytes_per_request": 5,
                "packet_batch64_p50_ns_per_request": 0.7,
                "packet_batch64_p95_ns_per_request": 0.8,
                "packet_batch64_record_bytes": 5,
            }
        },
    )
    _write_json(slo, {"headline": {"pass_gate": True}})
    _write_json(native, {"headline": {"native_ready": False, "pending_native_rows": 5}})
    _write_json(
        cross,
        {
            "headline": {
                "min_qjl_1bit_ratio_vs_framed": 51.2,
                "native_systems_complete": False,
                "pass_gate": True,
            }
        },
    )
    return {
        "fixed_packet": fixed,
        "control_suite": control,
        "score_packet_headroom": score,
        "public_receiver_repair": public,
        "train_source_score_repair": train_score,
        "hidden_summary_repair": hidden,
        "top2_contrastive_repair": top2,
        "hidden_innovation_repair": hidden_innovation,
        "packet_ring": ring,
        "serving_slo": slo,
        "native_readiness": native,
        "cross_benchmark_systems": cross,
    }


def test_acceptance_card_promotes_hidden_innovation_and_writes_outputs(tmp_path) -> None:
    payload = card.build_acceptance_card(
        output_dir=tmp_path / "out",
        artifact_paths=_fixture(tmp_path),
    )

    assert payload["pass_gate"] is True
    assert payload["headline"]["method_gate_pass"] is True
    assert payload["headline"]["systems_audit_pass"] is True
    assert payload["headline"]["native_queue_allowed"] is False
    assert payload["headline"]["best_repair_row_id"] == "hidden_innovation_repair"
    assert payload["headline"]["best_delta_vs_source_label_copy"] >= 0.02
    assert (tmp_path / "out" / "hellaswag_repair_systems_acceptance_card.json").exists()
    assert (tmp_path / "out" / "hellaswag_repair_systems_acceptance_card.csv").exists()
    assert (tmp_path / "out" / "hellaswag_repair_systems_acceptance_card.md").exists()
    assert (tmp_path / "out" / "manifest.json").exists()


def test_rows_record_label_copy_oracle_and_systems_fields(tmp_path) -> None:
    payload = card.build_acceptance_card(
        output_dir=tmp_path / "out",
        artifact_paths=_fixture(tmp_path),
    )
    rows = {row["row_id"]: row for row in payload["rows"]}

    assert rows["fixed_packet_vs_source_label_copy"]["raw_payload_bytes"] == 2.0
    assert rows["fixed_packet_vs_source_label_copy"]["framed_record_bytes"] == 5.0
    assert rows["hidden_summary_repair"]["paired_ci95_high_vs_source_label_copy"] < 0.0
    assert rows["score_margin_packet"]["delta_vs_source_label_copy"] > 0.0
    assert rows["score_margin_packet"]["delta_vs_source_label_copy"] < 0.02
    assert rows["train_source_score_repair"]["trained_label_copy_accuracy"] == 0.459
    assert rows["train_source_score_repair"]["delta_vs_trained_label_copy"] == -0.012
    assert rows["top2_contrastive_switch_repair"]["trained_label_copy_accuracy"] == 0.459
    assert rows["top2_contrastive_switch_repair"]["delta_vs_trained_label_copy"] == -0.010
    assert rows["top2_contrastive_switch_repair"]["paired_ci95_high_vs_source_label_copy"] == 0.019
    assert rows["hidden_innovation_repair"]["method_gate_pass"] is True
    assert rows["hidden_innovation_repair"]["delta_vs_source_label_copy"] == 0.037
    assert rows["hidden_innovation_repair"]["delta_vs_trained_label_copy"] == 0.040
    assert rows["hidden_innovation_repair"]["paired_ci95_low_vs_source_label_copy"] == 0.010
    assert all(row["systems_audit_pass"] for row in rows.values())
    assert all(not row["source_text_exposed"] and not row["source_kv_exposed"] for row in rows.values())
    assert payload["headline"]["max_oracle_gap_remaining"] > 0.20
    assert payload["headline"]["trained_label_copy_control_rows"] == 3
    assert payload["headline"]["best_delta_vs_trained_label_copy"] == 0.040
    assert any(check["check"] == "trained_label_copy_control_available" and check["pass"] for check in payload["checks"])


def test_written_csv_is_parseable(tmp_path) -> None:
    card.build_acceptance_card(
        output_dir=tmp_path / "out",
        artifact_paths=_fixture(tmp_path),
    )

    with (tmp_path / "out" / "hellaswag_repair_systems_acceptance_card.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 7
    assert rows[0]["row_id"]
    assert rows[0]["systems_audit_pass"] == "true"
    assert "delta_vs_trained_label_copy" in rows[0]
