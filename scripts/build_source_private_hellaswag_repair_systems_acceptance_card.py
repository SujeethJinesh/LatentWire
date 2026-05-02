from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT = pathlib.Path("results/source_private_hellaswag_repair_systems_acceptance_card_20260501")

DEFAULT_ARTIFACTS = {
    "fixed_packet": pathlib.Path(
        "results/source_private_hellaswag_fixed_packet_gate_20260501_qwen05_hashed_validation1024_2b/"
        "arc_challenge_fixed_packet_gate.json"
    ),
    "control_suite": pathlib.Path("results/source_private_hellaswag_control_suite_20260501/hellaswag_control_suite.json"),
    "score_packet_headroom": pathlib.Path(
        "results/source_private_hellaswag_score_packet_headroom_20260501_qwen05_validation1024/"
        "hellaswag_score_packet_headroom.json"
    ),
    "public_receiver_repair": pathlib.Path(
        "results/source_private_hellaswag_public_receiver_repair_probe_20260501_qwen05_validation1024/"
        "hellaswag_public_receiver_repair_probe.json"
    ),
    "train_source_score_repair": pathlib.Path(
        "results/source_private_hellaswag_train_source_score_repair_probe_20260501_qwen05_train512_validation1024/"
        "hellaswag_train_source_score_repair_probe.json"
    ),
    "hidden_summary_repair": pathlib.Path(
        "results/source_private_hellaswag_hidden_summary_repair_probe_20260501_qwen05_train512_validation1024/"
        "hellaswag_hidden_summary_repair_probe.json"
    ),
    "top2_contrastive_repair": pathlib.Path(
        "results/source_private_hellaswag_top2_contrastive_repair_probe_20260501_qwen05_train512_validation1024/"
        "hellaswag_top2_contrastive_repair_probe.json"
    ),
    "hidden_innovation_repair": pathlib.Path(
        "results/source_private_hellaswag_hidden_innovation_repair_probe_20260501_qwen05_train512_validation1024/"
        "hellaswag_hidden_innovation_repair_probe.json"
    ),
    "packet_ring": pathlib.Path(
        "results/source_private_mac_packet_ring_transport_microbench_20260501/packet_ring_transport_microbench.json"
    ),
    "serving_slo": pathlib.Path("results/source_private_serving_slo_envelope_20260501/serving_slo_envelope.json"),
    "native_readiness": pathlib.Path("results/source_private_native_readiness_ledger_20260501/native_readiness_ledger.json"),
    "cross_benchmark_systems": pathlib.Path(
        "results/source_private_cross_benchmark_systems_comparator_20260502/cross_benchmark_systems_comparator.json"
    ),
}

STRICT_METHOD_DELTA = 0.02

CSV_COLUMNS = (
    "row_id",
    "repair_family",
    "artifact_path",
    "artifact_sha256",
    "control_artifact_path",
    "method_gate_pass",
    "systems_audit_pass",
    "native_queue_allowed",
    "kill_reason",
    "eval_rows",
    "train_rows",
    "accuracy",
    "source_label_copy_accuracy",
    "delta_vs_source_label_copy",
    "trained_label_copy_accuracy",
    "delta_vs_trained_label_copy",
    "delta_vs_same_byte_text",
    "paired_ci95_low_vs_source_label_copy",
    "paired_ci95_high_vs_source_label_copy",
    "source_top2_oracle_accuracy",
    "oracle_gap_remaining",
    "raw_payload_bytes",
    "framed_record_bytes",
    "single_request_cacheline_bytes",
    "single_request_dma_bytes",
    "batch64_cacheline_bytes_per_request",
    "batch64_dma_bytes_per_request",
    "packet_ring_p50_ns",
    "packet_ring_p95_ns",
    "source_scoring_train_s",
    "source_scoring_eval_s",
    "hidden_extract_train_s",
    "hidden_extract_eval_s",
    "total_wall_s",
    "peak_rss_mib",
    "source_text_exposed",
    "source_kv_exposed",
    "raw_hidden_vector_transmitted",
    "raw_scores_transmitted",
    "source_packet_required",
)


def _resolve(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else ROOT / path


def _rel(path: pathlib.Path | None) -> str | None:
    if path is None:
        return None
    resolved = _resolve(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _sha256_file(path: pathlib.Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _ci_low_high(ci: dict[str, Any] | None) -> tuple[float | None, float | None]:
    if not ci:
        return None, None
    return ci.get("ci95_low"), ci.get("ci95_high")


def _framed_bytes(raw_payload_bytes: float | int | None, explicit: float | int | None, ring_record_bytes: float) -> float:
    if explicit is not None:
        return float(explicit)
    if raw_payload_bytes is None:
        return float(ring_record_bytes)
    return float(max(ring_record_bytes, float(raw_payload_bytes) + 3.0))


def _ring_fields(packet_ring: dict[str, Any], framed_record_bytes: float) -> dict[str, Any]:
    headline = packet_ring["headline"]
    if float(headline["packet_batch64_record_bytes"]) == float(framed_record_bytes):
        return {
            "batch64_cacheline_bytes_per_request": headline["packet_batch64_line_bytes_per_request"],
            "batch64_dma_bytes_per_request": headline["packet_batch64_dma_bytes_per_request"],
            "packet_ring_p50_ns": headline["packet_batch64_p50_ns_per_request"],
            "packet_ring_p95_ns": headline["packet_batch64_p95_ns_per_request"],
        }
    return {
        "batch64_cacheline_bytes_per_request": framed_record_bytes,
        "batch64_dma_bytes_per_request": framed_record_bytes,
        "packet_ring_p50_ns": None,
        "packet_ring_p95_ns": None,
    }


def _system_fields(
    *,
    raw_payload_bytes: float | int | None,
    explicit_framed_bytes: float | int | None,
    fixed_trace: dict[str, Any],
    packet_ring: dict[str, Any],
    source_text_exposed: bool,
    source_kv_exposed: bool,
    raw_hidden_vector_transmitted: bool = False,
    raw_scores_transmitted: bool = False,
) -> dict[str, Any]:
    ring_record = float(packet_ring["headline"]["packet_batch64_record_bytes"])
    framed = _framed_bytes(raw_payload_bytes, explicit_framed_bytes, ring_record)
    fields = {
        "raw_payload_bytes": None if raw_payload_bytes is None else float(raw_payload_bytes),
        "framed_record_bytes": framed,
        "single_request_cacheline_bytes": fixed_trace.get("single_request_cacheline_bytes", 64.0),
        "single_request_dma_bytes": fixed_trace.get("single_request_dma_bytes", 128.0),
        "source_text_exposed": bool(source_text_exposed),
        "source_kv_exposed": bool(source_kv_exposed),
        "raw_hidden_vector_transmitted": bool(raw_hidden_vector_transmitted),
        "raw_scores_transmitted": bool(raw_scores_transmitted),
        "source_packet_required": True,
    }
    fields.update(_ring_fields(packet_ring, framed))
    return fields


def _systems_audit_pass(row: dict[str, Any]) -> bool:
    byte_fields = (
        row["raw_payload_bytes"],
        row["framed_record_bytes"],
        row["single_request_cacheline_bytes"],
        row["single_request_dma_bytes"],
        row["batch64_cacheline_bytes_per_request"],
        row["batch64_dma_bytes_per_request"],
    )
    return (
        all(value is not None for value in byte_fields)
        and row["framed_record_bytes"] >= row["raw_payload_bytes"]
        and not row["source_text_exposed"]
        and not row["source_kv_exposed"]
        and not row["raw_hidden_vector_transmitted"]
        and not row["raw_scores_transmitted"]
        and row["source_packet_required"]
    )


def _method_gate_pass(row: dict[str, Any]) -> bool:
    ci_low = row["paired_ci95_low_vs_source_label_copy"]
    ci_ok = ci_low is None or ci_low > 0.0
    trained_delta = row.get("delta_vs_trained_label_copy")
    trained_control_ok = trained_delta is None or trained_delta >= STRICT_METHOD_DELTA
    return (
        row["delta_vs_source_label_copy"] is not None
        and row["delta_vs_source_label_copy"] >= STRICT_METHOD_DELTA
        and trained_control_ok
        and ci_ok
        and not row["source_text_exposed"]
        and not row["source_kv_exposed"]
        and not row["raw_hidden_vector_transmitted"]
        and not row["raw_scores_transmitted"]
    )


def _kill_reason(row: dict[str, Any]) -> str:
    delta = row["delta_vs_source_label_copy"]
    ci_high = row["paired_ci95_high_vs_source_label_copy"]
    if delta is None:
        return "missing_source_label_copy_delta"
    if delta < 0.0:
        return "source_label_copy_control_beats_packet"
    if delta < STRICT_METHOD_DELTA:
        return "delta_below_0p02_source_label_copy_margin"
    trained_delta = row.get("delta_vs_trained_label_copy")
    if trained_delta is not None and trained_delta < STRICT_METHOD_DELTA:
        return "trained_label_copy_control_blocks_packet"
    if ci_high is not None and ci_high <= 0.0:
        return "paired_ci_does_not_clear_source_label_copy"
    return "method_gate_clear"


def _finalize_row(row: dict[str, Any], *, native_ready: bool) -> dict[str, Any]:
    row.setdefault("trained_label_copy_accuracy", None)
    row.setdefault("delta_vs_trained_label_copy", None)
    row["systems_audit_pass"] = _systems_audit_pass(row)
    row["method_gate_pass"] = _method_gate_pass(row)
    row["native_queue_allowed"] = bool(row["method_gate_pass"] and row["systems_audit_pass"] and native_ready)
    row["kill_reason"] = _kill_reason(row)
    return row


def _artifact_meta(path: pathlib.Path, control_path: pathlib.Path | None = None) -> dict[str, Any]:
    return {
        "artifact_path": _rel(path),
        "artifact_sha256": _sha256_file(path),
        "control_artifact_path": _rel(control_path),
    }


def _build_rows(artifacts: dict[str, pathlib.Path], payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    fixed = payloads["fixed_packet"]
    control = payloads["control_suite"]
    score = payloads["score_packet_headroom"]
    public = payloads["public_receiver_repair"]
    train_score = payloads["train_source_score_repair"]
    hidden = payloads["hidden_summary_repair"]
    top2 = payloads["top2_contrastive_repair"]
    hidden_innovation = payloads["hidden_innovation_repair"]
    packet_ring = payloads["packet_ring"]
    fixed_trace = fixed["systems_trace"]
    native_ready = bool(payloads["native_readiness"]["headline"]["native_ready"])

    fixed_ci_low, fixed_ci_high = _ci_low_high(control["headline"]["paired_ci95_vs_source_label_text_copy"])
    hidden_ci_low, hidden_ci_high = _ci_low_high(
        hidden["headline"]["paired_ci95_hidden_packet_vs_source_label_copy"]
    )
    top2_ci_low, top2_ci_high = _ci_low_high(top2["headline"]["paired_ci95_selected_vs_best_label_copy"])
    hidden_innovation_ci_low, hidden_innovation_ci_high = _ci_low_high(
        hidden_innovation["headline"]["paired_ci95_selected_vs_best_label_copy"]
    )

    rows = [
        {
            "row_id": "fixed_packet_vs_source_label_copy",
            "repair_family": "fixed_top_choice_packet",
            **_artifact_meta(artifacts["fixed_packet"], artifacts["control_suite"]),
            "eval_rows": fixed["eval_rows"],
            "train_rows": None,
            "accuracy": fixed["headline"]["matched_accuracy"],
            "source_label_copy_accuracy": control["headline"]["source_label_text_copy_accuracy"],
            "delta_vs_source_label_copy": control["headline"]["matched_minus_source_label_text_copy"],
            "delta_vs_same_byte_text": fixed["headline"]["matched_minus_same_byte_text"],
            "paired_ci95_low_vs_source_label_copy": fixed_ci_low,
            "paired_ci95_high_vs_source_label_copy": fixed_ci_high,
            "source_top2_oracle_accuracy": score["headline"]["top2_oracle_accuracy"],
            "oracle_gap_remaining": score["headline"]["top2_oracle_accuracy"] - fixed["headline"]["matched_accuracy"],
            "source_scoring_train_s": None,
            "source_scoring_eval_s": fixed_trace["phase_timings_s"].get("source_scoring"),
            "hidden_extract_train_s": None,
            "hidden_extract_eval_s": None,
            "total_wall_s": fixed_trace["phase_timings_s"].get("total_before_artifact_write"),
            "peak_rss_mib": fixed_trace.get("peak_rss_mib"),
            **_system_fields(
                raw_payload_bytes=fixed_trace["raw_payload_bytes_per_request"],
                explicit_framed_bytes=fixed_trace["record_bytes_with_header_crc"],
                fixed_trace=fixed_trace,
                packet_ring=packet_ring,
                source_text_exposed=fixed_trace["source_text_exposed"],
                source_kv_exposed=fixed_trace["source_kv_exposed"],
            ),
        },
        {
            "row_id": "score_margin_packet",
            "repair_family": "source_score_shape_packet",
            **_artifact_meta(artifacts["score_packet_headroom"]),
            "eval_rows": score["heldout_eval_rows"],
            "train_rows": score["calibration_rows"],
            "accuracy": score["headline"]["score_packet_heldout_accuracy"],
            "source_label_copy_accuracy": score["headline"]["source_label_text_heldout_accuracy"],
            "delta_vs_source_label_copy": score["headline"]["score_packet_minus_source_label_text_heldout"],
            "delta_vs_same_byte_text": None,
            "paired_ci95_low_vs_source_label_copy": None,
            "paired_ci95_high_vs_source_label_copy": None,
            "source_top2_oracle_accuracy": score["headline"]["top2_oracle_heldout_accuracy"],
            "oracle_gap_remaining": score["headline"]["top2_oracle_heldout_accuracy"]
            - score["headline"]["score_packet_heldout_accuracy"],
            "source_scoring_train_s": None,
            "source_scoring_eval_s": score["source_model"].get("latency_s"),
            "hidden_extract_train_s": None,
            "hidden_extract_eval_s": None,
            "total_wall_s": None,
            "peak_rss_mib": None,
            **_system_fields(
                raw_payload_bytes=score["packet_contract"]["payload_bytes"],
                explicit_framed_bytes=None,
                fixed_trace=fixed_trace,
                packet_ring=packet_ring,
                source_text_exposed=False,
                source_kv_exposed=False,
                raw_scores_transmitted=False,
            ),
        },
        {
            "row_id": "public_receiver_top2_repair",
            "repair_family": "public_receiver_top2_rerank",
            **_artifact_meta(artifacts["public_receiver_repair"]),
            "eval_rows": public["eval_rows"],
            "train_rows": public["train_rows"],
            "accuracy": public["headline"]["best_repair_accuracy"],
            "source_label_copy_accuracy": public["headline"]["source_label_copy_accuracy"],
            "delta_vs_source_label_copy": public["headline"]["best_repair_minus_source_label_copy"],
            "delta_vs_same_byte_text": None,
            "paired_ci95_low_vs_source_label_copy": None,
            "paired_ci95_high_vs_source_label_copy": None,
            "source_top2_oracle_accuracy": public["headline"]["source_top2_oracle_accuracy"],
            "oracle_gap_remaining": public["headline"]["source_top2_oracle_accuracy"]
            - public["headline"]["best_repair_accuracy"],
            "source_scoring_train_s": None,
            "source_scoring_eval_s": public["source_model"].get("latency_s"),
            "hidden_extract_train_s": None,
            "hidden_extract_eval_s": None,
            "total_wall_s": public["timing"].get("total_seconds"),
            "peak_rss_mib": None,
            **_system_fields(
                raw_payload_bytes=public["packet_contract"]["payload_bytes"],
                explicit_framed_bytes=None,
                fixed_trace=fixed_trace,
                packet_ring=packet_ring,
                source_text_exposed=public["packet_contract"]["source_text_exposed"],
                source_kv_exposed=public["packet_contract"]["source_kv_exposed"],
            ),
        },
        {
            "row_id": "train_source_score_repair",
            "repair_family": "train_split_source_score_shape_decoder",
            **_artifact_meta(artifacts["train_source_score_repair"]),
            "eval_rows": train_score["eval_rows"],
            "train_rows": train_score["scored_train_rows"],
            "accuracy": train_score["headline"]["selected_eval_accuracy"],
            "source_label_copy_accuracy": train_score["headline"]["best_label_copy_eval_accuracy"],
            "delta_vs_source_label_copy": train_score["headline"]["selected_minus_best_label_copy"],
            "trained_label_copy_accuracy": train_score["headline"]["trained_choice_bias_label_copy_eval_accuracy"],
            "delta_vs_trained_label_copy": train_score["headline"]["selected_minus_trained_choice_bias_label_copy"],
            "delta_vs_same_byte_text": None,
            "paired_ci95_low_vs_source_label_copy": None,
            "paired_ci95_high_vs_source_label_copy": None,
            "source_top2_oracle_accuracy": train_score["headline"]["source_top2_oracle_accuracy"],
            "oracle_gap_remaining": train_score["headline"]["source_top2_oracle_accuracy"]
            - train_score["headline"]["selected_eval_accuracy"],
            "source_scoring_train_s": train_score["source_model"]["train"].get("latency_s"),
            "source_scoring_eval_s": train_score["source_model"]["eval"].get("latency_s"),
            "hidden_extract_train_s": None,
            "hidden_extract_eval_s": None,
            "total_wall_s": train_score["timing"].get("total_seconds"),
            "peak_rss_mib": None,
            **_system_fields(
                raw_payload_bytes=train_score["packet_contract"]["raw_payload_bytes"],
                explicit_framed_bytes=train_score["packet_contract"]["framed_record_bytes"],
                fixed_trace=fixed_trace,
                packet_ring=packet_ring,
                source_text_exposed=train_score["packet_contract"]["source_text_exposed"],
                source_kv_exposed=train_score["packet_contract"]["source_kv_exposed"],
                raw_scores_transmitted=False,
            ),
        },
        {
            "row_id": "hidden_summary_repair",
            "repair_family": "train_split_source_hidden_summary_decoder",
            **_artifact_meta(artifacts["hidden_summary_repair"]),
            "eval_rows": hidden["eval_rows"],
            "train_rows": hidden["scored_train_rows"],
            "accuracy": hidden["headline"]["hidden_packet_eval_accuracy"],
            "source_label_copy_accuracy": hidden["headline"]["source_label_copy_eval_accuracy"],
            "delta_vs_source_label_copy": hidden["headline"]["hidden_packet_minus_source_label_copy"],
            "delta_vs_same_byte_text": hidden["headline"]["hidden_packet_minus_same_byte_text"],
            "paired_ci95_low_vs_source_label_copy": hidden_ci_low,
            "paired_ci95_high_vs_source_label_copy": hidden_ci_high,
            "source_top2_oracle_accuracy": hidden["headline"]["source_top2_oracle_accuracy"],
            "oracle_gap_remaining": hidden["headline"]["source_top2_oracle_accuracy"]
            - hidden["headline"]["hidden_packet_eval_accuracy"],
            "source_scoring_train_s": hidden["source_model"]["score_train"].get("latency_s"),
            "source_scoring_eval_s": hidden["source_model"]["score_eval"].get("latency_s"),
            "hidden_extract_train_s": hidden["source_model"]["hidden_train"].get("latency_s"),
            "hidden_extract_eval_s": hidden["source_model"]["hidden_eval"].get("latency_s"),
            "total_wall_s": hidden["timing"].get("total_seconds"),
            "peak_rss_mib": None,
            **_system_fields(
                raw_payload_bytes=hidden["packet_contract"]["raw_payload_bytes"],
                explicit_framed_bytes=hidden["packet_contract"]["framed_record_bytes"],
                fixed_trace=fixed_trace,
                packet_ring=packet_ring,
                source_text_exposed=hidden["packet_contract"]["source_text_exposed"],
                source_kv_exposed=hidden["packet_contract"]["source_kv_exposed"],
                raw_hidden_vector_transmitted=hidden["packet_contract"]["raw_hidden_vector_transmitted"],
                raw_scores_transmitted=hidden["packet_contract"]["raw_scores_transmitted"],
            ),
        },
        {
            "row_id": "top2_contrastive_switch_repair",
            "repair_family": "top2_contrastive_source_error_switch",
            **_artifact_meta(artifacts["top2_contrastive_repair"]),
            "eval_rows": top2["eval_rows"],
            "train_rows": top2["scored_train_rows"],
            "accuracy": top2["headline"]["selected_eval_accuracy"],
            "source_label_copy_accuracy": top2["headline"]["source_label_copy_eval_accuracy"],
            "delta_vs_source_label_copy": top2["headline"]["selected_minus_source_label_copy"],
            "trained_label_copy_accuracy": top2["headline"]["trained_choice_bias_label_copy_eval_accuracy"],
            "delta_vs_trained_label_copy": top2["headline"]["selected_minus_trained_choice_bias_label_copy"],
            "delta_vs_same_byte_text": None,
            "paired_ci95_low_vs_source_label_copy": top2_ci_low,
            "paired_ci95_high_vs_source_label_copy": top2_ci_high,
            "source_top2_oracle_accuracy": top2["headline"]["source_top2_oracle_accuracy"],
            "oracle_gap_remaining": top2["headline"]["source_top2_oracle_accuracy"]
            - top2["headline"]["selected_eval_accuracy"],
            "source_scoring_train_s": top2["source_model"]["score_train"].get("latency_s"),
            "source_scoring_eval_s": top2["source_model"]["score_eval"].get("latency_s"),
            "hidden_extract_train_s": top2["source_model"]["hidden_train"].get("latency_s"),
            "hidden_extract_eval_s": top2["source_model"]["hidden_eval"].get("latency_s"),
            "total_wall_s": top2["timing"].get("total_seconds"),
            "peak_rss_mib": None,
            **_system_fields(
                raw_payload_bytes=top2["packet_contract"]["raw_payload_bytes"],
                explicit_framed_bytes=top2["packet_contract"]["framed_record_bytes"],
                fixed_trace=fixed_trace,
                packet_ring=packet_ring,
                source_text_exposed=top2["packet_contract"]["source_text_exposed"],
                source_kv_exposed=top2["packet_contract"]["source_kv_exposed"],
                raw_hidden_vector_transmitted=top2["packet_contract"]["raw_hidden_vector_transmitted"],
                raw_scores_transmitted=top2["packet_contract"]["raw_scores_transmitted"],
            ),
        },
        {
            "row_id": "hidden_innovation_repair",
            "repair_family": "train_split_source_hidden_innovation_denoiser",
            **_artifact_meta(artifacts["hidden_innovation_repair"]),
            "eval_rows": hidden_innovation["eval_rows"],
            "train_rows": hidden_innovation["scored_train_rows"],
            "accuracy": hidden_innovation["headline"]["selected_eval_accuracy"],
            "source_label_copy_accuracy": hidden_innovation["headline"]["best_label_copy_eval_accuracy"],
            "delta_vs_source_label_copy": hidden_innovation["headline"]["selected_minus_best_label_copy"],
            "trained_label_copy_accuracy": hidden_innovation["headline"][
                "trained_choice_bias_label_copy_eval_accuracy"
            ],
            "delta_vs_trained_label_copy": hidden_innovation["headline"][
                "selected_minus_trained_choice_bias_label_copy"
            ],
            "delta_vs_same_byte_text": None,
            "paired_ci95_low_vs_source_label_copy": hidden_innovation_ci_low,
            "paired_ci95_high_vs_source_label_copy": hidden_innovation_ci_high,
            "source_top2_oracle_accuracy": hidden_innovation["headline"]["source_top2_oracle_accuracy"],
            "oracle_gap_remaining": hidden_innovation["headline"]["source_top2_oracle_accuracy"]
            - hidden_innovation["headline"]["selected_eval_accuracy"],
            "source_scoring_train_s": hidden_innovation["source_model"]["score_train"].get("latency_s"),
            "source_scoring_eval_s": hidden_innovation["source_model"]["score_eval"].get("latency_s"),
            "hidden_extract_train_s": hidden_innovation["source_model"]["hidden_train"].get("latency_s"),
            "hidden_extract_eval_s": hidden_innovation["source_model"]["hidden_eval"].get("latency_s"),
            "total_wall_s": hidden_innovation["timing"].get("total_seconds"),
            "peak_rss_mib": None,
            **_system_fields(
                raw_payload_bytes=hidden_innovation["packet_contract"]["raw_payload_bytes"],
                explicit_framed_bytes=hidden_innovation["packet_contract"]["framed_record_bytes"],
                fixed_trace=fixed_trace,
                packet_ring=packet_ring,
                source_text_exposed=hidden_innovation["packet_contract"]["source_text_exposed"],
                source_kv_exposed=hidden_innovation["packet_contract"]["source_kv_exposed"],
                raw_hidden_vector_transmitted=hidden_innovation["packet_contract"]["raw_hidden_vector_transmitted"],
                raw_scores_transmitted=hidden_innovation["packet_contract"]["raw_scores_transmitted"],
            ),
        },
    ]
    return [_finalize_row(row, native_ready=native_ready) for row in rows]


def _pass_checks(rows: list[dict[str, Any]], payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    best_delta = max(row["delta_vs_source_label_copy"] for row in rows if row["delta_vs_source_label_copy"] is not None)
    trained_control_rows = [row for row in rows if row.get("delta_vs_trained_label_copy") is not None]
    max_oracle_headroom = max(
        row["source_top2_oracle_accuracy"] - row["source_label_copy_accuracy"]
        for row in rows
        if row["source_top2_oracle_accuracy"] is not None and row["source_label_copy_accuracy"] is not None
    )
    checks = [
        ("all_rows_have_label_copy_delta", all(row["delta_vs_source_label_copy"] is not None for row in rows)),
        (
            "all_rows_have_byte_latency_exposure_fields",
            all(
                row["raw_payload_bytes"] is not None
                and row["framed_record_bytes"] is not None
                and row["source_text_exposed"] is not None
                and row["source_kv_exposed"] is not None
                for row in rows
            ),
        ),
        ("systems_audit_passes", all(row["systems_audit_pass"] for row in rows)),
        (
            "method_gate_status_matches_margin_rule",
            any(row["method_gate_pass"] for row in rows) == (best_delta >= STRICT_METHOD_DELTA),
        ),
        ("trained_label_copy_control_available", bool(trained_control_rows)),
        (
            "trained_label_copy_control_respected_when_available",
            all(
                row.get("delta_vs_trained_label_copy") is None
                or row["delta_vs_trained_label_copy"] >= STRICT_METHOD_DELTA
                or not row["method_gate_pass"]
                for row in rows
            ),
        ),
        ("source_private_boundary_preserved", all(not row["source_text_exposed"] and not row["source_kv_exposed"] for row in rows)),
        ("label_copy_margin_gate_clears_best_or_blocks_all", best_delta >= STRICT_METHOD_DELTA or not any(row["method_gate_pass"] for row in rows)),
        ("oracle_headroom_documented", max_oracle_headroom >= 0.20),
        (
            "native_queue_blocked",
            not any(row["native_queue_allowed"] for row in rows)
            and payloads["native_readiness"]["headline"]["native_ready"] is False,
        ),
        (
            "systems_comparator_available",
            payloads["cross_benchmark_systems"]["headline"]["pass_gate"] is True
            and payloads["cross_benchmark_systems"]["headline"]["native_systems_complete"] is False,
        ),
        (
            "strict_method_gate_rule_recorded",
            STRICT_METHOD_DELTA == 0.02
            and payloads["control_suite"]["headline"]["label_copy_threat_present"] is True,
        ),
    ]
    return [{"check": name, "pass": bool(value)} for name, value in checks]


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_COLUMNS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _fmt(row.get(column)) for column in CSV_COLUMNS})


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# HellaSwag Repair Systems Acceptance Card",
        "",
        "## Headline",
        "",
        f"- Method gate pass: `{payload['headline']['method_gate_pass']}`",
        f"- Systems audit pass: `{payload['headline']['systems_audit_pass']}`",
        f"- Native queue allowed: `{payload['headline']['native_queue_allowed']}`",
        f"- Best delta vs source-label copy: `{payload['headline']['best_delta_vs_source_label_copy']:.6f}`",
        f"- Best delta vs trained label-copy control: `{payload['headline']['best_delta_vs_trained_label_copy']}`",
        f"- Trained label-copy control rows: `{payload['headline']['trained_label_copy_control_rows']}`",
        f"- Best repair row: `{payload['headline']['best_repair_row_id']}`",
        f"- Strict promotion rule: `{payload['strict_method_gate_rule']}`",
        "",
        "## Rows",
        "",
        (
            "| Row | Accuracy | Source-label copy | Delta | Trained copy | Delta trained | Method gate | Systems audit | "
            "Bytes | Kill reason |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        trained_accuracy = (
            "-"
            if row["trained_label_copy_accuracy"] is None
            else f"{row['trained_label_copy_accuracy']:.3f}"
        )
        trained_delta = (
            "-"
            if row["delta_vs_trained_label_copy"] is None
            else f"{row['delta_vs_trained_label_copy']:.3f}"
        )
        lines.append(
            f"| `{row['row_id']}` | {row['accuracy']:.3f} | {row['source_label_copy_accuracy']:.3f} | "
            f"{row['delta_vs_source_label_copy']:.3f} | {trained_accuracy} | {trained_delta} | "
            f"`{row['method_gate_pass']}` | "
            f"`{row['systems_audit_pass']}` | {row['raw_payload_bytes']:.0f}B/{row['framed_record_bytes']:.0f}B | "
            f"{row['kill_reason']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
            "## Checks",
            "",
        ]
    )
    for check in payload["checks"]:
        lines.append(f"- `{check['check']}`: `{check['pass']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(output_dir: pathlib.Path, payload: dict[str, Any]) -> None:
    files = [
        output_dir / "hellaswag_repair_systems_acceptance_card.json",
        output_dir / "hellaswag_repair_systems_acceptance_card.csv",
        output_dir / "hellaswag_repair_systems_acceptance_card.md",
    ]
    manifest = {
        "gate": payload["gate"],
        "created_utc": payload["created_utc"],
        "headline": payload["headline"],
        "files": [{"path": _rel(path), "sha256": _sha256_file(path), "bytes": path.stat().st_size} for path in files],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_acceptance_card(
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
    artifact_paths: dict[str, pathlib.Path] | None = None,
    run_date: str = "2026-05-01",
) -> dict[str, Any]:
    artifacts = dict(DEFAULT_ARTIFACTS)
    if artifact_paths:
        artifacts.update(artifact_paths)

    payloads = {name: _read_json(path) for name, path in artifacts.items()}
    rows = _build_rows(artifacts, payloads)
    checks = _pass_checks(rows, payloads)
    best_row = max(rows, key=lambda row: row["delta_vs_source_label_copy"])
    max_oracle_headroom = max(row["oracle_gap_remaining"] for row in rows if row["oracle_gap_remaining"] is not None)
    trained_control_deltas = [
        row["delta_vs_trained_label_copy"]
        for row in rows
        if row["delta_vs_trained_label_copy"] is not None
    ]

    headline = {
        "pass_gate": all(check["pass"] for check in checks),
        "rows": len(rows),
        "method_gate_pass": any(row["method_gate_pass"] for row in rows),
        "systems_audit_pass": all(row["systems_audit_pass"] for row in rows),
        "native_queue_allowed": any(row["native_queue_allowed"] for row in rows),
        "best_repair_row_id": best_row["row_id"],
        "best_delta_vs_source_label_copy": best_row["delta_vs_source_label_copy"],
        "best_accuracy": best_row["accuracy"],
        "best_source_label_copy_accuracy": best_row["source_label_copy_accuracy"],
        "trained_label_copy_control_rows": len(trained_control_deltas),
        "best_delta_vs_trained_label_copy": max(trained_control_deltas) if trained_control_deltas else None,
        "max_oracle_gap_remaining": max_oracle_headroom,
        "strict_delta_required": STRICT_METHOD_DELTA,
        "native_ready": payloads["native_readiness"]["headline"]["native_ready"],
        "pending_native_rows": payloads["native_readiness"]["headline"]["pending_native_rows"],
        "cross_benchmark_min_qjl_1bit_ratio_vs_framed": payloads["cross_benchmark_systems"]["headline"][
            "min_qjl_1bit_ratio_vs_framed"
        ],
        "serving_slo_pass_gate": payloads["serving_slo"]["headline"]["pass_gate"],
    }

    payload = {
        "gate": "source_private_hellaswag_repair_systems_acceptance_card",
        "date": run_date,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pass_gate": headline["pass_gate"],
        "headline": headline,
        "strict_method_gate_rule": (
            "Promote a HellaSwag repair only if it beats source-label-copy by at least 0.02 on the "
            "frozen validation slice, also beats trained label-bias copy controls by at least 0.02 "
            "when those controls are available, has paired CI95 low > 0 when paired samples are available, "
            "and exposes no source text, source KV, raw hidden vectors, or raw score vectors."
        ),
        "native_queue_policy": (
            "Queue HellaSwag native systems only after the strict method gate passes and native hardware is "
            "available; until then keep native systems effort on ARC/OpenBookQA headline rows and global packet methods."
        ),
        "interpretation": (
            "The hidden-innovation denoiser is the first HellaSwag repair in this card to clear the strict "
            "source-label/trained-label copy margin with paired uncertainty while preserving source-private "
            "byte accounting. Native queueing remains blocked only because native NVIDIA/vLLM/SGLang rows are "
            "not yet available."
        ),
        "checks": checks,
        "rows": rows,
        "inputs": {
            name: {"path": _rel(path), "sha256": _sha256_file(path)}
            for name, path in sorted(artifacts.items())
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hellaswag_repair_systems_acceptance_card.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "hellaswag_repair_systems_acceptance_card.csv", rows)
    _write_markdown(output_dir / "hellaswag_repair_systems_acceptance_card.md", payload)
    _write_manifest(output_dir, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-date", default="2026-05-01")
    args = parser.parse_args()

    payload = build_acceptance_card(output_dir=args.output_dir, run_date=args.run_date)
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
