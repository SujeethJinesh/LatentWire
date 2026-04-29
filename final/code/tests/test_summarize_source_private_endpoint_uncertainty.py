from __future__ import annotations

import json

from scripts.summarize_source_private_endpoint_uncertainty import (
    _exact_sign_p_value,
    _paired_counts,
    _paired_values,
    run_summary,
)


CONDITIONS = [
    "target_only",
    "matched_packet",
    "matched_byte_text_2",
    "random_same_byte_packet",
    "deranged_candidate_diag_table",
    "query_aware_diag_span",
    "structured_json_diag",
    "structured_free_text_diag",
    "full_hidden_log",
]


def _row(example_id: str, condition: str, *, correct: bool, payload_bytes: int = 2) -> dict:
    return {
        "example_id": example_id,
        "condition": condition,
        "correct": correct,
        "strict_correct": correct,
        "valid_prediction": True,
        "strict_valid_prediction": True,
        "payload_bytes": payload_bytes,
        "payload_tokens_proxy": 1 if payload_bytes else 0,
        "prompt_bytes": 100,
        "prompt_tokens": 25,
        "generated_tokens": 4,
        "ttft_ms": 10.0 + payload_bytes,
        "e2e_ms": 20.0 + payload_bytes,
    }


def _metric(rows: list[dict], condition: str) -> dict:
    selected = [row for row in rows if row["condition"] == condition]
    return {
        "accuracy": sum(row["correct"] for row in selected) / len(selected),
        "strict_accuracy": sum(row["strict_correct"] for row in selected) / len(selected),
        "valid_prediction_rate": 1.0,
        "mean_payload_bytes": sum(row["payload_bytes"] for row in selected) / len(selected),
        "mean_prompt_tokens": 25.0,
        "p50_ttft_ms": selected[0]["ttft_ms"],
        "p50_e2e_ms": selected[0]["e2e_ms"],
    }


def _write_smoke_result(path) -> None:
    path.mkdir()
    rows = []
    for idx in range(4):
        example_id = f"ex{idx}"
        rows.extend(
            [
                _row(example_id, "target_only", correct=idx == 0, payload_bytes=0),
                _row(example_id, "matched_packet", correct=True),
                _row(example_id, "matched_byte_text_2", correct=idx == 0),
                _row(example_id, "random_same_byte_packet", correct=False),
                _row(example_id, "deranged_candidate_diag_table", correct=idx == 0),
                _row(example_id, "query_aware_diag_span", correct=True, payload_bytes=14),
                _row(example_id, "structured_json_diag", correct=True, payload_bytes=21),
                _row(example_id, "structured_free_text_diag", correct=True, payload_bytes=17),
                _row(example_id, "full_hidden_log", correct=True, payload_bytes=360),
            ]
        )
    (path / "endpoint_proxy_rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    metrics = {condition: _metric(rows, condition) for condition in CONDITIONS}
    summary = {
        "pass_gate": True,
        "n": 4,
        "conditions": CONDITIONS,
        "prompt_style": "label_strict",
        "metrics": metrics,
        "packet_vs_query_payload_compression": 7.0,
        "packet_vs_full_log_payload_compression": 180.0,
        "full_log_ttft_delta_vs_packet_ms": 100.0,
        "full_log_e2e_delta_vs_packet_ms": 120.0,
    }
    (path / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def test_paired_values_and_sign_counts_are_directional() -> None:
    paired = {
        "a": {"matched": {"correct": True}, "target": {"correct": False}},
        "b": {"matched": {"correct": False}, "target": {"correct": True}},
        "c": {"matched": {"correct": True}, "target": {"correct": True}},
    }

    values = _paired_values(paired, method="matched", baseline="target")

    assert values == [1.0, -1.0, 0.0]
    assert _paired_counts(values)["wins"] == 1
    assert _paired_counts(values)["losses"] == 1
    assert _exact_sign_p_value(3, 0) == 0.25


def test_endpoint_uncertainty_summary_passes_on_smoke(tmp_path) -> None:
    result_dir = tmp_path / "endpoint"
    _write_smoke_result(result_dir)

    payload = run_summary(
        result_dirs=[result_dir],
        output_dir=tmp_path / "summary",
        bootstrap_samples=100,
        seed=7,
    )

    assert payload["pass_gate"] is True
    assert payload["rows"][0]["packet_accuracy"] == 1.0
    assert payload["rows"][0]["comparisons"]["target_only"]["delta_bootstrap95"]["ci95_low"] >= 0.25
    assert (tmp_path / "summary" / "summary.md").exists()
