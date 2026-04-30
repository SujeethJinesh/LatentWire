from __future__ import annotations

import json

from scripts.build_source_private_verifier_consumption_trace import build_verifier_consumption_trace


def _row(
    *,
    example_id: str,
    condition: str,
    correct: bool,
    payload_bytes: int,
    latency_ms: float,
    binary_passes: int,
) -> dict[str, object]:
    return {
        "example_id": example_id,
        "condition": condition,
        "correct": correct,
        "valid_prediction": True,
        "payload_bytes": payload_bytes,
        "payload_tokens": 1 if payload_bytes else 0,
        "generated_tokens": binary_passes,
        "latency_ms": latency_ms,
        "candidate_binary_logprobs": [
            {"candidate_label": f"c{i}", "yes_minus_no": 1.0}
            for i in range(binary_passes)
        ],
    }


def test_verifier_consumption_trace_reports_forward_passes_and_boundary_bytes(tmp_path) -> None:
    result_dir = tmp_path / "run"
    result_dir.mkdir()
    rows = []
    for index in range(4):
        example_id = f"e{index}"
        rows.append(
            _row(
                example_id=example_id,
                condition="target_only",
                correct=index == 0,
                payload_bytes=0,
                latency_ms=0.1,
                binary_passes=0,
            )
        )
        rows.append(
            _row(
                example_id=example_id,
                condition="matched_packet",
                correct=True,
                payload_bytes=2,
                latency_ms=10.0 + index,
                binary_passes=4,
            )
        )
        rows.append(
            _row(
                example_id=example_id,
                condition="shuffled_packet",
                correct=index == 0,
                payload_bytes=2,
                latency_ms=9.0,
                binary_passes=4,
            )
        )
        rows.append(
            _row(
                example_id=example_id,
                condition="structured_json_2byte",
                correct=index == 0,
                payload_bytes=2,
                latency_ms=9.0,
                binary_passes=4,
            )
        )
    (result_dir / "target_predictions.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    payload = build_verifier_consumption_trace(
        result_dirs=[result_dir],
        output_dir=tmp_path / "trace",
    )

    assert payload["pass_gate"] is True
    assert payload["headline"]["min_matched_minus_best_control"] == 0.75
    matched = next(row for row in payload["rows"] if row["condition"] == "matched_packet")
    structured = next(row for row in payload["rows"] if row["condition"] == "structured_json_2byte")

    assert matched["mean_binary_forward_passes"] == 4.0
    assert matched["mean_payload_bytes"] == 2.0
    assert matched["mean_packet_record_bytes"] == 5.0
    assert matched["single_request_cacheline_bytes"] == 64.0
    assert matched["single_request_dma_bytes"] == 128.0
    assert matched["batch64_line_bytes_per_request"] == 5.0
    assert matched["batch64_dma_bytes_per_request"] == 6.0
    assert matched["source_private"] is True
    assert structured["source_text_exposed"] is True

    manifest = json.loads((tmp_path / "trace" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["gate"] == "source_private_verifier_consumption_trace"
    assert (tmp_path / "trace" / "verifier_consumption_trace.csv").exists()
