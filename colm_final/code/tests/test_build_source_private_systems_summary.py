from __future__ import annotations

from scripts.build_source_private_systems_summary import aggregate_systems_summary, deterministic_system_rows


def _metrics() -> dict:
    names = [
        "target_only",
        "matched_repair_packet",
        "structured_text_matched",
        "structured_json_matched",
        "structured_free_text_matched",
        "full_hidden_log",
        "full_diag_text",
    ]
    out = {}
    for name in names:
        out[name] = {
            "accuracy": 0.25,
            "mean_payload_bytes": 2.0,
            "mean_payload_tokens": 1.0,
            "p50_latency_ms": 0.0,
        }
    out["matched_repair_packet"]["accuracy"] = 1.0
    out["full_hidden_log"]["accuracy"] = 1.0
    out["full_hidden_log"]["mean_payload_bytes"] = 200.0
    out["full_diag_text"]["accuracy"] = 1.0
    out["full_diag_text"]["mean_payload_bytes"] = 14.0
    out["target_only"]["mean_payload_bytes"] = 0.0
    return out


def test_deterministic_system_rows_compute_compression() -> None:
    rows = deterministic_system_rows("surface", {"metrics": _metrics()})
    packet = next(row for row in rows if row["interface"] == "2-byte diagnostic packet")

    assert packet["accuracy"] == 1.0
    assert packet["compression_vs_full_log"] == 100.0
    assert packet["compression_vs_full_diag"] == 7.0


def test_aggregate_systems_summary_has_headline() -> None:
    medium = {
        "rows": [
            {
                "prompt_mode": "trace_no_hint",
                "model": "m",
                "run_id": "r",
                "matched_accuracy": 1.0,
                "target_only_accuracy": 0.25,
                "best_source_destroying_control_accuracy": 0.25,
                "packet_valid_rate": 1.0,
                "mean_packet_bytes": 2.0,
                "mean_packet_tokens": 1.0,
                "p50_source_latency_ms": 10.0,
                "p95_source_latency_ms": 20.0,
            }
        ]
    }
    payload = aggregate_systems_summary(
        deterministic=[("s", {"metrics": _metrics()})],
        medium_summary=medium,
        target_paths=[],
    )

    assert payload["headline"]["packet_accuracy_min"] == 1.0
    assert payload["headline"]["matched_byte_text_accuracy_max"] == 0.25
    assert payload["model_packet_rows"][0]["packets_per_second_p50"] == 100.0
