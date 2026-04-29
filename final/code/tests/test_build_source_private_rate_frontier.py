from __future__ import annotations

import json

from scripts.build_source_private_rate_frontier import build_rate_frontier


def _metric(accuracy: float, bytes_: float) -> dict:
    return {
        "accuracy": accuracy,
        "mean_payload_bytes": bytes_,
        "mean_payload_tokens": 1.0 if bytes_ else 0.0,
        "p50_latency_ms": 0.1,
        "p95_latency_ms": 0.2,
    }


def _sweep() -> dict:
    rows = []
    for budget in [2, 32]:
        rows.append(
            {
                "budget_bytes": budget,
                "metrics": {
                    "target_only": _metric(0.25, 0.0),
                    "matched_repair_packet": _metric(1.0, float(budget)),
                    "structured_text_matched": _metric(0.25, float(budget)),
                    "structured_json_matched": _metric(1.0 if budget == 32 else 0.25, float(budget)),
                    "structured_free_text_matched": _metric(1.0 if budget == 32 else 0.25, float(budget)),
                    "full_hidden_log": _metric(1.0, 400.0),
                    "full_diag_text": _metric(1.0, 14.0),
                },
            }
        )
    return {"budget_summaries": rows}


def test_rate_frontier_reports_packet_byte_advantage(tmp_path) -> None:
    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text(json.dumps(_sweep()), encoding="utf-8")

    payload = build_rate_frontier(
        sweep_paths=[("surface", sweep_path)],
        output_dir=tmp_path / "frontier",
    )

    assert payload["pass_gate"] is True
    assert payload["headline"]["packet_oracle_bytes_max"] == 2.0
    assert payload["headline"]["json_oracle_bytes_min"] == 32.0
    assert payload["headline"]["packet_vs_json_oracle_compression_min"] == 16.0
    assert (tmp_path / "frontier" / "rate_frontier.md").exists()
