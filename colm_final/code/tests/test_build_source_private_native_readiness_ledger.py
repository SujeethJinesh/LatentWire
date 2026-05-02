from __future__ import annotations

import json

from scripts.build_source_private_native_readiness_ledger import build_native_readiness_ledger


def _write(path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_native_readiness_ledger_marks_mac_rows_and_native_blockers(tmp_path) -> None:
    frontier_path = tmp_path / "frontier.json"
    packet_path = tmp_path / "packet.json"
    serving_path = tmp_path / "serving.json"
    _write(
        frontier_path,
        {
            "policies": {
                "per_seed": {
                    "min_selected_candidate_accuracy": 0.65,
                    "max_selected_best_control_accuracy": 0.27,
                }
            },
            "selected_eval_rows": [
                {
                    "phase": "eval",
                    "policy": "per_seed",
                    "budget_bytes": 12,
                    "candidate_accuracy": 0.65,
                    "target_accuracy": 0.25,
                },
                {
                    "phase": "eval",
                    "policy": "per_seed",
                    "budget_bytes": 14,
                    "candidate_accuracy": 0.75,
                    "target_accuracy": 0.25,
                },
            ],
        },
    )
    _write(
        packet_path,
        {
            "headline": {
                "packet_batch64_record_bytes": 5,
                "packet_batch64_line_bytes_per_request": 5,
                "packet_batch64_p50_ns_per_request": 0.67,
            }
        },
    )
    _write(
        serving_path,
        {
            "headline": {
                "native_serving_gap": "GPU TPOT/goodput remains unmeasured",
                "packet_min_raw_bytes": 2,
                "packet_min_batch64_line_bytes": 5,
            },
            "rows": [
                {
                    "row_class": "endpoint_packet",
                    "accuracy": 0.67,
                    "delta_vs_target": 0.42,
                    "ttft_p50_ms": 470.0,
                }
            ],
        },
    )

    payload = build_native_readiness_ledger(
        frontier_path=frontier_path,
        packet_ring_path=packet_path,
        serving_slo_path=serving_path,
        output_dir=tmp_path / "out",
    )

    assert payload["pass_gate"] is False
    assert payload["headline"]["local_measured_rows"] == 3
    assert payload["headline"]["pending_native_rows"] == 5
    rows = {row["row_id"]: row for row in payload["rows"]}
    assert rows["latentwire_train_donor_frontier"]["source_private"] is True
    assert rows["latentwire_train_donor_frontier"]["record_bytes"] == 14.0
    assert rows["latentwire_mac_packet_ring"]["mac_packet_ring_p50_ns"] == 0.67
    assert rows["external_c2c"]["source_kv_exposed"] is True
    assert rows["external_vllm"]["native_kernel_status"] == "pending_native_required"
    assert (tmp_path / "out" / "native_readiness_ledger.json").exists()
    assert (tmp_path / "out" / "manifest.json").exists()
