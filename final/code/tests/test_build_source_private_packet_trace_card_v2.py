from __future__ import annotations

import json

from scripts import build_source_private_packet_trace_card_v2 as trace


def _write_json(path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_trace_card_v2_builds_claim_checklist(tmp_path) -> None:
    systems = {
        "headline": {
            "contract_failure_packet_accuracy": 0.25,
            "endpoint_packet_rows": 1,
            "endpoint_packet_rows_passing": 1,
            "same_byte_text_accuracy_max": 0.25,
        }
    }
    memory = {
        "headline": {
            "full_log_cacheline_ratio_min": 6.0,
            "full_log_raw_ratio_min": 183.0,
            "full_log_ttft_delta_ms_min": 164.0,
            "kv_cacheline_ratio_min": 336.0,
            "kv_raw_ratio_min": 10752.0,
            "packet_raw_bytes_min": 2.0,
            "packet_single_request_cacheline_bytes_min": 64.0,
            "query_aware_text_cacheline_ratio_min": 1.0,
            "query_aware_text_raw_ratio_min": 7.0,
        },
        "rows": [
            {
                "accuracy": 1.0,
                "batch64_packet_dma_bytes_per_request": 6.0,
                "batch64_packet_line_bytes_per_request": 5.0,
                "claim_class": "mac_endpoint_proxy",
                "communicated_object": "packet",
                "delta_vs_target": 0.75,
                "method": "packet",
                "raw_payload_bytes": 2.0,
                "row_class": "endpoint_packet",
                "single_request_cacheline_bytes": 64.0,
                "single_request_dma_bytes": 128.0,
                "source_destroying_controls": "passed",
                "source_kv_exposed": False,
                "source_private": True,
                "source_text_exposed": False,
                "surface": "toy",
                "target_accuracy": 0.25,
                "traffic_conclusion": "tiny packet",
            }
        ],
    }
    packet_isa = {
        "non_claims": ["This is not measured accelerator throughput."],
        "rows": [
            {
                "batch_size": 64,
                "cache_lines_64b_packed": 5,
                "dma_bursts_128b_packed": 3,
                "dma_bytes_per_request_packed": 6.0,
                "header_bytes": 2,
                "line_bytes_per_request_packed": 5.0,
                "packet_bytes": 5,
                "parity_bytes": 1,
                "payload_bytes": 2,
                "requests_per_128b_burst": 25,
                "requests_per_64b_line": 12,
            }
        ],
    }
    qwen = {
        "headline": {
            "max_p50_latency_ms": 900.0,
            "min_matched_minus_best_control": 0.75,
            "min_matched_minus_target": 0.75,
            "min_paired_ci95_low_vs_best_control": 0.64,
            "min_valid_prediction_rate": 1.0,
            "pass_rows": 1,
            "rows": 1,
        },
        "rows": [{"matched_mean_payload_bytes": 2.0}],
    }
    systems_path = tmp_path / "systems.json"
    memory_path = tmp_path / "memory.json"
    packet_path = tmp_path / "packet.json"
    qwen_path = tmp_path / "qwen.json"
    _write_json(systems_path, systems)
    _write_json(memory_path, memory)
    _write_json(packet_path, packet_isa)
    _write_json(qwen_path, qwen)

    payload = trace.build_packet_trace_card_v2(
        systems_frontier=systems_path,
        memory_ledger=memory_path,
        packet_isa=packet_path,
        qwen_receiver_uncertainty=qwen_path,
        output_dir=tmp_path / "out",
    )

    assert payload["pass_gate"] is True
    assert payload["headline"]["checklist_passed"] == payload["headline"]["checklist_total"]
    assert payload["packet_record_layout"]["packet_bytes_with_overhead"] == 5
    assert (tmp_path / "out" / "packet_trace_card_v2.md").exists()
