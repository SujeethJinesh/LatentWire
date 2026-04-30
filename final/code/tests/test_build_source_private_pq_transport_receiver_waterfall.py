from __future__ import annotations

import json

from scripts.build_source_private_pq_transport_receiver_waterfall import build_pq_transport_receiver_waterfall


def test_pq_transport_receiver_waterfall_joins_transport_and_receiver(tmp_path) -> None:
    transport = {
        "pass_gate": True,
        "rows": [
            {"profile": "packet_2b_payload_5b_record", "batch_size": 64, "record_bytes": 5, "source_text_exposed": False, "source_kv_exposed": False, "line_bytes_per_request": 5.0, "dma_bytes_per_request": 6.0, "p50_ns_per_request": 1.0, "p95_ns_per_request": 1.2},
            {"profile": "pq_packet_4b_payload_7b_record", "batch_size": 64, "record_bytes": 7, "source_text_exposed": False, "source_kv_exposed": False, "line_bytes_per_request": 7.0, "dma_bytes_per_request": 8.0, "p50_ns_per_request": 1.1, "p95_ns_per_request": 1.3},
            {"profile": "query_aware_text_14b", "batch_size": 64, "record_bytes": 14, "source_text_exposed": True, "source_kv_exposed": False, "line_bytes_per_request": 14.0, "dma_bytes_per_request": 14.0, "p50_ns_per_request": 1.2, "p95_ns_per_request": 1.4},
            {"profile": "full_hidden_log_370b", "batch_size": 64, "record_bytes": 370, "source_text_exposed": True, "source_kv_exposed": False, "line_bytes_per_request": 370.0, "dma_bytes_per_request": 370.0, "p50_ns_per_request": 7.0, "p95_ns_per_request": 8.0},
            {"profile": "qjl_1bit_kv_floor_21504b", "batch_size": 64, "record_bytes": 21504, "source_text_exposed": False, "source_kv_exposed": True, "line_bytes_per_request": 21504.0, "dma_bytes_per_request": 21504.0, "p50_ns_per_request": 200.0, "p95_ns_per_request": 210.0},
            {"profile": "kivi_2bit_kv_floor_43008b", "batch_size": 64, "record_bytes": 43008, "source_text_exposed": False, "source_kv_exposed": True, "line_bytes_per_request": 43008.0, "dma_bytes_per_request": 43008.0, "p50_ns_per_request": 400.0, "p95_ns_per_request": 410.0},
        ],
    }
    receiver = {
        "pass_gate": True,
        "headline": {
            "packet_record_bytes_per_request": 7,
            "max_table_prediction_mismatch_count": 0,
            "max_batch_prediction_mismatch_count": 0,
            "batch256_amortized_128b_packet_record_bytes_per_request": 7.0,
        },
        "rows": [
            {
                "resident_table_decode_p50_ms": 0.01,
                "resident_table_decode_p95_ms": 0.02,
                "batch_results": {"64": {"per_request_p50_ms": 0.03, "per_request_p95_ms": 0.04}},
            }
        ],
    }
    transport_path = tmp_path / "transport.json"
    receiver_path = tmp_path / "receiver.json"
    transport_path.write_text(json.dumps(transport), encoding="utf-8")
    receiver_path.write_text(json.dumps(receiver), encoding="utf-8")

    payload = build_pq_transport_receiver_waterfall(
        transport_path=transport_path,
        receiver_path=receiver_path,
        output_dir=tmp_path / "waterfall",
        batch_size=64,
    )

    assert payload["gate"] == "source_private_pq_transport_receiver_waterfall"
    assert payload["pass_gate"] is True
    assert payload["headline"]["pq_record_bytes"] == 7
    assert payload["headline"]["query_text_record_ratio_vs_pq"] == 2.0
    assert payload["headline"]["max_receiver_mismatch_count"] == 0
    assert payload["headline"]["transport_share_of_receiver_batch64_p50"] < 0.001
    assert any(row["profile"] == "pq_resident_table_decode_max" for row in payload["rows"])
    assert (tmp_path / "waterfall" / "pq_transport_receiver_waterfall.json").exists()
