from __future__ import annotations

from scripts import build_source_private_mac_packet_ring_transport_microbench as bench


def test_augment_rows_computes_same_batch_ratios() -> None:
    rows = [
        {
            "profile": "packet_2b_payload_5b_record",
            "record_bytes": 5,
            "batch_size": 64,
            "p50_ns_per_request": 10.0,
            "p95_ns_per_request": 12.0,
        },
        {
            "profile": "full_hidden_log_370b",
            "record_bytes": 370,
            "batch_size": 64,
            "p50_ns_per_request": 40.0,
            "p95_ns_per_request": 60.0,
        },
    ]

    augmented = bench._augment_rows(rows)
    full = next(row for row in augmented if row["profile"] == "full_hidden_log_370b")

    assert full["line_bytes_per_request"] == 370.0
    assert full["dma_bytes_per_request"] == 370.0
    assert full["ratio_p50_vs_packet_same_batch"] == 4.0
    assert full["ratio_p95_vs_packet_same_batch"] == 5.0
