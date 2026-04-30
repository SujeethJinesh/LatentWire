from __future__ import annotations

from scripts.build_source_private_pq_receiver_batch_microbench import (
    _amortized_burst_bytes,
    build_pq_receiver_batch_microbench,
)


def test_amortized_burst_bytes_rounds_batch_payloads() -> None:
    assert _amortized_burst_bytes(payload_bytes=4, batch_size=1, burst_bytes=128) == 128.0
    assert _amortized_burst_bytes(payload_bytes=4, batch_size=8, burst_bytes=128) == 16.0
    assert _amortized_burst_bytes(payload_bytes=4, batch_size=64, burst_bytes=128) == 4.0
    assert _amortized_burst_bytes(payload_bytes=7, batch_size=1, burst_bytes=128) == 128.0
    assert _amortized_burst_bytes(payload_bytes=7, batch_size=8, burst_bytes=128) == 16.0
    assert _amortized_burst_bytes(payload_bytes=7, batch_size=64, burst_bytes=128) == 8.0
    assert _amortized_burst_bytes(payload_bytes=7, batch_size=256, burst_bytes=128) == 7.0


def test_pq_receiver_batch_microbench_matches_scalar_decoder(tmp_path) -> None:
    payload = build_pq_receiver_batch_microbench(
        output_dir=tmp_path / "pq_batch",
        train_examples=64,
        eval_examples=32,
        train_seed=5,
        eval_seed=6,
        remap_seeds=[101],
        budget_bytes=2,
        variants=["canonical", "protected_hadamard"],
        feature_dim=64,
        candidate_view="slot",
        ridge=1e-2,
        opq_iterations=1,
        table_repeats=1,
        batch_repeats=2,
        batch_sizes=[1, 8],
    )

    assert payload["gate"] == "source_private_pq_receiver_batch_microbench"
    assert payload["headline"]["rows"] == 2
    assert payload["headline"]["max_table_prediction_mismatch_count"] == 0
    assert payload["headline"]["max_batch_prediction_mismatch_count"] == 0
    assert payload["headline"]["packet_record_bytes_per_request"] == 5
    assert payload["rows"][0]["resident_table_decode_p50_ms"] >= 0.0
    assert payload["rows"][0]["batch_results"]["1"]["per_request_p50_ms"] >= 0.0
    assert payload["rows"][0]["batch_size_invariant"] is True
    assert payload["rows"][0]["batch_results"]["8"]["amortized_packet_record_128b_burst_bytes_per_request"] == 16.0
    assert (tmp_path / "pq_batch" / "pq_receiver_batch_microbench.json").exists()
    assert (tmp_path / "pq_batch" / "pq_receiver_batch_microbench.csv").exists()
