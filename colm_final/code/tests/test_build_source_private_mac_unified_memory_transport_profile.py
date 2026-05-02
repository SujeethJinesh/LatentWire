from __future__ import annotations

import json

from scripts.build_source_private_mac_unified_memory_transport_profile import (
    DEFAULT_ENDPOINT_ROWS,
    DEFAULT_ENDPOINT_SUMMARIES,
    DEFAULT_KV_BASELINE_TABLE,
    DEFAULT_PACKET_ISA_FRONTIER,
    _ceil_bytes,
    build_mac_unified_memory_transport_profile,
)


def test_ceil_bytes_respects_zero_and_transfer_quantum() -> None:
    assert _ceil_bytes(0, 64) == 0.0
    assert _ceil_bytes(2, 64) == 64.0
    assert _ceil_bytes(64, 64) == 64.0
    assert _ceil_bytes(65, 64) == 128.0


def test_mac_unified_memory_transport_profile_joins_endpoint_and_kv_rows(tmp_path) -> None:
    payload = build_mac_unified_memory_transport_profile(
        endpoint_rows=list(DEFAULT_ENDPOINT_ROWS),
        endpoint_summaries=list(DEFAULT_ENDPOINT_SUMMARIES),
        kv_baseline_table=DEFAULT_KV_BASELINE_TABLE,
        packet_isa_frontier=DEFAULT_PACKET_ISA_FRONTIER,
        output_dir=tmp_path / "transport_profile",
        amortized_batch_size=64,
    )

    assert payload["gate"] == "source_private_mac_unified_memory_transport_profile"
    assert payload["pass_gate"] is True
    assert payload["headline"]["exact_id_parity"] is True
    assert payload["headline"]["surfaces"] == 2
    assert payload["headline"]["rows"] == 18
    assert payload["headline"]["packet_payload_bytes"] == [2.0]
    assert payload["headline"]["matched_packet_min_delta_vs_target"] >= 0.40
    assert payload["headline"]["max_source_destroying_control_delta_vs_target"] <= 0.02
    assert payload["headline"]["query_aware_text_raw_ratio_min"] == 7.0
    assert payload["headline"]["query_aware_text_line_ratio_min"] == 1.0
    assert payload["headline"]["full_log_raw_ratio_min"] > 180.0
    assert payload["headline"]["full_log_line_ratio_min"] >= 6.0
    assert payload["headline"]["packet_batch64_line_bytes_per_request_min"] == 5.0
    assert payload["headline"]["packet_batch64_dma_bytes_per_request_min"] == 6.0
    assert "host_profile" in payload
    assert "execution_note" in payload["host_profile"]

    packet_rows = [row for row in payload["rows"] if row["condition"] == "matched_packet"]
    query_rows = [row for row in payload["rows"] if row["condition"] == "query_aware_diag_span"]
    full_log_rows = [row for row in payload["rows"] if row["condition"] == "full_hidden_log"]

    assert all(row["source_private"] is True for row in packet_rows)
    assert all(row["transport_record_bytes"] == 5.0 for row in packet_rows)
    assert all(row["source_text_exposed"] is False for row in packet_rows)
    assert all(row["single_request_line_bytes_64b"] == 64.0 for row in packet_rows)
    assert all(row["batch64_packet_record_line_bytes_per_request"] == 5.0 for row in packet_rows)
    assert all(row["source_text_exposed"] is True for row in query_rows)
    assert all(row["line_ratio_vs_packet"] == 1.0 for row in query_rows)
    assert all("exposes private evidence text" in row["transport_conclusion"] for row in query_rows)
    assert all(row["qjl_1bit_kv_delta_bytes_vs_packet"] > 600_000 for row in full_log_rows)
    assert all(row["source_kv_exposed"] is False for row in payload["rows"])

    summary = json.loads(
        (tmp_path / "transport_profile" / "mac_unified_memory_transport_profile.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((tmp_path / "transport_profile" / "manifest.json").read_text(encoding="utf-8"))
    assert summary["headline"] == payload["headline"]
    assert "mac_unified_memory_transport_profile.csv" in manifest["artifacts"]
    assert (tmp_path / "transport_profile" / "mac_unified_memory_transport_profile.md").exists()
