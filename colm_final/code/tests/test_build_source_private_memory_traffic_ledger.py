from __future__ import annotations

import json

from scripts.build_source_private_memory_traffic_ledger import (
    DEFAULT_HARDWARE_FRONTIER,
    DEFAULT_PACKET_ISA_FRONTIER,
    DEFAULT_SYSTEMS_FRONTIER,
    build_memory_traffic_ledger,
)


def test_memory_traffic_ledger_joins_packet_traffic_and_ttft(tmp_path) -> None:
    payload = build_memory_traffic_ledger(
        systems_frontier=DEFAULT_SYSTEMS_FRONTIER,
        hardware_frontier=DEFAULT_HARDWARE_FRONTIER,
        packet_isa_frontier=DEFAULT_PACKET_ISA_FRONTIER,
        output_dir=tmp_path / "ledger",
        amortized_batch_size=64,
    )

    assert payload["gate"] == "source_private_memory_traffic_ledger"
    assert payload["pass_gate"] is True
    assert payload["headline"]["packet_raw_bytes_min"] == 2.0
    assert payload["headline"]["packet_single_request_cacheline_bytes_min"] == 64.0
    assert payload["headline"]["packet_single_request_dma_bytes_min"] == 128.0
    assert payload["headline"]["packet_batch_line_bytes_per_request_min"] == 5.0
    assert payload["headline"]["query_aware_text_raw_ratio_min"] >= 7.0
    assert payload["headline"]["query_aware_text_cacheline_ratio_min"] == 1.0
    assert payload["headline"]["full_log_cacheline_ratio_min"] >= 6.0
    assert payload["headline"]["full_log_ttft_delta_ms_min"] >= 100.0
    assert payload["headline"]["kv_cacheline_ratio_min"] >= 300.0

    rows = payload["rows"]
    packet = next(row for row in rows if row["row_class"] == "endpoint_packet")
    query_text = next(row for row in rows if row["method"] == "query-aware diagnostic text")
    full_log = next(row for row in rows if row["method"] == "full hidden-log relay")
    kv = next(row for row in rows if row["row_class"] == "kv_byte_floor")

    assert packet["claim_class"] == "mac_endpoint_proxy"
    assert packet["source_private"] is True
    assert packet["source_text_exposed"] is False
    assert packet["source_kv_exposed"] is False
    assert packet["batch64_packet_line_bytes_per_request"] == 5.0
    assert "batched" in packet["traffic_conclusion"]
    assert query_text["source_text_exposed"] is True
    assert query_text["cacheline_ratio_vs_packet"] == 1.0
    assert "exposes private text" in query_text["traffic_conclusion"]
    assert full_log["p50_ttft_delta_vs_packet_ms"] >= 100.0
    assert kv["source_kv_exposed"] is True
    assert kv["claim_class"] == "kv_cache_lower_bound"

    summary = json.loads((tmp_path / "ledger" / "memory_traffic_ledger.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "ledger" / "manifest.json").read_text(encoding="utf-8"))
    assert summary["headline"] == payload["headline"]
    assert "memory_traffic_ledger.csv" in manifest["artifacts"]
    assert (tmp_path / "ledger" / "memory_traffic_ledger.md").exists()
