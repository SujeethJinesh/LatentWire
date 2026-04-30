from __future__ import annotations

import json

from scripts.build_source_private_serving_slo_envelope import (
    DEFAULT_MEMORY_TRAFFIC_LEDGER,
    DEFAULT_PACKET_ISA_FRONTIER,
    build_serving_slo_envelope,
)


def test_serving_slo_envelope_marks_proxy_and_non_claims(tmp_path) -> None:
    payload = build_serving_slo_envelope(
        memory_traffic_ledger=DEFAULT_MEMORY_TRAFFIC_LEDGER,
        packet_isa_frontier=DEFAULT_PACKET_ISA_FRONTIER,
        output_dir=tmp_path / "slo",
    )

    assert payload["gate"] == "source_private_serving_slo_envelope"
    assert payload["pass_gate"] is True
    assert payload["headline"]["rows"] >= 10
    assert payload["headline"]["ttft_measured_rows"] >= 4
    assert payload["headline"]["goodput_claim_allowed_rows"] == 0
    assert payload["headline"]["gpu_counter_required_rows"] == payload["headline"]["rows"]
    assert payload["headline"]["packet_min_raw_bytes"] == 2.0
    assert payload["headline"]["packet_min_batch64_line_bytes"] == 5.0
    assert payload["headline"]["packet_min_batch64_dma_bytes"] == 6.0

    packet = next(row for row in payload["rows"] if row["row_class"] == "endpoint_packet")
    full_log = next(row for row in payload["rows"] if row["method"] == "full hidden-log relay")
    query_text = next(row for row in payload["rows"] if row["method"] == "query-aware diagnostic text")
    kv_floor = next(row for row in payload["rows"] if row["row_class"] == "kv_byte_floor")

    assert packet["ttft_measurement_available"] is True
    assert packet["tpot_has_measurement"] is False
    assert packet["goodput_claim_allowed"] is False
    assert packet["gpu_counter_required"] is True
    assert packet["line_bytes_b64"] == 5.0
    assert packet["ttft_slo_750_margin_ms"] > 0.0
    assert "production" in packet["paper_claim"]
    assert full_log["source_text_exposed"] is True
    assert full_log["ttft_delta_vs_packet_ms"] >= 100.0
    assert query_text["ttft_measurement_available"] is False
    assert query_text["source_text_exposed"] is True
    assert kv_floor["source_kv_exposed"] is True
    assert "byte-floor" in kv_floor["paper_claim"]

    summary = json.loads((tmp_path / "slo" / "serving_slo_envelope.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "slo" / "manifest.json").read_text(encoding="utf-8"))
    assert summary["headline"] == payload["headline"]
    assert "serving_slo_envelope.csv" in manifest["artifacts"]
    assert (tmp_path / "slo" / "serving_slo_envelope.md").exists()
