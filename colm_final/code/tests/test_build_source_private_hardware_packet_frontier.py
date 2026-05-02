from __future__ import annotations

import json

from scripts.build_source_private_hardware_packet_frontier import (
    DEFAULT_SYSTEMS_FRONTIER,
    build_source_private_hardware_packet_frontier,
)


def test_hardware_packet_frontier_records_memory_accounting(tmp_path) -> None:
    payload = build_source_private_hardware_packet_frontier(
        systems_frontier=DEFAULT_SYSTEMS_FRONTIER,
        output_dir=tmp_path / "hardware_frontier",
    )

    assert payload["gate"] == "source_private_hardware_packet_frontier"
    assert payload["pass_gate"] is True
    assert payload["headline"]["packet_raw_bytes_min"] == 2.0
    assert payload["headline"]["packet_cacheline_bytes_min"] == 64.0
    assert payload["headline"]["query_aware_text_raw_ratio_min"] >= 7.0
    assert payload["headline"]["kv_raw_ratio_min"] >= 1000.0
    assert payload["headline"]["kv_cacheline_ratio_min"] >= 300.0

    endpoint = next(row for row in payload["rows"] if row["row_class"] == "endpoint_packet")
    full_log = next(row for row in payload["rows"] if row["method"] == "full hidden-log relay")
    kv = next(row for row in payload["rows"] if row["row_class"] == "kv_byte_floor")

    assert endpoint["cache_lines_64b"] == 1
    assert endpoint["dma_bursts_128b"] == 1
    assert endpoint["source_private"] is True
    assert full_log["cacheline_ratio_vs_packet"] > 1.0
    assert kv["source_kv_exposed"] is True

    contract = json.loads((tmp_path / "hardware_frontier" / "packet_contract.json").read_text())
    manifest = json.loads((tmp_path / "hardware_frontier" / "manifest.json").read_text())
    assert contract["contract_name"].startswith("LatentWire")
    assert "shuffled-source" in contract["control_requirements"]
    assert "hardware_packet_frontier.csv" in manifest["artifacts"]
    assert (tmp_path / "hardware_frontier" / "hardware_packet_frontier.md").exists()
