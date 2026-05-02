from __future__ import annotations

import json

from scripts.build_source_private_packet_isa_batch_frontier import (
    DEFAULT_HARDWARE_FRONTIER,
    build_packet_isa_batch_frontier,
)


def test_packet_isa_batch_frontier_records_packing_efficiency(tmp_path) -> None:
    payload = build_packet_isa_batch_frontier(
        hardware_frontier=DEFAULT_HARDWARE_FRONTIER,
        output_dir=tmp_path / "packet_isa",
        payload_bytes=[2, 4, 8],
        batch_sizes=[1, 4, 16],
        header_bytes=2,
        parity_bytes=1,
    )

    assert payload["gate"] == "source_private_packet_isa_batch_frontier"
    assert payload["pass_gate"] is True
    assert payload["headline"]["minimum_packet_bytes_with_overhead"] == 5
    assert payload["headline"]["best_min_payload_line_bytes_per_request"] <= 8.0
    assert payload["headline"]["max_line_packing_efficiency"] > 1.0

    single = next(row for row in payload["rows"] if row["payload_bytes"] == 2 and row["batch_size"] == 1)
    packed = next(row for row in payload["rows"] if row["payload_bytes"] == 2 and row["batch_size"] == 16)
    assert single["line_bytes_per_request_packed"] == 64.0
    assert packed["line_bytes_per_request_packed"] < single["line_bytes_per_request_packed"]
    assert payload["packet_contract"]["contract_name"].startswith("LatentWire")

    summary = json.loads((tmp_path / "packet_isa" / "packet_isa_batch_frontier.json").read_text())
    manifest = json.loads((tmp_path / "packet_isa" / "manifest.json").read_text())
    assert summary["headline"] == payload["headline"]
    assert "packet_isa_batch_frontier.csv" in manifest["artifacts"]
