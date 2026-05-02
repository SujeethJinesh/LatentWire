from __future__ import annotations

from pathlib import Path

from scripts.build_source_private_conditional_pq_packet_isa_waterfall import build_table


def _artifact_root() -> Path:
    if Path("results").exists():
        return Path("results")
    if Path("../results").exists():
        return Path("../results")
    raise AssertionError("results artifact directory not found")


def test_conditional_packet_isa_waterfall_builds_from_existing_artifacts() -> None:
    artifacts = _artifact_root()
    payload = build_table(
        conditional_summary=artifacts
        / (
            "source_private_conditional_pq_innovation_gate_20260430/summary/"
            "conditional_pq_innovation_summary.json"
        ),
        waterfall_path=artifacts
        / (
            "source_private_pq_transport_receiver_waterfall_20260430/"
            "pq_transport_receiver_waterfall.json"
        ),
    )

    assert payload["pass_gate"] is True
    assert payload["rows"][0]["record_bytes"] == 5
    assert payload["rows"][1]["record_bytes"] == 7
    assert payload["rows"][0]["source_text_exposed"] is False
    assert payload["rows"][2]["source_text_exposed"] is True
    assert payload["rows"][4]["source_kv_exposed"] is True
