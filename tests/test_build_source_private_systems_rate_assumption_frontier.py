from __future__ import annotations

import json

from scripts.build_source_private_systems_rate_assumption_frontier import (
    DEFAULT_KV_TABLE,
    DEFAULT_RATE_FRONTIER,
    DEFAULT_SEMANTIC_MEDIUM,
    DEFAULT_SYSTEMS_CAVEAT,
    build_systems_rate_assumption_frontier,
)


def test_systems_rate_assumption_frontier_records_assumptions(tmp_path) -> None:
    payload = build_systems_rate_assumption_frontier(
        systems_caveat=DEFAULT_SYSTEMS_CAVEAT,
        rate_frontier=DEFAULT_RATE_FRONTIER,
        kv_table=DEFAULT_KV_TABLE,
        semantic_medium=DEFAULT_SEMANTIC_MEDIUM,
        output_dir=tmp_path / "frontier",
    )

    assert payload["gate"] == "source_private_systems_rate_assumption_frontier"
    assert payload["pass_gate"] is True
    assert payload["headline"]["semantic_medium_pass_rows"] == payload["headline"]["semantic_medium_total_rows"]
    assert payload["headline"]["same_byte_text_accuracy_max"] == 0.25
    assert payload["headline"]["query_aware_text_bytes_vs_packet"] >= 7.0
    assert payload["headline"]["min_kv_byte_floor_vs_packet"] >= 1000.0
    assert payload["headline"]["external_reference_rows"] >= 4

    rows = payload["rows"]
    endpoint = next(row for row in rows if row["row_class"] == "endpoint_packet")
    kv = next(row for row in rows if row["row_class"] == "kv_byte_floor")
    external = next(row for row in rows if row["method"].startswith("C2C"))
    contract = next(row for row in rows if row["row_class"] == "contract_failure")

    assert endpoint["source_private"] is True
    assert endpoint["source_destroying_controls"] == "passed"
    assert endpoint["claim_allowed"] == "headline_endpoint_proxy"
    assert kv["claim_allowed"] == "accounting_only"
    assert kv["source_kv_exposed"] is True
    assert external["claim_allowed"] == "reference_only"
    assert external["source_destroying_controls"] == "not_applicable"
    assert contract["accuracy"] == contract["target_accuracy"]

    summary = json.loads((tmp_path / "frontier" / "systems_rate_assumption_frontier.json").read_text())
    manifest = json.loads((tmp_path / "frontier" / "manifest.json").read_text())
    assert summary["headline"] == payload["headline"]
    assert "systems_rate_assumption_frontier.csv" in manifest["artifacts"]
    assert (tmp_path / "frontier" / "systems_rate_assumption_frontier.md").exists()
