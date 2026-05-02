from __future__ import annotations

import csv
import json

from scripts.build_source_private_iclr_gate_tree_and_connector_plan import build_gate_tree


def test_gate_tree_payload_records_readiness_and_required_nodes(tmp_path) -> None:
    payload = build_gate_tree(tmp_path / "gate")

    assert payload["paper_readiness"] == "COLM workshop plausible; ICLR full paper blocked."
    assert payload["current_story"].startswith("Fixed-byte source-private packets")
    assert "tokenwise" in payload["exact_gap"]
    assert len(payload["technical_contributions"]) == 3
    assert "OpenBookQA 3B" in payload["next_exact_gate"]

    nodes = {node["node_id"]: node for node in payload["nodes"]}
    assert nodes["fixed_byte_packet_protocol"]["status"] == "promote_for_colm"
    assert nodes["systems_boundary_accounting"]["status"] == "promote_for_colm"
    assert nodes["tinyllama_mean_cache_connectors"]["status"] == "cut"
    assert nodes["tokenwise_query_connector"]["status"] == "alive_next_gate"
    assert nodes["native_systems_rows"]["status"] == "blocked_required"


def test_gate_tree_writes_parseable_artifacts(tmp_path) -> None:
    build_gate_tree(tmp_path / "gate")

    out = tmp_path / "gate"
    payload = json.loads((out / "iclr_gate_tree_and_connector_plan.json").read_text(encoding="utf-8"))
    with (out / "iclr_gate_tree_nodes.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    md = (out / "iclr_gate_tree_and_connector_plan.md").read_text(encoding="utf-8")
    svg = (out / "iclr_gate_tree.svg").read_text(encoding="utf-8")
    runbook = (out / "tokenwise_connector_runbook.md").read_text(encoding="utf-8")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))

    assert payload["gate"] == "source_private_iclr_gate_tree_and_connector_plan"
    assert len(rows) == 7
    assert "## Next Exact Gate" in md
    assert "Pasteur" in md
    assert svg.startswith("<svg")
    assert "C2C/KVComm/KVCOMM" in runbook
    assert "QJL/TurboQuant" in runbook
    assert "source-label-copy audit upper bound" in runbook
    assert len(manifest["files"]) == 5
