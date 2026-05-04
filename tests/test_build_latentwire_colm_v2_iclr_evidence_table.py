from __future__ import annotations

import csv
import json

from scripts import build_latentwire_colm_v2_iclr_evidence_table as table


def _row(branch: str, status: str, *, score: float = 0.5, baseline: float = 0.25) -> dict:
    return {
        "artifact": branch.lower().replace(" ", "_"),
        "baseline": baseline,
        "branch": branch,
        "ci95_low": score - baseline,
        "decision": f"decision for {branch}",
        "delta": score - baseline,
        "evidence": f"evidence for {branch}",
        "record_bytes": 7,
        "score": score,
        "status": status,
    }


def _triage() -> dict:
    return {
        "branch_rows": [
            _row("Conditional PQ", "promote_for_colm_v2_only", score=0.65),
            _row("HellaSwag Fixed", "promote_for_colm_v2_systems_baseline", score=0.53),
            _row("Target Capacity", "capacity_alive_not_source_private_method", score=0.94),
            _row("Scalar Integrity", "ruled_out_simple_integrity_threshold", score=0.42, baseline=0.46),
            _row("Query Resampler", "ruled_out_current_target_native_encoder_family", score=0.5, baseline=0.5),
        ],
        "claim_boundaries": ["Do not claim broad cross-family latent communication yet."],
        "current_contributions": [
            {
                "name": "source_private_low_rate_packets",
                "status": "alive_for_colm_v2",
                "gap": "needs broader receiver evidence",
            }
        ],
        "next_exact_gate": {
            "fallback_path": "COLM_v2 table integration",
            "name": "new_interface_or_colm_v2_integration_gate",
            "pass_bar": "positive paired CI",
            "primary_path": "new source-causal interface",
            "required_controls": ["target_only"],
        },
        "readiness": {
            "colm_v2": "scoped_positive_ready_for_writeup_if_claims_are_narrow",
            "iclr": "blocked_by_lack_of_broad_or_learned_positive_receiver",
        },
        "story": "Scoped source-private packets plus strict destructive controls.",
        "submission_gap": "ICLR needs a positive learned receiver.",
    }


def test_build_table_classifies_colm_and_iclr_rows() -> None:
    built = table.build_table(_triage())

    assert [row["branch"] for row in built["colm_v2_core_rows"]] == [
        "Conditional PQ",
        "HellaSwag Fixed",
    ]
    assert "Target Capacity" in [row["branch"] for row in built["colm_v2_supporting_rows"]]
    assert "Query Resampler" in [row["branch"] for row in built["iclr_blocker_rows"]]
    assert any(item["work"] == "Cache-to-Cache (C2C)" for item in built["literature_boundaries"])
    assert "Sparse Resonance Packets" in built["paper_decision"]["iclr_claim_not_yet_supported"]


def test_write_outputs_emits_json_csv_and_markdown(tmp_path) -> None:
    built = table.build_table(_triage())
    output_dir = tmp_path / "out"
    paper_path = tmp_path / "paper.md"

    table.write_outputs(built, output_dir, paper_path)

    payload = json.loads((output_dir / "evidence_table.json").read_text(encoding="utf-8"))
    assert payload["readiness"]["iclr"].startswith("blocked")

    rows = list(csv.DictReader((output_dir / "evidence_table.csv").open(encoding="utf-8")))
    assert rows
    assert {row["section"] for row in rows} >= {"colm_v2_core", "iclr_blocker"}

    markdown = paper_path.read_text(encoding="utf-8")
    assert "## COLM_v2 Core Rows" in markdown
    assert "Cache-to-Cache (C2C)" in markdown
    assert "new_interface_or_colm_v2_integration_gate" in markdown
