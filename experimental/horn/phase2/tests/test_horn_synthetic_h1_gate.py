from __future__ import annotations

import json
from pathlib import Path

from experimental.horn.phase2.horn_synthetic_h1_gate import run_gate
from experimental.shared.check_gate_packet import validate_gate_packet


def test_synthetic_h1a_rehearsal_exercises_real_schema(tmp_path: Path) -> None:
    output_dir = tmp_path / "horn_h1a"

    summary = run_gate(output_dir=output_dir)
    report = validate_gate_packet(
        output_dir,
        mode="real",
        project="horn",
        expected_decision_prefix="SCHEMA_REHEARSAL_NOT_PROMOTABLE",
    )

    assert report["ok"], report["errors"]
    assert report["row_count"] == 72
    assert summary["decision"] == "SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_HORN_H1A"
    # The synthetic fixture drives the metric evaluator through a passing path,
    # but the packet decision remains schema-rehearsal/non-promoting.
    assert summary["gate_status"] == "PASS_REAL_H1A_DIRECTIONAL_ASYMMETRY_SCREEN"
    assert summary["decision"].startswith("SCHEMA_REHEARSAL_NOT_PROMOTABLE")
    assert summary["prompt_count"] == 12
    assert summary["selected_h1_direction"] == "ssm->attention"
    assert summary["permuted_direction_ratio"] <= 1.0
    assert "synthetic-only" in summary["claim_boundary"]
    assert (output_dir / "summary.md").exists()


def test_synthetic_h1a_rehearsal_cannot_use_promoting_decision(tmp_path: Path) -> None:
    output_dir = tmp_path / "horn_h1a"
    run_gate(output_dir=output_dir)
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["decision"] = summary["gate_status"]
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = validate_gate_packet(output_dir, mode="real", project="horn")

    assert not report["ok"]
    assert any("schema-rehearsal packet must use" in error for error in report["errors"])
