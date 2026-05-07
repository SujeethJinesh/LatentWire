from __future__ import annotations

import json
from pathlib import Path

from experimental.hbsm.phase2.hbsm_synthetic_b1_gate import run_gate
from experimental.shared.check_gate_packet import validate_gate_packet


def test_synthetic_b1_rehearsal_exercises_real_schema(tmp_path: Path) -> None:
    output_dir = tmp_path / "hbsm_b1"

    summary = run_gate(output_dir=output_dir)
    report = validate_gate_packet(
        output_dir,
        mode="real",
        project="hbsm",
        expected_decision_prefix="SCHEMA_REHEARSAL_NOT_PROMOTABLE",
    )

    assert report["ok"], report["errors"]
    assert report["row_count"] == 720
    assert summary["decision"] == "SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_HBSM_B1"
    assert summary["gate_status"] == "PASS_REAL_B1_SENSITIVITY_HETEROGENEITY"
    assert summary["primary_row_count"] == 480
    assert summary["scoring_layer_count"] == 40
    assert "synthetic-only" in summary["claim_boundary"]
    assert (output_dir / "summary.md").exists()


def test_synthetic_b1_rehearsal_cannot_use_promoting_decision(tmp_path: Path) -> None:
    output_dir = tmp_path / "hbsm_b1"
    run_gate(output_dir=output_dir)
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["decision"] = summary["gate_status"]
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = validate_gate_packet(output_dir, mode="real", project="hbsm")

    assert not report["ok"]
    assert any("schema-rehearsal packet must use" in error for error in report["errors"])
