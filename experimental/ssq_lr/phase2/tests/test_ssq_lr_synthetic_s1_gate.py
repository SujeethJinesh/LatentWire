from __future__ import annotations

import json
from pathlib import Path

from experimental.shared.check_gate_packet import validate_gate_packet
from experimental.ssq_lr.phase2.ssq_lr_synthetic_s1_gate import run_gate


def test_synthetic_s1_rehearsal_exercises_real_schema(tmp_path: Path) -> None:
    output_dir = tmp_path / "ssq_lr_synthetic_s1"

    summary = run_gate(output_dir=output_dir)
    report = validate_gate_packet(output_dir, mode="real", project="ssq_lr")

    assert report["ok"], report["errors"]
    assert report["decision"] == "SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_SSQ_LR_S1"
    assert report["row_count"] == 12 * 6 * 4
    # The synthetic fixture drives the metric evaluator through a passing path,
    # but the packet decision remains schema-rehearsal/non-promoting.
    assert summary["gate_status"] == "PASS_REAL_S1_HETEROGENEITY"
    assert summary["decision"].startswith("SCHEMA_REHEARSAL_NOT_PROMOTABLE")
    assert summary["evidence_kind"] == "schema_rehearsal"
    assert summary["promotable"] is False
    assert "synthetic-only" in summary["claim_boundary"]
    summary_md = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "Fixture-only evaluator fields, not a promotion decision" in summary_md
    assert "raw evaluator readout only" in summary_md


def test_synthetic_s1_rehearsal_cannot_use_promoting_decision(tmp_path: Path) -> None:
    output_dir = tmp_path / "ssq_lr_synthetic_s1"
    run_gate(output_dir=output_dir)
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["decision"] = str(summary["gate_status"])
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    report = validate_gate_packet(output_dir, mode="real", project="ssq_lr")

    assert not report["ok"]
    assert any("schema-rehearsal packet must use" in error for error in report["errors"])


def test_synthetic_s1_rehearsal_requires_non_promoting_summary_flags(tmp_path: Path) -> None:
    output_dir = tmp_path / "ssq_lr_synthetic_s1"
    run_gate(output_dir=output_dir)
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["promotable"] = True
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    report = validate_gate_packet(output_dir, mode="real", project="ssq_lr")

    assert not report["ok"]
    assert any("promotable must be false" in error for error in report["errors"])
