from __future__ import annotations

import json

from scripts.run_source_private_conditional_semantic_syndrome_gate import run_gate


def test_conditional_semantic_syndrome_gate_writes_artifacts(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "semantic_syndrome",
        budgets=[2],
        train_examples=24,
        eval_examples=8,
        seed=3,
        dim=32,
        ridge=1e-2,
        candidate_view="synonym_stress",
    )

    summary = json.loads((tmp_path / "semantic_syndrome" / "conditional_semantic_syndrome_gate.json").read_text())
    manifest = json.loads((tmp_path / "semantic_syndrome" / "manifest.json").read_text())

    assert payload["gate"] == "source_private_conditional_semantic_syndrome_gate"
    assert summary["candidate_view"] == "synonym_stress"
    assert set(summary["headline"]["direction_pass"]) == {"core_to_holdout", "holdout_to_core", "same_family_all"}
    assert "conditional_semantic_syndrome_gate.json" in manifest["artifacts"]
    assert (tmp_path / "semantic_syndrome" / "core_to_holdout" / "predictions_budget2.jsonl").exists()


def test_conditional_semantic_syndrome_rows_include_controls(tmp_path) -> None:
    run_gate(
        output_dir=tmp_path / "semantic_syndrome_controls",
        budgets=[2],
        train_examples=24,
        eval_examples=8,
        seed=5,
        dim=32,
        ridge=1e-2,
        candidate_view="native",
    )

    direction = json.loads((tmp_path / "semantic_syndrome_controls" / "core_to_holdout" / "summary.json").read_text())
    row = direction["budget_summaries"][0]
    metrics = row["metrics"]
    assert "conditional_semantic_syndrome" in metrics
    assert "wrong_projection_source" in metrics
    assert "oracle_candidate_residual" in metrics
    assert row["budget_bytes"] == 2
    assert row["paired_bootstrap_vs_target"]["ci95_high"] >= row["paired_bootstrap_vs_target"]["ci95_low"]
