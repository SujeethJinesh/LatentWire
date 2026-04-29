from __future__ import annotations

import json

from scripts.run_source_private_tool_trace_learned_syndrome import run_gate


def test_tool_trace_learned_syndrome_passes_real_feature_smoke(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path,
        train_examples=128,
        eval_examples=64,
        train_family_set="all",
        eval_family_set="all",
        candidates=4,
        feature_dim=256,
        budgets=[6],
        train_seed=3,
        eval_seed=4,
        ridge=1e-2,
    )

    row = payload["budget_summaries"][0]
    assert payload["pass_gate"] is True
    assert payload["exact_id_parity"] is True
    assert row["matched_accuracy"] >= row["best_no_source_accuracy"] + 0.15
    assert row["metrics"]["zero_source"]["accuracy"] == row["metrics"]["target_only"]["accuracy"]
    assert row["metrics"]["full_diag_oracle"]["accuracy"] == 1.0


def test_tool_trace_learned_syndrome_writes_artifacts(tmp_path) -> None:
    run_gate(
        output_dir=tmp_path,
        train_examples=64,
        eval_examples=32,
        train_family_set="all",
        eval_family_set="all",
        candidates=4,
        feature_dim=128,
        budgets=[6],
        train_seed=5,
        eval_seed=6,
        ridge=1e-2,
    )

    summary = json.loads((tmp_path / "summary.json").read_text())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert summary["budget_summaries"][0]["budget_bytes"] == 6
    assert "predictions_budget6.jsonl" in manifest["artifacts"]
    assert (tmp_path / "predictions_budget6.jsonl").exists()
