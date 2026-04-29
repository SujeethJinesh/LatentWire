from __future__ import annotations

import json

from scripts.run_source_private_candidate_embedding_receiver import run_gate


def test_candidate_embedding_receiver_passes_smoke(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "receiver",
        train_examples=128,
        eval_examples=64,
        train_family_set="all",
        eval_family_set="all",
        candidates=4,
        feature_dim=256,
        budgets=[4],
        train_seed=3,
        eval_seed=4,
        ridge=1e-2,
    )

    row = payload["budget_summaries"][0]
    assert payload["gate"] == "source_private_candidate_embedding_receiver"
    assert payload["exact_id_parity"] is True
    assert row["matched_accuracy"] >= row["target_only_accuracy"] + 0.15
    assert row["matched_accuracy"] >= row["best_destructive_control_accuracy"] + 0.15
    assert row["full_diag_oracle_accuracy"] >= 0.95


def test_candidate_embedding_receiver_writes_artifacts(tmp_path) -> None:
    output_dir = tmp_path / "receiver"
    run_gate(
        output_dir=output_dir,
        train_examples=64,
        eval_examples=32,
        train_family_set="all",
        eval_family_set="all",
        candidates=4,
        feature_dim=128,
        budgets=[2],
        train_seed=5,
        eval_seed=6,
        ridge=1e-2,
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert summary["budget_summaries"][0]["budget_bytes"] == 2
    assert "predictions_budget2.jsonl" in manifest["artifacts"]
    assert (output_dir / "predictions_budget2.jsonl").exists()
