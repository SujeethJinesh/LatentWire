from __future__ import annotations

import json

from scripts.run_source_private_learned_syndrome_smoke import run_gate


def test_learned_syndrome_smoke_passes_low_rate(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path,
        train_examples=256,
        eval_examples=64,
        candidates=4,
        latent_dim=24,
        source_dim=32,
        budgets=[1, 2],
        seed=1,
        ridge=1e-2,
    )

    assert payload["exact_id_parity"] is True
    assert payload["candidate_pool_recall"] == 1.0
    assert payload["pass_gate"] is True
    passing = next(row for row in payload["budget_summaries"] if row["pass_gate"])
    assert passing["matched_accuracy"] >= passing["best_no_source_accuracy"] + 0.15
    assert passing["metrics"]["zero_source"]["accuracy"] == passing["metrics"]["target_only"]["accuracy"]
    assert passing["metrics"]["full_text_oracle"]["accuracy"] == 1.0


def test_learned_syndrome_writes_reproducible_artifacts(tmp_path) -> None:
    run_gate(
        output_dir=tmp_path,
        train_examples=64,
        eval_examples=32,
        candidates=4,
        latent_dim=16,
        source_dim=20,
        budgets=[1],
        seed=11,
        ridge=1e-2,
    )

    summary = json.loads((tmp_path / "summary.json").read_text())
    manifest = json.loads((tmp_path / "manifest.json").read_text())

    assert summary["budget_summaries"][0]["budget_bytes"] == 1
    assert "predictions_budget1.jsonl" in manifest["artifacts"]
    assert (tmp_path / "predictions_budget1.jsonl").exists()
