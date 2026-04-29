from __future__ import annotations

import json

from scripts.run_source_private_masked_innovation_receiver import run_gate


def test_masked_innovation_receiver_writes_artifacts(tmp_path) -> None:
    output_dir = tmp_path / "masked_innovation"
    payload = run_gate(
        output_dir=output_dir,
        train_examples=32,
        eval_examples=16,
        train_family_set="all",
        eval_family_set="all",
        candidates=4,
        feature_dim=64,
        anchor_count=24,
        candidate_view="anchor_relative",
        source_topk=16,
        target_topk=12,
        budgets=[2],
        train_seed=3,
        eval_seed=4,
        ridge=1e-2,
        mask_repeats=0,
        calibration_examples=12,
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    manifest = json.loads((output_dir / "manifest.json").read_text())
    row = payload["budget_summaries"][0]
    assert payload["gate"] == "source_private_masked_innovation_receiver"
    assert payload["exact_id_parity"] is True
    assert summary["calibration_examples"] == 12
    assert summary["candidate_view"] == "anchor_relative"
    assert summary["anchor_count"] == 24
    assert row["budget_bytes"] == 2
    assert "public_only_innovation" in row["metrics"]
    assert "shuffled_mask_or_atoms" in row["metrics"]
    assert "predictions_budget2.jsonl" in manifest["artifacts"]
    assert (output_dir / "predictions_budget2.jsonl").exists()


def test_masked_innovation_receiver_shared_text_view(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "masked_innovation_shared",
        train_examples=24,
        eval_examples=12,
        train_family_set="all",
        eval_family_set="all",
        candidates=4,
        feature_dim=64,
        anchor_count=24,
        candidate_view="shared_text",
        source_topk=16,
        target_topk=12,
        budgets=[2],
        train_seed=7,
        eval_seed=8,
        ridge=1e-2,
        mask_repeats=0,
        calibration_examples=8,
    )

    assert payload["candidate_view"] == "shared_text"
    assert payload["representation_dim"] == 64
    assert payload["budget_summaries"][0]["budget_bytes"] == 2
