from __future__ import annotations

import json

from scripts.run_source_private_learned_synonym_dictionary_packet_gate import run_gate


def test_learned_synonym_dictionary_gate_writes_artifacts(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "learned_synonym_dictionary",
        budgets=[4],
        train_examples=32,
        eval_examples=12,
        seed=11,
        candidate_atom_view="synonym_stress",
        candidate_calibration="all_public",
        calibration_examples=48,
        feature_dim=64,
        ridge=0.5,
        top_k=6,
        min_score=0.01,
    )

    summary = json.loads(
        (tmp_path / "learned_synonym_dictionary" / "learned_synonym_dictionary_packet_gate.json").read_text()
    )
    manifest = json.loads((tmp_path / "learned_synonym_dictionary" / "manifest.json").read_text())

    assert payload["gate"] == "source_private_learned_synonym_dictionary_packet_gate"
    assert summary["candidate_atom_view"] == "synonym_stress"
    assert set(summary["headline"]["direction_pass"]) == {"core_to_holdout", "holdout_to_core", "same_family_all"}
    assert "learned_synonym_dictionary_packet_gate.json" in manifest["artifacts"]
    assert (tmp_path / "learned_synonym_dictionary" / "core_to_holdout" / "predictions_budget4.jsonl").exists()


def test_learned_synonym_dictionary_rows_include_knockout_and_controls(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "learned_synonym_dictionary_controls",
        budgets=[4],
        train_examples=24,
        eval_examples=8,
        seed=13,
        candidate_atom_view="synonym_stress",
        candidate_calibration="train_only",
        calibration_examples=24,
        feature_dim=48,
        ridge=0.5,
        top_k=6,
        min_score=0.0,
    )

    direction = json.loads(
        (tmp_path / "learned_synonym_dictionary_controls" / "core_to_holdout" / "summary.json").read_text()
    )
    row = direction["budget_summaries"][0]
    metrics = row["metrics"]
    assert "learned_synonym_dictionary_packet" in metrics
    assert "atom_id_derangement" in metrics
    assert "top_atom_knockout" in metrics
    assert "private_random_knockout" in metrics
    assert row["budget_bytes"] == 4
    assert row["paired_bootstrap_vs_target"]["ci95_high"] >= row["paired_bootstrap_vs_target"]["ci95_low"]
    assert payload["headline"]["max_learned_synonym_dictionary_accuracy"] >= 0.0
