from __future__ import annotations

import json

from scripts.run_source_private_shared_sparse_crosscoder_packet_gate import run_gate


def test_shared_sparse_crosscoder_packet_gate_writes_artifacts(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "shared_sparse",
        budgets=[4],
        train_examples=32,
        eval_examples=16,
        seed=3,
        candidate_atom_view="native",
    )

    summary = json.loads((tmp_path / "shared_sparse" / "shared_sparse_crosscoder_packet_gate.json").read_text())
    manifest = json.loads((tmp_path / "shared_sparse" / "manifest.json").read_text())

    assert payload["gate"] == "source_private_shared_sparse_crosscoder_packet_gate"
    assert summary["budgets"] == [4]
    assert set(summary["headline"]["direction_pass"]) == {"core_to_holdout", "holdout_to_core", "same_family_all"}
    assert "shared_sparse_crosscoder_packet_gate.json" in manifest["artifacts"]
    assert (tmp_path / "shared_sparse" / "core_to_holdout" / "predictions_budget4.jsonl").exists()


def test_shared_sparse_crosscoder_packet_rows_include_controls(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "shared_sparse_controls",
        budgets=[4],
        train_examples=24,
        eval_examples=8,
        seed=5,
        candidate_atom_view="native",
    )

    direction = json.loads((tmp_path / "shared_sparse_controls" / "core_to_holdout" / "summary.json").read_text())
    row = direction["budget_summaries"][0]
    metrics = row["metrics"]
    assert "shared_sparse_packet" in metrics
    assert "atom_id_derangement" in metrics
    assert "top_atom_knockout" in metrics
    assert "private_random_knockout" in metrics
    assert row["budget_bytes"] == 4
    assert row["paired_bootstrap_vs_target"]["ci95_high"] >= row["paired_bootstrap_vs_target"]["ci95_low"]
    assert payload["headline"]["max_shared_sparse_accuracy"] >= 0.0


def test_shared_sparse_crosscoder_packet_synonym_stress_view(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "shared_sparse_synonym",
        budgets=[4],
        train_examples=24,
        eval_examples=8,
        seed=7,
        candidate_atom_view="synonym_stress",
    )

    assert payload["candidate_atom_view"] == "synonym_stress"
    direction = json.loads((tmp_path / "shared_sparse_synonym" / "core_to_holdout" / "summary.json").read_text())
    assert direction["candidate_atom_view"] == "synonym_stress"
    assert direction["budget_summaries"][0]["budget_bytes"] == 4
