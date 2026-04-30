from __future__ import annotations

import json

from scripts.run_source_private_anchor_relative_crosscoder_receiver_gate import (
    CONDITIONS,
    SOURCE_DESTROYING_CONTROLS,
    run_gate,
)


def test_anchor_relative_crosscoder_receiver_gate_writes_strict_artifacts(tmp_path) -> None:
    output_dir = tmp_path / "anchor_crosscoder"
    payload = run_gate(
        output_dir=output_dir,
        train_examples=48,
        eval_examples=16,
        train_family_set="core",
        eval_family_set="holdout",
        candidates=4,
        feature_dim=96,
        candidate_feature_dims=0,
        receiver_kind="ridge",
        packet_feature_mode="anchor_relative",
        anchor_count=24,
        candidate_view="diag_only",
        diagnostic_table_mode="plausible_decoys",
        budgets=[2],
        train_seed=11,
        eval_seed=12,
        ridge=1e-2,
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    manifest = json.loads((output_dir / "manifest.json").read_text())

    assert payload["gate"] == "source_private_anchor_relative_crosscoder_receiver_gate"
    assert summary["candidate_view"] == "diag_only"
    assert summary["diagnostic_table_mode"] == "plausible_decoys"
    assert summary["exact_id_parity"] is True
    assert summary["candidate_pool_recall"] == 1.0
    assert set(summary["conditions"]) == set(CONDITIONS)
    assert set(summary["source_destroying_controls"]) == set(SOURCE_DESTROYING_CONTROLS)
    assert "predictions_budget2.jsonl" in manifest["artifacts"]
    assert (output_dir / "predictions_budget2.jsonl").exists()


def test_anchor_relative_crosscoder_receiver_gate_has_ordered_id_parity_and_controls(tmp_path) -> None:
    output_dir = tmp_path / "anchor_crosscoder_controls"
    payload = run_gate(
        output_dir=output_dir,
        train_examples=40,
        eval_examples=12,
        train_family_set="holdout",
        eval_family_set="core",
        candidates=4,
        feature_dim=80,
        candidate_feature_dims=0,
        receiver_kind="code_similarity",
        packet_feature_mode="learned_anchor_relative",
        anchor_count=20,
        candidate_view="semantic",
        diagnostic_table_mode="plausible_decoys",
        budgets=[2],
        train_seed=13,
        eval_seed=14,
        ridge=1e-2,
    )

    row = payload["budget_summaries"][0]
    metrics = row["metrics"]
    parity = row["condition_id_parity"]

    assert payload["packet_feature_mode"] == "learned_anchor_relative"
    assert payload["anchor_build_mode"] == "deterministic_spherical_kmeans"
    assert row["exact_ordered_id_parity"] is True
    assert parity["all_conditions_same_order"] is True
    assert "public_only_sidecar" in metrics
    assert "feature_id_permutation" in metrics
    assert "top_feature_knockout" in metrics
    assert row["paired_bootstrap_vs_target"]["ci95_high"] >= row["paired_bootstrap_vs_target"]["ci95_low"]
    assert row["paired_bootstrap_vs_best_control"]["ci95_high"] >= row["paired_bootstrap_vs_best_control"]["ci95_low"]
