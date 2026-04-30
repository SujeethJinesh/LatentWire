from __future__ import annotations

import json

from scripts import run_source_private_public_only_receiver_ablation as gate


def test_public_only_features_include_candidate_rows() -> None:
    examples = gate.make_benchmark(examples=4, candidates=4, seed=3, family_set="all")
    features = gate._public_features(examples[0], feature_dim=32, candidate_view="semantic", include_vectors=True)

    assert features.shape == (4, 35)


def test_public_only_gate_writes_artifacts(tmp_path) -> None:
    payload = gate.run_gate(
        output_dir=tmp_path,
        train_examples=32,
        eval_examples=16,
        train_seed=5,
        eval_seed=6,
        train_start_index=0,
        eval_start_index=10_000,
        train_family_set="all",
        eval_family_set="all",
        diagnostic_table_mode="legacy",
        candidates=4,
        feature_dim=32,
        ridge=1e-2,
        candidate_view="semantic",
        remap_slot_seed=None,
        include_vectors=True,
        max_allowed_lift=0.05,
    )

    assert payload["summary"]["n"] == 16
    assert payload["summary"]["train_eval_id_intersection_count"] == 0
    assert 0.0 <= payload["summary"]["public_only_accuracy"] <= 1.0
    assert (tmp_path / "predictions.jsonl").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert "run_summary.json" in manifest["artifacts"]
