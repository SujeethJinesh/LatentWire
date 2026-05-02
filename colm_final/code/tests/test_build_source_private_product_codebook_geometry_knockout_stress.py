from __future__ import annotations

import json

from scripts.build_source_private_product_codebook_geometry_knockout_stress import (
    build_geometry_knockout_stress,
)


def test_geometry_knockout_stress_compares_canonical_and_protected_variants(tmp_path) -> None:
    payload = build_geometry_knockout_stress(
        output_dir=tmp_path / "geometry_knockout",
        remap_seeds=[101],
        budgets=[2],
        variants=["canonical", "opq_procrustes", "protected_hadamard"],
        train_examples=64,
        eval_examples=24,
        feature_dim=64,
        train_seed=5,
        eval_seed=6,
        candidate_view="slot",
        bootstrap_samples=50,
        opq_iterations=1,
    )

    assert payload["gate"] == "source_private_product_codebook_geometry_knockout_stress"
    assert payload["headline"]["rows"] == 3
    assert {row["variant"] for row in payload["rows"]} == {
        "canonical",
        "opq_procrustes",
        "protected_hadamard",
    }
    for row in payload["rows"]:
        assert row["exact_id_parity"] is True
        assert row["candidate_pool_recall"] == 1.0
        assert row["mean_payload_bytes"] == 2
        assert row["payload_entropy"]["unique_payloads"] >= 1
        assert row["payload_reuse"]["singleton_n"] + row["payload_reuse"]["collision_n"] == 24
        assert "top_codeword_removed_worst" in row["paired_comparisons"]
        assert "source_minus_canonical" in row
    predictions = (
        tmp_path
        / "geometry_knockout"
        / "remap_101"
        / "budget_2"
        / "protected_hadamard"
        / "predictions.jsonl"
    )
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    assert {row["condition"] for row in rows} >= {
        "source",
        "top_codeword_removed_mean",
        "random_codeword_removed_mean",
        "mean_payload",
    }
    assert (tmp_path / "geometry_knockout" / "product_codebook_geometry_knockout_stress.json").exists()
    assert (tmp_path / "geometry_knockout" / "manifest.json").exists()
