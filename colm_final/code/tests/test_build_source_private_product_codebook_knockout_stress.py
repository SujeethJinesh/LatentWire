from __future__ import annotations

import json

from scripts.build_source_private_product_codebook_knockout_stress import build_knockout_stress


def test_product_codebook_knockout_stress_writes_artifacts_and_parity(tmp_path) -> None:
    payload = build_knockout_stress(
        output_dir=tmp_path / "knockout",
        remap_seeds=[101],
        budgets=[2],
        train_examples=64,
        eval_examples=24,
        feature_dim=128,
        train_seed=5,
        eval_seed=6,
        candidate_view="slot",
        bootstrap_samples=50,
    )

    assert payload["gate"] == "source_private_product_codebook_knockout_stress"
    assert payload["headline"]["rows"] == 1
    row = payload["rows"][0]
    assert row["exact_id_parity"] is True
    assert row["candidate_pool_recall"] == 1.0
    assert row["mean_payload_bytes"] == 2
    assert "top_worst_lift_removed_fraction" in row
    assert row["payload_entropy"]["unique_payloads"] >= 1
    assert len(row["payload_entropy"]["codeword_summaries"]) == 2
    assert "product_codebook_source" in row["metrics"]
    assert "top_codeword_removed_worst" in row["paired_comparisons"]

    predictions_path = tmp_path / "knockout" / "remap_101" / "predictions_budget2.jsonl"
    rows = [json.loads(line) for line in predictions_path.read_text().splitlines()]
    by_condition: dict[str, list[dict[str, object]]] = {}
    for prediction in rows:
        by_condition.setdefault(str(prediction["condition"]), []).append(prediction)
    assert set(by_condition) == {
        "target_only",
        "product_codebook_source",
        "top_codeword_removed_mean",
        "top_codeword_removed_worst",
        "random_codeword_removed_mean",
        "random_codeword_removed_random",
        "top_codeword_only",
        "mean_payload",
    }
    assert {len(condition_rows) for condition_rows in by_condition.values()} == {24}
    assert {prediction["payload_bytes"] for prediction in by_condition["product_codebook_source"]} == {2}
    assert {prediction["payload_bytes"] for prediction in by_condition["top_codeword_removed_worst"]} == {2}
    assert {prediction["payload_bytes"] for prediction in by_condition["target_only"]} == {0}
    assert (tmp_path / "knockout" / "product_codebook_knockout_stress.json").exists()
    assert (tmp_path / "knockout" / "manifest.json").exists()
