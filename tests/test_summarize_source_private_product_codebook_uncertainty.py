from __future__ import annotations

import json

from scripts.summarize_source_private_product_codebook_uncertainty import summarize_uncertainty


def _write_prediction_rows(path, *, budget: int) -> None:
    rows = []
    for index in range(8):
        answer = "a" if index < 6 else "b"
        rows.extend(
            [
                {
                    "example_id": f"e{index}",
                    "condition": "target_only",
                    "answer": answer,
                    "prediction": "a",
                    "correct": answer == "a",
                    "budget_bytes": budget,
                },
                {
                    "example_id": f"e{index}",
                    "condition": "scalar_quantized_source",
                    "answer": answer,
                    "prediction": answer if index < 7 else "a",
                    "correct": index < 7,
                    "budget_bytes": budget,
                },
                {
                    "example_id": f"e{index}",
                    "condition": "product_codebook_source",
                    "answer": answer,
                    "prediction": answer,
                    "correct": True,
                    "budget_bytes": budget,
                },
            ]
        )
        for condition in [
            "product_codebook_label_shuffled_ridge",
            "product_codebook_constrained_shuffled_source",
            "product_codebook_answer_masked_source",
            "product_codebook_permuted_codes",
            "product_codebook_random_same_byte",
            "qjl_residual_source",
            "protected_rotated_residual_source",
            "rotation_sign_source",
        ]:
            rows.append(
                {
                    "example_id": f"e{index}",
                    "condition": condition,
                    "answer": answer,
                    "prediction": "a",
                    "correct": answer == "a",
                    "budget_bytes": budget,
                }
            )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_product_codebook_uncertainty_summary_writes_artifacts(tmp_path) -> None:
    gate_dir = tmp_path / "gate"
    run_dir = gate_dir / "remap_101"
    run_dir.mkdir(parents=True)
    _write_prediction_rows(run_dir / "predictions_budget2.jsonl", budget=2)

    payload = summarize_uncertainty(
        product_gate_dir=gate_dir,
        product_gate_json=None,
        output_dir=tmp_path / "uncertainty",
        bootstrap_samples=50,
        seed=7,
    )

    assert payload["gate"] == "source_private_product_codebook_uncertainty"
    assert payload["headline"]["rows"] == 1
    assert payload["rows"][0]["exact_id_parity"] is True
    assert "paired_vs_best_control_ci95_low" in payload["rows"][0]
    assert (tmp_path / "uncertainty" / "product_codebook_uncertainty.json").exists()
