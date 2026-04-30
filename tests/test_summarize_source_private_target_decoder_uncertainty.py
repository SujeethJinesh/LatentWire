from __future__ import annotations

import json

from scripts.summarize_source_private_target_decoder_uncertainty import run_summary


def test_target_decoder_uncertainty_summary_writes_artifacts(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = []
    for index in range(12):
        answer = "a" if index < 6 else "b"
        for condition in [
            "target_only",
            "matched_packet",
            "shuffled_packet",
            "random_same_byte",
            "structured_json_2byte",
            "structured_free_text_2byte",
        ]:
            prediction = answer if condition == "matched_packet" else "a"
            rows.append(
                {
                    "example_id": f"e{index}",
                    "condition": condition,
                    "answer_label": answer,
                    "prediction": prediction,
                    "correct": prediction == answer,
                    "valid_prediction": True,
                    "latency_ms": 1.0,
                    "payload_bytes": 2 if condition != "target_only" else 0,
                    "payload_tokens": 1 if condition != "target_only" else 0,
                    "generated_tokens": 1,
                }
            )
    (run_dir / "target_predictions.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    metrics = {}
    for condition in {row["condition"] for row in rows}:
        condition_rows = [row for row in rows if row["condition"] == condition]
        metrics[condition] = {
            "accuracy": sum(1 for row in condition_rows if row["correct"]) / len(condition_rows),
            "valid_prediction_rate": 1.0,
            "p50_latency_ms": 1.0,
            "mean_generated_tokens": 1.0,
            "mean_payload_bytes": 2.0 if condition != "target_only" else 0.0,
        }
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "n": 12,
                "pass_gate": True,
                "exact_id_parity": True,
                "matched_accuracy": metrics["matched_packet"]["accuracy"],
                "target_only_accuracy": metrics["target_only"]["accuracy"],
                "matched_minus_target": metrics["matched_packet"]["accuracy"] - metrics["target_only"]["accuracy"],
                "matched_minus_best_control": metrics["matched_packet"]["accuracy"] - metrics["shuffled_packet"]["accuracy"],
                "metrics": metrics,
            }
        ),
        encoding="utf-8",
    )

    payload = run_summary(
        result_dirs=[run_dir],
        output_dir=tmp_path / "summary",
        bootstrap_samples=50,
        seed=5,
    )

    assert payload["gate"] == "source_private_target_decoder_uncertainty"
    assert payload["headline"]["rows"] == 1
    assert payload["rows"][0]["matched_valid_prediction_rate"] == 1.0
    assert (tmp_path / "summary" / "target_decoder_uncertainty.json").exists()
