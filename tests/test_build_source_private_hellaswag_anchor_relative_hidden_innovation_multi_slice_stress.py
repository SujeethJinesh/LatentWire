import json

from scripts import build_source_private_hellaswag_anchor_relative_hidden_innovation_multi_slice_stress as multi


def _artifact(path, *, start: int, selected: float, best_label: float, score_only: float, pass_gate: bool) -> None:
    payload = {
        "gate": "source_private_hellaswag_anchor_relative_hidden_innovation_gate",
        "pass_gate": pass_gate,
        "eval_path": f"results/unit/hellaswag_validation_rows_{start}_{start + 4}.jsonl",
        "packet_contract": {
            "raw_payload_bytes": 2,
            "framed_record_bytes": 5,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_vector_transmitted": False,
            "raw_scores_transmitted": False,
        },
        "headline": {
            "eval_rows": 4,
            "selected_eval_accuracy": selected,
            "source_label_copy_eval_accuracy": best_label,
            "trained_choice_bias_label_copy_eval_accuracy": best_label,
            "best_label_copy_eval_accuracy": best_label,
            "selected_minus_best_label_copy": selected - best_label,
            "paired_ci95_low_vs_best_label_copy": -0.01 if not pass_gate else 0.01,
            "paired_ci95_high_vs_best_label_copy": 0.03,
            "score_only_bagged_control_accuracy": score_only,
            "selected_minus_score_only_bagged_control": selected - score_only,
            "paired_ci95_low_vs_score_only_bagged": 0.01,
            "zero_hidden_control_accuracy": score_only,
            "selected_minus_zero_hidden_control": selected - score_only,
            "wrong_example_hidden_control_accuracy": best_label - 0.01,
            "candidate_roll_hidden_control_accuracy": best_label - 0.01,
            "anchor_id_shuffle_control_accuracy": best_label - 0.01,
            "anchor_value_roll_control_accuracy": best_label - 0.01,
        },
        "jackknife_summary": {
            "pass_count": 3 if pass_gate else 0,
            "row_count": 3,
            "selected_minus_best_label_copy_min": selected - best_label,
            "paired_ci95_low_vs_best_label_copy_min": -0.01 if not pass_gate else 0.01,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_anchor_relative_multi_slice_records_demoted_common_basis(tmp_path) -> None:
    artifacts = []
    for index, start in enumerate((0, 4, 8)):
        path = tmp_path / f"slice_{index}.json"
        _artifact(path, start=start, selected=0.51, best_label=0.50, score_only=0.49, pass_gate=False)
        artifacts.append(path)

    payload = multi.build_gate(
        output_dir=tmp_path / "out",
        slice_artifacts=tuple(artifacts),
        dense_multi_slice_artifact=None,
        run_date="2026-05-01",
    )

    assert payload["pass_gate"] is False
    assert payload["headline"]["slice_count"] == 3
    assert payload["headline"]["total_eval_rows"] == 12
    assert payload["headline"]["contiguous_validation_prefix"] is True
    assert payload["headline"]["source_private_packet"] is True
    assert payload["headline"]["weighted_delta_vs_best_label_copy"] > 0.0
    assert payload["headline"]["min_delta_vs_best_label_copy"] < multi.STRICT_DELTA
    assert "common-basis blocker" in payload["interpretation"]
    assert (tmp_path / "out" / "hellaswag_anchor_relative_hidden_innovation_multi_slice_stress.json").exists()
