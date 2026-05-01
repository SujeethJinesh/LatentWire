from __future__ import annotations

import json

from scripts import build_source_private_hellaswag_learned_residual_basis_multi_slice_stress as gate


def _payload(*, accuracy: float, label: float, score: float, passed: bool) -> dict[str, object]:
    variant = {
        "variant": "pca256_top2_gold",
        "selected_eval_accuracy": accuracy,
        "source_label_copy_eval_accuracy": label,
        "trained_choice_bias_label_copy_eval_accuracy": label - 0.01,
        "best_label_copy_eval_accuracy": label,
        "selected_minus_best_label_copy": accuracy - label,
        "paired_ci95_low_vs_best_label_copy": 0.01 if passed else -0.01,
        "paired_ci95_high_vs_best_label_copy": 0.04,
        "score_only_bagged_control_accuracy": score,
        "selected_minus_score_only_bagged_control": accuracy - score,
        "paired_ci95_low_vs_score_only_bagged": 0.01 if passed else -0.01,
        "paired_ci95_high_vs_score_only_bagged": 0.04,
        "zero_hidden_control_accuracy": score,
        "selected_minus_zero_hidden_control": accuracy - score,
        "wrong_example_hidden_control_accuracy": label - 0.01,
        "candidate_roll_hidden_control_accuracy": label - 0.02,
        "basis_dim_roll_control_accuracy": label - 0.01,
        "basis_sign_flip_control_accuracy": label - 0.03,
        "random_basis_same_dim_control_accuracy": label - 0.01,
        "scout_pass_rule": passed,
    }
    return {
        "gate": "source_private_hellaswag_learned_residual_basis_scout",
        "scout_pass": passed,
        "headline": {"eval_rows": 1024},
        "packet_contract": {
            "raw_payload_bytes": 2,
            "framed_record_bytes": 5,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "raw_hidden_vector_transmitted": False,
            "raw_scores_transmitted": False,
            "basis_coefficients_transmitted": False,
        },
        "variant_rows": [variant],
    }


def test_multi_slice_gate_aggregates_failed_slices(tmp_path) -> None:
    paths = []
    for index, start in enumerate((0, 1024, 2048, 3072, 4096)):
        path = tmp_path / f"validation{start}_{start + 1024}.json"
        payload = _payload(
            accuracy=0.53 if index >= 3 else 0.45,
            label=0.50 if index >= 3 else 0.44,
            score=0.48 if index >= 3 else 0.43,
            passed=index >= 3,
        )
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)

    payload = gate.build_gate(
        output_dir=tmp_path / "out",
        slice_artifacts=tuple(paths),
    )

    assert payload["gate"] == "source_private_hellaswag_learned_residual_basis_multi_slice_stress"
    assert payload["pass_gate"] is False
    assert payload["headline"]["slice_count"] == 5
    assert payload["headline"]["pass_slice_count"] == 2
    assert payload["headline"]["source_private_packet"] is True
    assert (tmp_path / "out" / "hellaswag_learned_residual_basis_multi_slice_stress.json").exists()
    assert (tmp_path / "out" / "slice_rows.csv").exists()
