from __future__ import annotations

import numpy as np

from scripts import build_target_self_resonance_hellaswag_source_codebook_candidate_repair_gate as gate


def test_code_parts_are_roll_sensitive() -> None:
    scores = [0.1, 1.2, -0.4, 0.9]
    bins = [0.2, 0.5, 1.0]

    parts = gate._code_parts(scores, margin_bins=bins, entropy_bins=bins)
    rolled = gate._code_parts(np.roll(np.asarray(scores), 1), margin_bins=bins, entropy_bins=bins)

    assert parts != rolled
    assert gate._code_key(parts, mode="top1") == (1,)
    assert len(gate._code_key(parts, mode="top1_top2_margin_entropy")) == 4


def test_fit_codebook_repair_can_prefer_source_code() -> None:
    records = [
        {
            "answer_index": 1,
            "source_scores": [0.0, 2.0, 1.0, -1.0],
            "full_scores": [0.0, 3.0, 0.5, -0.5],
            "frozen_target_slots_scores": [1.0, 0.0, 0.2, -0.5],
        },
        {
            "answer_index": 2,
            "source_scores": [0.0, 1.0, 2.0, -1.0],
            "full_scores": [0.0, 0.5, 3.0, -0.5],
            "frozen_target_slots_scores": [1.0, 0.0, 0.2, -0.5],
        },
        {
            "answer_index": 1,
            "source_scores": [0.0, 2.1, 0.9, -1.0],
            "full_scores": [0.0, 3.0, 0.5, -0.5],
            "frozen_target_slots_scores": [1.0, 0.0, 0.2, -0.5],
        },
    ]
    codebook = gate._fit_codebooks(
        records,
        margin_bins=[0.5, 1.5],
        entropy_bins=[0.5, 0.9],
        mode="top1",
        laplace=0.1,
    )

    repaired, key = gate._repair_scores(
        [1.0, 0.0, 0.2, -0.5],
        [0.0, 2.0, 1.0, -1.0],
        codebook=codebook,
        prior_weight=2.0,
        delta_weight=1.0,
    )
    zero, zero_key = gate._repair_scores(
        [1.0, 0.0, 0.2, -0.5],
        None,
        codebook=codebook,
        prior_weight=2.0,
        delta_weight=1.0,
    )

    assert key == "1"
    assert zero_key == "__global__"
    assert gate._prediction(repaired) == 1
    assert len(zero) == 4


def _prediction_row(condition: str, prediction: int, answer: int) -> dict:
    return {
        "row_id": "r0",
        "content_id": "c0",
        "condition": condition,
        "answer_index": answer,
        "answer_label": chr(ord("A") + answer),
        "prediction_index": prediction,
        "prediction_label": chr(ord("A") + prediction),
        "correct": prediction == answer,
        "full_prompt_prediction_index": prediction,
        "full_prompt_prediction_label": chr(ord("A") + prediction),
        "agrees_with_full_prompt": prediction == answer,
        "margin": 0.1,
        "kl_to_full": 0.05,
        "kl_was_nonfinite": False,
        "nonfinite_score": False,
        "codebook_key": "",
        "scores": [float(prediction == index) for index in range(4)],
    }


def test_condition_metrics_include_codebook_controls() -> None:
    rows = []
    for condition in gate.CONDITIONS:
        rows.append(_prediction_row(condition, 2 if condition == "source_codebook_repair" else 1, 2))

    metrics = gate._condition_metrics(rows, seed=13, bootstrap_samples=20)

    assert metrics["source_codebook_repair"]["accuracy"] == 1.0
    assert "paired_vs_full_prompt_accuracy" in metrics["wrong_source_codebook"]
