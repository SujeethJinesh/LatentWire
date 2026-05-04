from __future__ import annotations

import math

import numpy as np
import torch

from scripts import build_target_self_resonance_hellaswag_consistency_refined_slot_gate as gate


def test_consistency_slot_refiner_shape_and_gate() -> None:
    torch.manual_seed(7)
    refiner = gate.ConsistencySlotRefiner(
        feature_dim=10,
        score_feature_dim=26,
        embed_dim=6,
        hidden_dim=8,
        prefix_len=3,
        initial_refine_gate=-8.0,
    )

    residual = refiner(torch.randn(10), torch.randn(26))

    assert residual.shape == (3, 6)
    assert torch.isfinite(residual).all()
    assert 0.0 < float(torch.sigmoid(refiner.refine_gate)) < 0.01


def test_consistency_slot_refiner_rejects_batched_inputs() -> None:
    refiner = gate.ConsistencySlotRefiner(
        feature_dim=4,
        score_feature_dim=26,
        embed_dim=3,
        hidden_dim=5,
        prefix_len=2,
        initial_refine_gate=-8.0,
    )

    try:
        refiner(torch.randn(2, 4), torch.randn(26))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for batched source features")

    try:
        refiner(torch.randn(4), torch.randn(2, 26))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for batched score features")


def test_score_features_include_current_and_source_state() -> None:
    target_scores = [0.1, 2.0, -0.4, 1.0]
    source_scores = [1.0, 0.3, 2.5, -0.2]

    features = gate._score_features(target_scores, source_scores)
    rolled = gate._score_features(target_scores, np.roll(source_scores, 1))

    assert features.shape == (26,)
    assert features.dtype == np.float32
    assert not np.allclose(features, rolled)
    assert math.isclose(float(features[4:8].sum()), 1.0)
    assert math.isclose(float(features[8:12].sum()), 1.0)
    assert math.isclose(float(features[17:21].sum()), 1.0)
    assert math.isclose(float(features[21:25].sum()), 1.0)
    assert -1.0 <= float(features[12]) <= 1.0
    assert -1.0 <= float(features[25]) <= 1.0


def _prediction_row(row_id: str, condition: str, prediction: int, answer: int, kl: float) -> dict:
    return {
        "row_id": row_id,
        "content_id": row_id,
        "condition": condition,
        "answer_index": answer,
        "answer_label": chr(ord("A") + answer),
        "prediction_index": prediction,
        "prediction_label": chr(ord("A") + prediction),
        "correct": prediction == answer,
        "full_prompt_prediction_index": prediction,
        "full_prompt_prediction_label": chr(ord("A") + prediction),
        "agrees_with_full_prompt": True,
        "margin": 0.2,
        "kl_to_full": kl,
        "scores": [float(prediction == index) for index in range(4)],
    }


def test_condition_metrics_cover_refinement_controls() -> None:
    rows = []
    full_predictions = [0, 1, 2, 3]
    answers = [0, 1, 0, 2]
    for index, (full_prediction, answer) in enumerate(zip(full_predictions, answers, strict=True)):
        row_id = f"r{index}"
        for condition in gate.CONDITIONS:
            prediction = full_prediction if condition in {"full_prompt", "source_consistency_refined_slots"} else 0
            rows.append(_prediction_row(row_id, condition, prediction, answer, 0.05))

    metrics = gate._condition_metrics(rows, seed=19, bootstrap_samples=25)

    assert metrics["source_consistency_refined_slots"]["agreement_with_full_prompt"] == 1.0
    assert metrics["zero_source_refine"]["paired_vs_full_prompt_accuracy"]["samples"] == 25
    assert metrics["refine_step_shuffle"]["n"] == 4
