from __future__ import annotations

import math

import numpy as np
import torch

from scripts import build_target_self_resonance_hellaswag_source_residual_slot_gate as gate


def test_source_residual_slot_encoder_shape_and_gate() -> None:
    torch.manual_seed(5)
    encoder = gate.SourceResidualSlotEncoder(feature_dim=10, embed_dim=6, hidden_dim=8, prefix_len=3)

    residual = encoder(torch.randn(10))

    assert residual.shape == (3, 6)
    assert torch.isfinite(residual).all()
    assert 0.0 < float(torch.sigmoid(encoder.residual_gate)) < 0.02


def test_source_residual_slot_encoder_rejects_batched_features() -> None:
    encoder = gate.SourceResidualSlotEncoder(feature_dim=4, embed_dim=3, hidden_dim=5, prefix_len=2)

    try:
        encoder(torch.randn(2, 4))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_top2_margin_features_are_compact_and_roll_sensitive() -> None:
    scores = np.asarray([0.2, 1.5, -0.3, 1.0], dtype=np.float64)

    features = gate._source_packet_features(scores, feature_mode="top2_margin")
    rolled = gate._source_packet_features(np.roll(scores, 1), feature_mode="top2_margin")

    assert features.shape == (10,)
    assert features.dtype == np.float32
    assert not np.allclose(features, rolled)
    assert math.isclose(float(features[:4].sum()), 1.0)
    assert math.isclose(float(features[4:8].sum()), 1.0)
    assert -1.0 <= float(features[8]) <= 1.0
    assert 0.0 <= float(features[9]) <= 1.0


def test_score_z_features_are_centered() -> None:
    features = gate._source_packet_features([1.0, 2.0, 3.0, 4.0], feature_mode="score_z")

    assert features.shape == (4,)
    assert abs(float(features.mean())) < 1e-6


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


def test_condition_metrics_cover_source_controls() -> None:
    rows = []
    full_predictions = [0, 1, 2, 3]
    answers = [0, 1, 0, 2]
    for index, (full_prediction, answer) in enumerate(zip(full_predictions, answers, strict=True)):
        row_id = f"r{index}"
        for condition in gate.CONDITIONS:
            prediction = full_prediction if condition in {"full_prompt", "source_residual_slots"} else 0
            rows.append(_prediction_row(row_id, condition, prediction, answer, 0.05))

    metrics = gate._condition_metrics(rows, seed=17, bootstrap_samples=25)

    assert metrics["source_residual_slots"]["agreement_with_full_prompt"] == 1.0
    assert metrics["wrong_source_residual"]["paired_vs_full_prompt_accuracy"]["samples"] == 25
