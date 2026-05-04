from __future__ import annotations

import math

import numpy as np
import torch

from scripts import build_target_self_resonance_hellaswag_source_oracle_distill_gate as gate


def test_source_oracle_prefix_encoder_shape_and_gate() -> None:
    torch.manual_seed(11)
    base = torch.randn(3, 6)
    encoder = gate.SourceOraclePrefixEncoder(
        feature_dim=5,
        embed_dim=6,
        hidden_dim=7,
        prefix_len=3,
        base_prefix=base,
        initial_residual_gate=-2.0,
    )

    out = encoder(torch.randn(5))

    assert out.shape == (3, 6)
    assert torch.isfinite(out).all()
    assert 0.0 < float(torch.sigmoid(encoder.residual_gate)) < 0.2
    assert not torch.allclose(out.cpu(), base)


def test_source_oracle_prefix_encoder_rejects_batched_codes() -> None:
    encoder = gate.SourceOraclePrefixEncoder(
        feature_dim=4,
        embed_dim=3,
        hidden_dim=5,
        prefix_len=2,
        base_prefix=torch.zeros(2, 3),
    )

    try:
        encoder(torch.randn(2, 4))
    except ValueError as exc:
        assert "1D" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_feature_projection_shapes_and_standardizes_train_codes() -> None:
    features = np.asarray(
        [
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 3.0],
        ],
        dtype=np.float32,
    )

    projection = gate._fit_feature_projection(features, code_dim=2)
    codes = np.stack([gate._project_feature(row, projection) for row in features], axis=0)

    assert projection.code_dim == 2
    assert codes.shape == (4, 2)
    assert np.isfinite(codes).all()
    assert np.allclose(codes.mean(axis=0), np.zeros(2), atol=1e-5)


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


def test_condition_metrics_cover_source_oracle_controls() -> None:
    rows = []
    full_predictions = [0, 1, 2, 3]
    answers = [0, 1, 0, 2]
    for index, (full_prediction, answer) in enumerate(zip(full_predictions, answers, strict=True)):
        row_id = f"r{index}"
        for condition in gate.CONDITIONS:
            prediction = full_prediction if condition in {"full_prompt", "source_oracle_distill_prefix"} else 0
            rows.append(_prediction_row(row_id, condition, prediction, answer, 0.05))

    metrics = gate._condition_metrics(rows, seed=17, bootstrap_samples=25)

    assert metrics["source_oracle_distill_prefix"]["agreement_with_full_prompt"] == 1.0
    assert metrics["wrong_source_code"]["paired_vs_full_prompt_accuracy"]["samples"] == 25
    assert math.isclose(metrics["source_oracle_distill_prefix"]["mean_kl_to_full"], 0.05)
