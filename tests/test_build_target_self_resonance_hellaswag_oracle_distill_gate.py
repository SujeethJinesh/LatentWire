from __future__ import annotations

import math

import torch

from scripts import build_target_self_resonance_hellaswag_oracle_distill_gate as gate


def test_oracle_distill_encoder_preserves_shape() -> None:
    torch.manual_seed(9)
    encoder = gate.OracleDistillPrefixEncoder(embed_dim=6, hidden_dim=4, prefix_len=3)
    prefix = torch.randn(3, 6)

    out = encoder(prefix)

    assert out.shape == (3, 6)
    assert not torch.allclose(out, prefix)


def test_oracle_distill_encoder_rejects_wrong_prefix_len() -> None:
    encoder = gate.OracleDistillPrefixEncoder(embed_dim=5, hidden_dim=3, prefix_len=4)

    try:
        encoder(torch.randn(3, 5))
    except ValueError as exc:
        assert "wrong prefix length" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_prefix_distill_loss_is_scaled_mse() -> None:
    prefix = torch.ones(2, 3)
    oracle = torch.ones(2, 3)
    shifted = torch.ones(2, 3) * 2.0

    assert torch.allclose(gate._prefix_distill_loss(prefix, oracle, embed_rms=0.5), torch.zeros(()))
    assert math.isclose(float(gate._prefix_distill_loss(prefix, shifted, embed_rms=2.0)), 0.25)


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
        "scores": [float(prediction == index) for index in range(3)],
    }


def test_condition_metrics_for_oracle_distill_conditions() -> None:
    full_predictions = [0, 1, 2, 1]
    answers = [0, 1, 0, 2]
    rows = []
    for index, (full_prediction, answer) in enumerate(zip(full_predictions, answers, strict=True)):
        row_id = f"r{index}"
        rows.append(_prediction_row(row_id, "full_prompt", full_prediction, answer, 0.0))
        rows.append(_prediction_row(row_id, "chunk_mean_prefix", full_prediction, answer, 0.05))
        rows.append(_prediction_row(row_id, "oracle_distill_encoder", full_prediction, answer, 0.01))
        rows.append(_prediction_row(row_id, "slots_only_oracle_distill", 0, answer, 0.20))
        rows.append(_prediction_row(row_id, "zero_prefix", 0, answer, 0.30))
        rows.append(_prediction_row(row_id, "random_same_norm_prefix", 1, answer, 0.40))
        rows.append(_prediction_row(row_id, "shuffled_oracle_distill", 2, answer, 0.50))
        rows.append(_prediction_row(row_id, "candidate_derangement", (full_prediction + 1) % 3, answer, 0.60))

    metrics = gate._condition_metrics(rows, seed=11, bootstrap_samples=25)

    assert metrics["full_prompt"]["accuracy"] == 0.5
    assert metrics["oracle_distill_encoder"]["agreement_with_full_prompt"] == 1.0
    assert math.isclose(metrics["oracle_distill_encoder"]["mean_kl_to_full"], 0.01)
    assert metrics["slots_only_oracle_distill"]["paired_vs_full_prompt_accuracy"]["samples"] == 25
