from __future__ import annotations

import math

import torch

from scripts import build_target_self_resonance_hellaswag_chunk_encoder_gate as gate


def test_chunk_prefix_encoder_preserves_prefix_shape() -> None:
    torch.manual_seed(7)
    encoder = gate.ChunkPrefixEncoder(embed_dim=5, hidden_dim=3, prefix_len=4)
    chunk = torch.randn(4, 5)

    out = encoder(chunk)

    assert out.shape == (4, 5)
    assert not torch.allclose(out, chunk)


def test_slots_only_encoder_ignores_input() -> None:
    initial = torch.randn(3, 4)
    encoder = gate.SlotsOnlyEncoder(initial_prefix=initial)

    left = encoder(torch.randn(3, 4))
    right = encoder(torch.randn(3, 4))

    assert torch.allclose(left, right)
    assert torch.allclose(left, initial)


def test_prefix_rms_loss_is_zero_at_target_rms() -> None:
    prefix = torch.ones(2, 4) * 0.5

    assert torch.allclose(gate._prefix_rms_loss(prefix, embed_rms=0.5), torch.zeros(()))
    assert float(gate._prefix_rms_loss(prefix, embed_rms=1.0)) > 0.0


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


def test_condition_metrics_include_full_agreement_and_pair_ci() -> None:
    full_predictions = [0, 1, 2, 1]
    answers = [0, 1, 0, 2]
    rows = []
    for index, (full_prediction, answer) in enumerate(zip(full_predictions, answers, strict=True)):
        row_id = f"r{index}"
        rows.append(_prediction_row(row_id, "full_prompt", full_prediction, answer, 0.0))
        rows.append(_prediction_row(row_id, "chunk_mean_prefix", full_prediction, answer, 0.05))
        rows.append(_prediction_row(row_id, "learned_chunk_encoder", full_prediction, answer, 0.01))
        rows.append(_prediction_row(row_id, "slots_only_encoder", 0, answer, 0.20))
        rows.append(_prediction_row(row_id, "zero_prefix", 0, answer, 0.30))
        rows.append(_prediction_row(row_id, "random_same_norm_prefix", 1, answer, 0.40))
        rows.append(_prediction_row(row_id, "shuffled_chunk_encoder", 2, answer, 0.50))
        rows.append(_prediction_row(row_id, "candidate_derangement", (full_prediction + 1) % 3, answer, 0.60))

    metrics = gate._condition_metrics(rows, seed=11, bootstrap_samples=25)

    assert metrics["full_prompt"]["accuracy"] == 0.5
    assert metrics["learned_chunk_encoder"]["agreement_with_full_prompt"] == 1.0
    assert math.isclose(metrics["learned_chunk_encoder"]["mean_kl_to_full"], 0.01)
    assert metrics["slots_only_encoder"]["paired_vs_full_prompt_accuracy"]["samples"] == 25


def test_condition_metrics_counts_nonfinite_rows() -> None:
    rows = []
    for condition in gate.CONDITIONS:
        row = _prediction_row("r0", condition, 0, 0, float("nan"))
        row["kl_was_nonfinite"] = condition == "learned_chunk_encoder"
        row["nonfinite_score"] = condition == "learned_chunk_encoder"
        rows.append(row)

    metrics = gate._condition_metrics(rows, seed=13, bootstrap_samples=10)

    assert metrics["learned_chunk_encoder"]["nonfinite_kl_count"] == 1
    assert metrics["learned_chunk_encoder"]["nonfinite_score_row_count"] == 1
    assert metrics["learned_chunk_encoder"]["mean_kl_to_full"] == 0.0
