from __future__ import annotations

import math

import torch

from scripts import build_target_self_resonance_hellaswag_query_resampler_gate as gate


def test_prompt_query_resampler_shape_and_attention() -> None:
    torch.manual_seed(3)
    encoder = gate.PromptQueryResampler(embed_dim=6, hidden_dim=4, prefix_len=3)
    prompt_embeds = torch.randn(7, 6)

    prefix, attention = encoder.forward_with_attention(prompt_embeds)

    assert prefix.shape == (3, 6)
    assert attention.shape == (3, 7)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(3), atol=1e-6)
    assert torch.isfinite(prefix).all()


def test_prompt_query_resampler_rejects_bad_input() -> None:
    encoder = gate.PromptQueryResampler(embed_dim=5, hidden_dim=3, prefix_len=2)

    for bad in (torch.randn(5), torch.empty(0, 5)):
        try:
            encoder(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def test_attention_summary_reports_normalized_entropy() -> None:
    attention = torch.tensor([[0.5, 0.5], [0.9, 0.1]], dtype=torch.float32)

    summary = gate._attention_summary(attention)

    assert 0.0 < summary["normalized_attention_entropy"] <= 1.0
    assert math.isclose(summary["mean_attention_max"], 0.7, rel_tol=1e-6)


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


def test_condition_metrics_include_query_controls() -> None:
    full_predictions = [0, 1, 2, 1]
    answers = [0, 1, 0, 2]
    rows = []
    for index, (full_prediction, answer) in enumerate(zip(full_predictions, answers, strict=True)):
        row_id = f"r{index}"
        rows.append(_prediction_row(row_id, "full_prompt", full_prediction, answer, 0.0))
        rows.append(_prediction_row(row_id, "chunk_mean_prefix", full_prediction, answer, 0.05))
        rows.append(_prediction_row(row_id, "query_resampler", full_prediction, answer, 0.01))
        rows.append(_prediction_row(row_id, "slots_only_query", 0, answer, 0.20))
        rows.append(_prediction_row(row_id, "zero_prefix", 0, answer, 0.30))
        rows.append(_prediction_row(row_id, "random_same_norm_prefix", 1, answer, 0.40))
        rows.append(_prediction_row(row_id, "shuffled_query_resampler", 2, answer, 0.50))
        rows.append(_prediction_row(row_id, "candidate_derangement", (full_prediction + 1) % 3, answer, 0.60))

    metrics = gate._condition_metrics(rows, seed=11, bootstrap_samples=25)

    assert metrics["full_prompt"]["accuracy"] == 0.5
    assert metrics["query_resampler"]["agreement_with_full_prompt"] == 1.0
    assert math.isclose(metrics["query_resampler"]["mean_kl_to_full"], 0.01)
    assert metrics["slots_only_query"]["paired_vs_full_prompt_accuracy"]["samples"] == 25
