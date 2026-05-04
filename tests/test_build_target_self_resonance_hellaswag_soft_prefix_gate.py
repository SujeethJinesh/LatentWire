from __future__ import annotations

import math

import torch

from scripts import build_target_self_resonance_hellaswag_soft_prefix_gate as gate


def test_choice_scores_from_context_is_differentiable() -> None:
    class ToyTarget(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(5, 11, bias=False)

        def forward(self, *, inputs_embeds, attention_mask, use_cache=False):
            del attention_mask, use_cache
            return type("Output", (), {"logits": self.proj(inputs_embeds.cumsum(dim=1))})()

    torch.manual_seed(13)
    target = ToyTarget()
    embed_tokens = torch.nn.Embedding(11, 5)
    context = torch.randn(3, 5, requires_grad=True)
    continuations = [
        torch.tensor([4]),
        torch.tensor([5, 6]),
        torch.tensor([7, 8, 9]),
    ]

    scores = gate._choice_scores_from_context(
        target_model=target,
        embed_tokens=embed_tokens,
        context_embeds=context,
        continuation_ids=continuations,
        length_normalize=True,
    )

    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()
    scores.sum().backward()
    assert context.grad is not None
    assert float(context.grad.norm()) > 0.0


def test_kl_to_full_is_zero_for_identical_scores_and_positive_for_mismatch() -> None:
    full = [2.0, 0.5, -1.0]

    assert gate._kl_to_full(full, full) == 0.0
    assert gate._kl_to_full([-1.0, 0.5, 2.0], full) > 1.0


def test_prompt_chunk_mean_prefix_preserves_shape_and_rms() -> None:
    torch.manual_seed(3)
    embed_tokens = torch.nn.Embedding(17, 6)
    prefix = gate._prompt_chunk_mean_prefix(
        prompt_ids=torch.tensor([1, 2, 3, 4, 5, 6, 7]),
        embed_tokens=embed_tokens,
        prefix_len=4,
        embed_rms=0.75,
        device="cpu",
    )

    assert prefix.shape == (4, 6)
    assert torch.allclose(prefix.float().pow(2).mean(dim=1).sqrt(), torch.full((4,), 0.75), atol=1e-5)


def test_random_same_norm_prefix_preserves_row_norms() -> None:
    reference = torch.tensor([[3.0, 4.0], [0.0, 2.0]], dtype=torch.float32)

    noise = gate._random_same_norm_prefix(reference=reference, seed=5)

    assert noise.shape == reference.shape
    assert torch.allclose(noise.norm(dim=1), reference.norm(dim=1), atol=1e-6)
    assert not torch.allclose(noise, reference)


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
        "margin": 0.1 * (prediction + 1),
        "kl_to_full": kl,
        "scores": [float(prediction == index) for index in range(3)],
    }


def test_condition_metrics_tracks_accuracy_agreement_and_kl() -> None:
    full_predictions = [0, 1, 2]
    answers = [0, 1, 0]
    rows = []
    for row_index, (full_prediction, answer) in enumerate(zip(full_predictions, answers, strict=True)):
        row_id = f"r{row_index}"
        rows.append(_prediction_row(row_id, "full_prompt", full_prediction, answer, 0.0))
        rows.append(_prediction_row(row_id, "optimized_soft_prefix", full_prediction, answer, 0.01))
        rows.append(_prediction_row(row_id, "chunk_mean_prefix", full_prediction, answer, 0.05))
        rows.append(_prediction_row(row_id, "zero_prefix", 1, answer, 0.20))
        rows.append(_prediction_row(row_id, "random_same_norm_prefix", 2, answer, 0.30))
        rows.append(_prediction_row(row_id, "shuffled_optimized_prefix", 0, answer, 0.40))
        rows.append(_prediction_row(row_id, "candidate_derangement", (full_prediction + 1) % 3, answer, 0.50))

    metrics = gate._condition_metrics(rows, seed=7, bootstrap_samples=25)

    assert metrics["full_prompt"]["accuracy"] == 2 / 3
    assert metrics["optimized_soft_prefix"]["agreement_with_full_prompt"] == 1.0
    assert math.isclose(metrics["optimized_soft_prefix"]["mean_kl_to_full"], 0.01)
    assert metrics["zero_prefix"]["paired_vs_full_prompt_accuracy"]["samples"] == 25
