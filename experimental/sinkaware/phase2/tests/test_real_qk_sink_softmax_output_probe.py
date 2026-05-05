import torch
import pytest

from experimental.sinkaware.phase2.real_qk_sink_softmax_output_probe import (
    _attention_error_metrics,
    _attention_error_metrics_by_head,
    _fit_layer_predictors,
    _paired_head_improvements,
    _predict_sink_logits,
)


def test_rank_predictor_recovers_per_head_sink_logits() -> None:
    q = []
    pos = []
    y = []
    for idx in range(96):
        q0 = torch.tensor([float(idx % 7), float(idx % 5), 1.0])
        q1 = torch.tensor([float(idx % 3), float(idx % 11), -1.0])
        q_heads = torch.stack([q0, q1])
        p = torch.tensor([idx / 95.0])
        sink = torch.stack(
            [
                torch.tensor([0.4 * q0[0] - 0.2 * q0[1] + p[0], -0.1 * q0[0] + 0.3 * q0[1]]),
                torch.tensor([0.2 * q1[0] + 0.5 * q1[1], -0.4 * q1[0] + 0.1 * q1[1] - p[0]]),
            ]
        )
        q.append(q_heads)
        pos.append(p)
        y.append(sink)

    models = _fit_layer_predictors(torch.stack(q), torch.stack(pos), torch.stack(y), ranks=(2,))
    pred = _predict_sink_logits(models, torch.stack([torch.tensor([4.0, 2.0, 1.0]), torch.tensor([1.0, 5.0, -1.0])]), torch.tensor([0.5]), "rank2")
    expected = torch.tensor([[1.7, 0.2], [2.7, -0.4]])

    assert torch.allclose(pred, expected, atol=0.15)


def test_attention_error_metrics_are_zero_for_exact_sink_logits() -> None:
    exact_logits = torch.tensor([[1.0, 0.5, -0.5], [0.2, -0.1, 0.7]])
    values = torch.randn(2, 3, 4)

    metrics = _attention_error_metrics(exact_logits, exact_logits[:, :2], values, sink_tokens=2)

    assert metrics["sink_logit_rmse"] == 0.0
    assert metrics["sink_mass_mae"] == 0.0
    assert metrics["attention_l1"] == 0.0
    assert metrics["output_rel_l2"] == 0.0

    per_head = _attention_error_metrics_by_head(exact_logits, exact_logits[:, :2], values, sink_tokens=2)
    assert len(per_head) == 2
    assert all(row["sink_logit_rmse"] == 0.0 for row in per_head)
    assert all(row["sink_mass_mae"] == 0.0 for row in per_head)
    assert all(row["attention_l1"] == 0.0 for row in per_head)
    assert all(row["output_rel_l2"] == 0.0 for row in per_head)


def test_paired_head_improvements_report_win_rate() -> None:
    head_rows = {
        0: {
            0: {
                "position": {
                    "sink_logit_rmse": 3.0,
                    "sink_mass_mae": 0.3,
                    "attention_l1": 0.5,
                    "output_rel_l2": 0.4,
                },
                "rank2": {
                    "sink_logit_rmse": 2.0,
                    "sink_mass_mae": 0.2,
                    "attention_l1": 0.4,
                    "output_rel_l2": 0.2,
                },
            },
            1: {
                "position": {
                    "sink_logit_rmse": 1.0,
                    "sink_mass_mae": 0.1,
                    "attention_l1": 0.2,
                    "output_rel_l2": 0.1,
                },
                "rank2": {
                    "sink_logit_rmse": 1.5,
                    "sink_mass_mae": 0.2,
                    "attention_l1": 0.3,
                    "output_rel_l2": 0.2,
                },
            },
        }
    }

    paired = _paired_head_improvements(head_rows)

    assert paired["rank2"]["n_layer_heads"] == 2
    assert paired["rank2"]["output_rel_l2_improvement_mean"] == pytest.approx(0.05)
    assert paired["rank2"]["output_rel_l2_win_rate"] == 0.5
