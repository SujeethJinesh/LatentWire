import pytest
import torch

from experimental.sinkaware.phase2.rank2_cross_model_falsification_gate import (
    _aggregate_models,
    _aggregate_seed_rows,
    _model_status,
    _split_heads,
    _status,
)


def test_split_heads_returns_head_major_projection() -> None:
    projected = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)

    heads = _split_heads(projected, n_heads=2)

    assert heads.shape == (2, 3, 4)
    assert torch.equal(heads[0, 0], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.equal(heads[1, 2], torch.tensor([20.0, 21.0, 22.0, 23.0]))


def test_aggregate_seed_rows_tracks_cross_model_metrics() -> None:
    rows = [
        {
            "output_rel_l2_improvement_vs_position": 0.04,
            "sink_mass_mae_improvement_vs_position": 0.02,
            "attention_l1_improvement_vs_position": 0.03,
            "paired_head_vs_position": {"rank2": {"output_rel_l2_win_rate": 0.25}},
        },
        {
            "output_rel_l2_improvement_vs_position": 0.02,
            "sink_mass_mae_improvement_vs_position": 0.01,
            "attention_l1_improvement_vs_position": 0.02,
            "paired_head_vs_position": {"rank2": {"output_rel_l2_win_rate": 0.50}},
        },
    ]

    aggregate = _aggregate_seed_rows(rows)

    assert aggregate["output_rel_l2_improvement_vs_position"]["mean"] == pytest.approx(0.03)
    assert aggregate["sink_mass_mae_improvement_vs_position"]["mean"] == pytest.approx(0.015)
    assert aggregate["output_rel_l2_head_win_rate"]["mean"] == pytest.approx(0.375)
    assert aggregate["all_seeds_rank2_beats_position"]["value"] is True
    assert aggregate["min_output_rel_l2_improvement"]["value"] == pytest.approx(0.02)


def test_status_weakens_when_any_model_is_negative() -> None:
    model_results = [
        {"aggregate": {"output_rel_l2_improvement_vs_position": {"mean": 0.03}}},
        {"aggregate": {"output_rel_l2_improvement_vs_position": {"mean": -0.01}}},
    ]

    aggregate = _aggregate_models(model_results)

    assert aggregate["all_models_positive"]["value"] is False
    assert _status(aggregate).startswith("WEAKENED")


def test_model_status_requires_promotion_margin() -> None:
    aggregate = {
        "output_rel_l2_improvement_vs_position": {"mean": 0.01, "ci95": 0.0},
        "all_seeds_rank2_beats_position": {"value": True, "n": 1},
        "min_output_rel_l2_improvement": {"value": 0.01},
    }

    assert _model_status(aggregate).startswith("WEAKLY ALIVE")
