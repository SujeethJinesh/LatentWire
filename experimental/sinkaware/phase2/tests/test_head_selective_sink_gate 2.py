import pytest

from experimental.sinkaware.phase2.head_selective_sink_gate import _mixed_summary, _select_heads


def test_select_heads_uses_validation_improvement():
    validation_heads = {
        0: {
            0: {
                "position": {"output_rel_l2": 0.3},
                "rank2": {"output_rel_l2": 0.2},
            },
            1: {
                "position": {"output_rel_l2": 0.1},
                "rank2": {"output_rel_l2": 0.2},
            },
        }
    }
    assert _select_heads(validation_heads) == {0: {0}}


def test_mixed_summary_uses_selected_rank2_otherwise_position():
    test_heads = {
        0: {
            0: {
                "position": {
                    "sink_logit_rmse": 0.5,
                    "sink_mass_mae": 0.5,
                    "attention_l1": 0.5,
                    "output_rel_l2": 0.5,
                },
                "rank2": {
                    "sink_logit_rmse": 0.2,
                    "sink_mass_mae": 0.2,
                    "attention_l1": 0.2,
                    "output_rel_l2": 0.2,
                },
            },
            1: {
                "position": {
                    "sink_logit_rmse": 0.1,
                    "sink_mass_mae": 0.1,
                    "attention_l1": 0.1,
                    "output_rel_l2": 0.1,
                },
                "rank2": {
                    "sink_logit_rmse": 0.9,
                    "sink_mass_mae": 0.9,
                    "attention_l1": 0.9,
                    "output_rel_l2": 0.9,
                },
            },
        }
    }
    summary = _mixed_summary(test_heads, {0: {0}})
    assert summary["output_rel_l2"] == pytest.approx(0.15)
    assert summary["sink_mass_mae"] == pytest.approx(0.15)
