import pytest

from experimental.sinkaware.phase2.rank2_length_sink_sweep_gate import (
    _aggregate_configs,
    _config_summary,
    _status,
)


def test_config_summary_extracts_rank2_improvement() -> None:
    result = {
        "max_length": 64,
        "sink_tokens": 2,
        "seed_rows": [
            {
                "summary": {
                    "position": {"output_rel_l2": 0.20},
                    "rank2": {"output_rel_l2": 0.15},
                }
            },
            {
                "summary": {
                    "position": {"output_rel_l2": 0.18},
                    "rank2": {"output_rel_l2": 0.16},
                }
            },
        ],
        "aggregate": {
            "output_rel_l2_improvement_vs_position": {"mean": 0.035, "ci95": 0.02},
            "sink_mass_mae_improvement_vs_position": {"mean": 0.01, "ci95": 0.01},
            "attention_l1_improvement_vs_position": {"mean": 0.02, "ci95": 0.01},
            "output_rel_l2_head_win_rate": {"mean": 0.4, "ci95": 0.1},
            "all_seeds_rank2_beats_position": {"value": True, "n": 2},
        },
    }

    summary = _config_summary(result)

    assert summary["position_output_rel_l2_mean"] == pytest.approx(0.19)
    assert summary["rank2_output_rel_l2_mean"] == pytest.approx(0.155)
    assert summary["all_seeds_rank2_beats_position"] is True


def test_aggregate_configs_tracks_min_and_all_positive() -> None:
    rows = [
        {
            "output_rel_l2_improvement": {"mean": 0.03, "ci95": 0.01},
            "output_rel_l2_head_win_rate": {"mean": 0.3, "ci95": 0.01},
            "all_seeds_rank2_beats_position": True,
        },
        {
            "output_rel_l2_improvement": {"mean": 0.02, "ci95": 0.01},
            "output_rel_l2_head_win_rate": {"mean": 0.5, "ci95": 0.01},
            "all_seeds_rank2_beats_position": True,
        },
    ]

    aggregate = _aggregate_configs(rows)

    assert aggregate["min_output_rel_l2_improvement"] == pytest.approx(0.02)
    assert aggregate["configs_all_seeds_positive"]["value"] is True
    assert aggregate["output_rel_l2_head_win_rate_across_configs"]["mean"] == pytest.approx(0.4)


def test_status_marks_sweep_alive_when_all_configs_clear_margin() -> None:
    aggregate = {
        "configs_all_seeds_positive": {"value": True, "n": 2},
        "min_output_rel_l2_improvement": 0.02,
    }

    assert _status([], aggregate).startswith("ALIVE but bounded")
