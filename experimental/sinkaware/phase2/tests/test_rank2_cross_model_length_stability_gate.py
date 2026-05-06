import pytest

from experimental.sinkaware.phase2.rank2_cross_model_length_stability_gate import (
    _aggregate_lengths,
    _length_summary,
    _status,
)


def _model_result(model_name: str, family: str, improvement: float) -> dict[str, object]:
    position = 0.20
    rank2 = position - improvement
    return {
        "model_name": model_name,
        "model_family": family,
        "seed_rows": [
            {
                "summary": {
                    "position": {"output_rel_l2": position},
                    "rank2": {"output_rel_l2": rank2},
                }
            }
        ],
        "aggregate": {
            "output_rel_l2_improvement_vs_position": {"mean": improvement, "ci95": 0.0},
            "sink_mass_mae_improvement_vs_position": {"mean": 0.01, "ci95": 0.0},
            "attention_l1_improvement_vs_position": {"mean": 0.02, "ci95": 0.0},
            "output_rel_l2_head_win_rate": {"mean": 0.75, "ci95": 0.0},
            "all_seeds_rank2_beats_position": {"value": improvement > 0.0, "n": 1},
            "min_output_rel_l2_improvement": {"value": improvement},
        },
    }


def test_length_summary_extracts_model_length_rows() -> None:
    length_result = {
        "max_length": 96,
        "model_results": [
            _model_result("distilgpt2", "gpt2", 0.03),
            _model_result("facebook/opt-125m", "opt", 0.05),
        ],
    }

    summary = _length_summary(length_result)

    assert summary["max_length"] == 96
    assert summary["all_models_positive"] is True
    assert summary["min_model_output_rel_l2_improvement"] == pytest.approx(0.03)
    assert summary["output_rel_l2_improvement_across_models"]["mean"] == pytest.approx(0.04)
    assert summary["model_summaries"][0]["rank2_output_rel_l2_mean"] == pytest.approx(0.17)


def test_aggregate_lengths_tracks_min_and_seed_positivity() -> None:
    length_summaries = [
        {
            "model_summaries": [
                {
                    "output_rel_l2_improvement": {"mean": 0.03, "ci95": 0.0},
                    "output_rel_l2_head_win_rate": {"mean": 0.70, "ci95": 0.0},
                    "all_seeds_rank2_beats_position": True,
                },
                {
                    "output_rel_l2_improvement": {"mean": 0.04, "ci95": 0.0},
                    "output_rel_l2_head_win_rate": {"mean": 0.80, "ci95": 0.0},
                    "all_seeds_rank2_beats_position": True,
                },
            ]
        },
        {
            "model_summaries": [
                {
                    "output_rel_l2_improvement": {"mean": 0.02, "ci95": 0.0},
                    "output_rel_l2_head_win_rate": {"mean": 0.60, "ci95": 0.0},
                    "all_seeds_rank2_beats_position": True,
                },
            ]
        },
    ]

    aggregate = _aggregate_lengths(length_summaries)

    assert aggregate["all_model_lengths_positive"]["value"] is True
    assert aggregate["all_seeds_positive"]["value"] is True
    assert aggregate["min_model_length_output_rel_l2_improvement"]["value"] == pytest.approx(0.02)
    assert aggregate["output_rel_l2_head_win_rate_across_model_lengths"]["mean"] == pytest.approx(0.70)


def test_status_weakens_on_negative_model_length_row() -> None:
    aggregate = {
        "all_model_lengths_positive": {"value": False, "n": 2},
        "all_seeds_positive": {"value": False, "n": 2},
        "min_model_length_output_rel_l2_improvement": {"value": -0.01},
        "output_rel_l2_improvement_across_model_lengths": {"mean": 0.01, "ci95": 0.0},
    }

    assert _status(aggregate).startswith("WEAKENED")
