import pytest

from experimental.sinkaware.phase2.rank2_trace_frozen_split_gate import (
    _aggregate_seed_rows,
    _split_trace_indices,
    _status,
)


def test_split_trace_indices_are_seeded_disjoint_and_cover_traces() -> None:
    train_a, test_a = _split_trace_indices(12, 0.67, seed=3)
    train_b, test_b = _split_trace_indices(12, 0.67, seed=3)
    train_c, test_c = _split_trace_indices(12, 0.67, seed=4)

    assert train_a == train_b
    assert test_a == test_b
    assert train_a != train_c
    assert set(train_a).isdisjoint(set(test_a))
    assert sorted(train_a + test_a) == list(range(12))
    assert len(train_a) == 8
    assert len(test_a) == 4


def test_aggregate_seed_rows_tracks_trace_split_repeat_metrics() -> None:
    rows = [
        {
            "output_rel_l2_improvement_vs_position": 0.04,
            "sink_mass_mae_improvement_vs_position": 0.01,
            "attention_l1_improvement_vs_position": 0.02,
            "paired_head_vs_position": {"rank2": {"output_rel_l2_win_rate": 0.30}},
        },
        {
            "output_rel_l2_improvement_vs_position": 0.02,
            "sink_mass_mae_improvement_vs_position": 0.03,
            "attention_l1_improvement_vs_position": 0.04,
            "paired_head_vs_position": {"rank2": {"output_rel_l2_win_rate": 0.50}},
        },
    ]

    aggregate = _aggregate_seed_rows(rows)

    assert aggregate["output_rel_l2_improvement_vs_position"]["mean"] == pytest.approx(0.03)
    assert aggregate["sink_mass_mae_improvement_vs_position"]["mean"] == pytest.approx(0.02)
    assert aggregate["output_rel_l2_head_win_rate"]["mean"] == pytest.approx(0.40)
    assert aggregate["all_trace_splits_rank2_beats_position"]["value"] is True
    assert aggregate["min_output_rel_l2_improvement"]["value"] == pytest.approx(0.02)


def test_status_marks_trace_splits_alive_when_all_clear_margin() -> None:
    aggregate = {
        "output_rel_l2_improvement_vs_position": {"mean": 0.03, "ci95": 0.01},
        "all_trace_splits_rank2_beats_position": {"value": True, "n": 2},
        "min_output_rel_l2_improvement": {"value": 0.02},
    }

    assert _status([], aggregate).startswith("ALIVE but bounded")
