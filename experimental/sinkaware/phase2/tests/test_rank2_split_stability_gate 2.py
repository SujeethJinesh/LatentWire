import pytest

from experimental.sinkaware.phase2.rank2_split_stability_gate import (
    _aggregate_seed_rows,
    _split_indices,
    _status,
)


def test_split_indices_are_seeded_and_disjoint() -> None:
    train_a, test_a = _split_indices(10, 0.6, seed=7)
    train_b, test_b = _split_indices(10, 0.6, seed=7)
    train_c, test_c = _split_indices(10, 0.6, seed=8)

    assert train_a.tolist() == train_b.tolist()
    assert test_a.tolist() == test_b.tolist()
    assert train_a.tolist() != train_c.tolist()
    assert set(train_a.tolist()).isdisjoint(set(test_a.tolist()))
    assert len(train_a) == 6
    assert len(test_a) == 4


def test_aggregate_seed_rows_reports_mean_ci_and_all_positive() -> None:
    rows = [
        {
            "output_rel_l2_improvement_vs_position": 0.03,
            "sink_mass_mae_improvement_vs_position": 0.01,
            "attention_l1_improvement_vs_position": 0.02,
            "paired_head_vs_position": {"rank2": {"output_rel_l2_win_rate": 0.4}},
        },
        {
            "output_rel_l2_improvement_vs_position": 0.01,
            "sink_mass_mae_improvement_vs_position": 0.03,
            "attention_l1_improvement_vs_position": 0.04,
            "paired_head_vs_position": {"rank2": {"output_rel_l2_win_rate": 0.6}},
        },
    ]

    aggregate = _aggregate_seed_rows(rows)

    assert aggregate["output_rel_l2_improvement_vs_position"]["mean"] == pytest.approx(0.02)
    assert aggregate["sink_mass_mae_improvement_vs_position"]["mean"] == pytest.approx(0.02)
    assert aggregate["output_rel_l2_head_win_rate"]["mean"] == pytest.approx(0.5)
    assert aggregate["all_seeds_rank2_beats_position"]["value"] is True


def test_status_marks_positive_repeats_alive_but_weak() -> None:
    aggregate = {
        "output_rel_l2_improvement_vs_position": {"mean": 0.02, "ci95": 0.01},
        "all_seeds_rank2_beats_position": {"value": True, "n": 2},
    }

    assert _status([], aggregate).startswith("ALIVE but still weak")
