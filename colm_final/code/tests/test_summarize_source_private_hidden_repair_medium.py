from __future__ import annotations

from scripts.summarize_source_private_hidden_repair_medium import _bootstrap_delta_ci, _percentile


def test_percentile_uses_nearest_rank_over_sorted_values() -> None:
    assert _percentile([3.0, 1.0, 2.0], 0.0) == 1.0
    assert _percentile([3.0, 1.0, 2.0], 0.5) == 2.0
    assert _percentile([3.0, 1.0, 2.0], 1.0) == 3.0


def test_bootstrap_delta_ci_is_paired_over_rows() -> None:
    rows = [
        {"conditions": {"matched": {"correct": True}, "target": {"correct": False}}},
        {"conditions": {"matched": {"correct": True}, "target": {"correct": False}}},
        {"conditions": {"matched": {"correct": False}, "target": {"correct": False}}},
        {"conditions": {"matched": {"correct": True}, "target": {"correct": True}}},
    ]

    ci = _bootstrap_delta_ci(rows, left="matched", right="target", samples=200, seed=7)

    assert 0.0 <= ci["low"] <= ci["mean"] <= ci["high"] <= 1.0
    assert ci["mean"] > 0.25
