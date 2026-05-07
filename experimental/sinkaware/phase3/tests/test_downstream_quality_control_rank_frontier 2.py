import json
from pathlib import Path

import pytest

from experimental.sinkaware.phase3.downstream_quality_control_gate import _patch_modes


def test_patch_modes_include_requested_rank_frontier() -> None:
    assert _patch_modes((1, 2, 4, 8)) == (
        "exact",
        "position",
        "rank1",
        "rank2",
        "rank4",
        "rank8",
    )


def test_saved_48_trace_rank_frontier_artifact_is_integrated() -> None:
    artifact = (
        Path(__file__).resolve().parents[1]
        / "downstream_rank_frontier_traces48_len96_sink4.json"
    )
    result = json.loads(artifact.read_text(encoding="utf-8"))

    assert result["max_traces"] == 48
    assert result["max_length"] == 96
    assert result["sink_tokens"] == 4
    assert result["seeds"] == [0, 1, 2]
    assert result["ranks"] == [1, 2, 4, 8]

    aggregate = result["aggregate"]
    assert aggregate["all_models_exact_noop_ok"]["value"] is True
    assert aggregate["all_models_rank2_closer_by_loss"]["value"] is True
    assert aggregate["all_models_rank2_closer_by_kl"]["value"] is True

    frontier = aggregate["rank_frontier"]
    modes = ("rank1", "rank2", "rank4", "rank8")
    abs_loss = [
        frontier[mode]["abs_loss_delta_vs_baseline_across_models"]["mean"]
        for mode in modes
    ]
    top1 = [
        frontier[mode]["top1_disagreement_rate_across_models"]["mean"]
        for mode in modes
    ]
    loss_improvement = [
        frontier[mode]["loss_improvement_vs_position_across_models"]["mean"]
        for mode in modes
    ]

    assert all(left > right for left, right in zip(abs_loss, abs_loss[1:]))
    assert all(left > right for left, right in zip(top1, top1[1:]))
    assert all(left < right for left, right in zip(loss_improvement, loss_improvement[1:]))
    assert abs_loss == pytest.approx([0.137439, 0.096183, 0.061600, 0.043640], abs=1e-6)
