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
