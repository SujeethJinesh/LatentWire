from outlier_migrate.methods.decdec import select_decayed_average
from outlier_migrate.methods.m11b import select_union
from outlier_migrate.methods.static import select_static_topk


SCORES = {
    100: [10.0, 1.0, 0.0, 0.0],
    1000: [0.0, 9.0, 1.0, 0.0],
    5000: [0.0, 0.0, 8.0, 1.0],
    10000: [0.0, 0.0, 0.0, 7.0],
}


def test_static_topk_uses_requested_position() -> None:
    assert select_static_topk(SCORES, 1, position=1000) == {1}


def test_union_collects_per_position_top_channels() -> None:
    assert select_union(SCORES, 1, positions=(100, 1000, 5000, 10000)) == {0, 1, 2, 3}


def test_decayed_average_prefers_later_positions() -> None:
    assert select_decayed_average(SCORES, 1) == {3}
