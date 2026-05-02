from __future__ import annotations

from scripts import build_source_private_commonsenseqa_score_packet_headroom as headroom
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate


def _row(row_id: str, answer_index: int) -> arc_gate.ArcRow:
    return arc_gate.ArcRow(
        row_id=row_id,
        content_id=row_id,
        question=f"Question {row_id}?",
        choices=("alpha", "beta", "gamma"),
        choice_labels=("A", "B", "C"),
        answer_index=answer_index,
        answer_label=("A", "B", "C")[answer_index],
    )


def test_top2_threshold_can_beat_top_label_on_heldout_when_low_margin_errors_repeat() -> None:
    rows = [
        _row("cal_low_margin_error", 1),
        _row("heldout_low_margin_error", 1),
        _row("cal_high_margin_correct", 0),
        _row("heldout_high_margin_correct", 0),
    ]
    scores = [
        [0.51, 0.50, 0.00],
        [0.52, 0.51, 0.00],
        [1.00, 0.10, 0.00],
        [1.10, 0.10, 0.00],
    ]
    calibration = [0, 2]
    heldout = [1, 3]

    selection = headroom._select_threshold(rows, scores, calibration)
    threshold = selection["selected"]["threshold"]
    predictions = headroom._top2_predictions_with_threshold(scores, threshold)

    assert threshold > 0.01
    assert headroom._prediction_accuracy(rows, [0, 0, 0, 0], heldout) == 0.5
    assert headroom._prediction_accuracy(rows, predictions, heldout) == 1.0


def test_candidate_thresholds_include_switch_and_no_switch_options() -> None:
    thresholds = headroom._candidate_thresholds([0.1, 0.5, 0.2], [0, 1, 2])

    assert thresholds[0] == float("-inf")
    assert thresholds[-1] > 0.5
    assert any(0.1 < threshold < 0.2 for threshold in thresholds)


def test_rank_bin_decoder_can_select_non_top_rank_by_margin_bin() -> None:
    rows = [
        _row("cal_low_margin_second", 1),
        _row("heldout_low_margin_second", 1),
        _row("cal_high_margin_first", 0),
        _row("heldout_high_margin_first", 0),
    ]
    scores = [
        [0.51, 0.50, 0.00],
        [0.52, 0.51, 0.00],
        [1.00, 0.10, 0.00],
        [1.10, 0.10, 0.00],
    ]

    decoder = headroom._fit_rank_bin_decoder(rows, scores, [0, 2], bins=2, max_rank=2)
    predictions = headroom._rank_bin_predictions(scores, decoder)

    assert decoder["selected_rank_by_bin"][0] == 1
    assert decoder["selected_rank_by_bin"][-1] == 0
    assert headroom._prediction_accuracy(rows, predictions, [1, 3]) == 1.0
