from __future__ import annotations

from scripts import build_source_private_commonsenseqa_score_fusion_packet_probe as fusion
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


def test_quantized_fusion_can_use_scores_beyond_top_labels() -> None:
    rows = [
        _row("cal", 1),
        _row("heldout", 1),
    ]
    source_scores = fusion._quantized_source_rows(
        [
            [1.00, 0.95, 0.00],
            [1.00, 0.95, 0.00],
        ]
    )
    receiver_scores = fusion._zscore_rows(
        [
            [0.00, 0.95, 1.00],
            [0.00, 0.95, 1.00],
        ]
    )

    selection = fusion._select_fusion_weight(rows, source_scores, receiver_scores, [0])
    predictions = fusion._fusion_predictions(
        source_scores,
        receiver_scores,
        source_weight=selection["selected"]["source_weight"],
    )

    assert fusion._top_predictions(source_scores) == [0, 0]
    assert fusion._top_predictions(receiver_scores) == [2, 2]
    assert fusion._accuracy(rows, predictions, [1]) == 1.0


def test_label_pair_rule_reports_best_top_label_baseline() -> None:
    rows = [
        _row("cal_source", 0),
        _row("heldout_source", 0),
        _row("cal_receiver", 1),
        _row("heldout_receiver", 1),
    ]
    source_predictions = [0, 0, 2, 2]
    receiver_predictions = [2, 2, 1, 1]

    rule = fusion._select_label_pair_rule(rows, source_predictions, receiver_predictions, [0, 2])

    assert rule["selected"]["rule"] in {"always_source", "always_receiver"}
    assert rule["selected"]["calibration_accuracy"] == 0.5
