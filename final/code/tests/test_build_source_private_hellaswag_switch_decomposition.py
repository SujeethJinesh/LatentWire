from __future__ import annotations

from scripts import build_source_private_hellaswag_switch_decomposition as switch
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate


def _row(row_id: str, answer_index: int) -> arc_gate.ArcRow:
    return arc_gate.ArcRow(
        row_id=row_id,
        content_id=f"content-{row_id}",
        question=f"question {row_id}",
        choices=("a", "b", "c", "d"),
        choice_labels=("A", "B", "C", "D"),
        answer_index=answer_index,
        answer_label=("A", "B", "C", "D")[answer_index],
    )


def test_top2_oracle_predictions_keep_top1_or_gold_runner_up() -> None:
    rows = [_row("0", 0), _row("1", 1), _row("2", 3)]
    scores = [
        [4.0, 3.0, 2.0, 1.0],
        [4.0, 3.0, 2.0, 1.0],
        [4.0, 3.0, 2.0, 1.0],
    ]

    assert switch._top2_oracle_predictions(rows, scores) == [0, 1, 0]


def test_switch_metrics_report_precision_recall_and_false_switches() -> None:
    rows = [_row("0", 0), _row("1", 1), _row("2", 0), _row("3", 2)]
    scores = [
        [4.0, 3.0, 2.0, 1.0],
        [4.0, 3.0, 2.0, 1.0],
        [4.0, 3.0, 2.0, 1.0],
        [4.0, 3.0, 2.0, 1.0],
    ]
    predictions = [0, 1, 1, 1]

    metrics = switch._switch_metrics(rows=rows, scores=scores, predictions=predictions)

    assert metrics["accuracy"] == 0.5
    assert metrics["switch_count"] == 3
    assert metrics["good_switch_count"] == 1
    assert metrics["false_switch_from_gold_top1_count"] == 1
    assert metrics["missed_switch_count"] == 0
    assert metrics["net_switch_gain_count"] == 0
    assert metrics["switch_precision"] == 1 / 3
    assert metrics["switch_recall_over_gold_top2"] == 1.0
    assert metrics["false_switch_away_from_gold_top1_rate"] == 0.5
    assert metrics["outside_top2_gold_count"] == 1
    assert metrics["headroom_capture_vs_source_top1"] == 0.0


def test_random_switch_predictions_match_requested_rate() -> None:
    scores = [
        [4.0, 3.0, 2.0, 1.0],
        [4.0, 3.0, 2.0, 1.0],
        [4.0, 3.0, 2.0, 1.0],
        [4.0, 3.0, 2.0, 1.0],
    ]

    predictions = switch._random_switch_predictions(scores, switch_count=2, seed=123)

    assert len(predictions) == 4
    assert sum(pred == 1 for pred in predictions) == 2
    assert set(predictions) <= {0, 1}


def test_candidate_row_adds_headroom_and_score_control_fields() -> None:
    rows = [_row("0", 0), _row("1", 1), _row("2", 0), _row("3", 2)]
    scores = [
        [4.0, 3.0, 2.0, 1.0],
        [4.0, 3.0, 2.0, 1.0],
        [4.0, 3.0, 2.0, 1.0],
        [4.0, 3.0, 2.0, 1.0],
    ]

    row = switch._candidate_row(
        eval_name="toy",
        candidate_name="toy_switch",
        rows=rows,
        scores=scores,
        predictions=[0, 1, 1, 1],
        best_label_predictions=[0, 0, 0, 0],
        score_only_predictions=[0, 0, 0, 0],
        bootstrap_seed=5,
        bootstrap_samples=20,
    )

    assert row["eval_name"] == "toy"
    assert row["candidate"] == "toy_switch"
    assert row["minus_best_label_copy"] == 0.0
    assert row["minus_score_only_switch"] == 0.0
    assert row["recoverable_headroom_vs_source_top1"] == 0.25
    assert row["headroom_capture_vs_source_top1"] == 0.0
