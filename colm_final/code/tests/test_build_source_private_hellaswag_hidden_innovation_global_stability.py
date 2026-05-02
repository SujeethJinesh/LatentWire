from __future__ import annotations

from scripts import build_source_private_hellaswag_hidden_innovation_global_stability as global_stability
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


def test_best_label_predictions_keep_trained_control_separate() -> None:
    rows = [_row("0", 0), _row("1", 1), _row("2", 1), _row("3", 0)]
    source = [0, 0, 0, 0]
    trained = [[0, 1, 1, 1]]

    best, source_acc, trained_acc, best_acc, trained_choice = global_stability._best_label_predictions(
        rows,
        source,
        trained,
    )

    assert best == trained[0]
    assert trained_choice == trained[0]
    assert source_acc == 0.5
    assert trained_acc == 0.75
    assert best_acc == 0.75


def test_best_label_predictions_tie_prefers_source_but_reports_trained() -> None:
    rows = [_row("0", 0), _row("1", 1), _row("2", 1), _row("3", 0)]
    source = [0, 0, 0, 0]
    trained = [[1, 1, 1, 1]]

    best, source_acc, trained_acc, best_acc, trained_choice = global_stability._best_label_predictions(
        rows,
        source,
        trained,
    )

    assert best == source
    assert trained_choice == trained[0]
    assert source_acc == 0.5
    assert trained_acc == 0.5
    assert best_acc == 0.5


def test_policy_predictions_hybrid_uses_vote_when_hidden_mean_matches_score_mean() -> None:
    predictions = global_stability._policy_predictions(
        policy="hybrid_vote_on_score_agreement",
        hidden_mean_predictions=[0, 1, 2, 3],
        hidden_vote_predictions=[3, 2, 1, 0],
        score_mean_predictions=[0, 0, 2, 2],
    )

    assert predictions == [3, 1, 1, 3]


def test_readout_passes_when_matched_hidden_beats_controls_cleanly() -> None:
    rows = [_row(str(index), index % 4) for index in range(16)]
    selected = [row.answer_index for row in rows]
    weak_control = [0 for _ in rows]

    readout = global_stability._readout(
        rows=rows,
        predictions=selected,
        best_label_predictions=weak_control,
        score_predictions=weak_control,
        zero_predictions=weak_control,
        wrong_predictions=weak_control,
        candidate_roll_predictions=weak_control,
        source_label_accuracy=0.25,
        trained_label_accuracy=0.25,
        best_label_accuracy=0.25,
        bootstrap_seed=123,
        bootstrap_samples=50,
    )

    assert readout["rows"] == 16
    assert readout["correct"] == 16
    assert readout["selected_minus_best_label_copy"] == 0.75
    assert readout["selected_minus_score_only_bagged_control"] == 0.75
    assert readout["pass_gate"] is True
