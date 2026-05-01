from __future__ import annotations

import numpy as np

from scripts import build_source_private_hellaswag_hidden_summary_repair_probe as probe
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate


def _row(row_id: str, answer_index: int) -> arc_gate.ArcRow:
    choices = ("red apple", "blue sky", "green grass", "white snow")
    return arc_gate.ArcRow(
        row_id=row_id,
        content_id=f"content-{row_id}",
        question=f"Context for {row_id}",
        choices=choices,
        choice_labels=("A", "B", "C", "D"),
        answer_index=answer_index,
        answer_label=("A", "B", "C", "D")[answer_index],
        source_name="unit",
    )


def test_hidden_pair_scorer_recovers_candidate_axis() -> None:
    rows = [_row("r0", 0), _row("r1", 1), _row("r2", 2), _row("r3", 3)]
    features = np.zeros((len(rows), 4, 3), dtype=np.float64)
    for row_index, row in enumerate(rows):
        for choice_index in range(4):
            features[row_index, choice_index] = np.asarray(
                [
                    2.0 if choice_index == row.answer_index else -1.0,
                    float(choice_index == 0),
                    float(row_index) / 10.0,
                ],
                dtype=np.float64,
            )

    scorer = probe._fit_hidden_pair_scorer(rows, features, ridge=1e-6)
    _, predictions = probe._predict_hidden_rows(rows, features, scorer)

    assert predictions == [row.answer_index for row in rows]
    assert scorer["feature_dim"] == 3
    assert scorer["train_pair_rows"] == 16


def test_packet_rows_include_destructive_and_text_controls() -> None:
    rows = [_row("r0", 0), _row("r1", 1), _row("r2", 2)]
    predictions = [0, 1, 2]

    packet_rows = probe._packet_rows(
        rows=rows,
        source_predictions=predictions,
        budget_bytes=2,
        feature_dim=64,
        code_dim=32,
        seed=7,
        index_prior_rows=rows,
    )
    metrics = probe._condition_metrics(packet_rows)

    assert arc_gate.MATCHED_CONDITION in metrics
    assert "same_byte_structured_text" in metrics
    assert "candidate_derangement" in metrics
    assert metrics[arc_gate.MATCHED_CONDITION]["n"] == len(rows)


def test_paired_ci_predictions_reports_positive_delta() -> None:
    rows = [_row("r0", 0), _row("r1", 1), _row("r2", 2), _row("r3", 3)]
    candidate = [0, 1, 2, 3]
    baseline = [1, 1, 1, 1]

    ci = probe._paired_ci_predictions(rows, candidate, baseline, seed=11, samples=50)

    assert ci["mean"] > 0.0
    assert ci["ci95_high"] >= ci["ci95_low"]
