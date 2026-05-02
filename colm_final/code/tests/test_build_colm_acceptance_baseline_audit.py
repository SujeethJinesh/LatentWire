from __future__ import annotations

import argparse

import pytest

from scripts import build_colm_acceptance_baseline_audit as audit
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate


def _arc_row(row_id: str, answer_index: int) -> arc_gate.ArcRow:
    return arc_gate.ArcRow(
        row_id=row_id,
        content_id=f"content-{row_id}",
        question=f"What is true for {row_id}?",
        choices=("ice", "fire", "rain"),
        choice_labels=("A", "B", "C"),
        answer_index=answer_index,
        answer_label=("A", "B", "C")[answer_index],
    )


def _prediction(
    *,
    condition: str,
    row: arc_gate.ArcRow,
    prediction_index: int,
    seed: int = 7,
) -> dict:
    return audit._prediction_row(
        condition=condition,
        row=row,
        prediction_index=prediction_index,
        payload_bytes=1,
        seed=seed,
        split="test",
        metadata={"source_selected_index": prediction_index},
    )


def test_add_baseline_rows_adds_source_index_text_rank_and_random() -> None:
    rows = [_arc_row("r0", 0), _arc_row("r1", 1), _arc_row("r2", 2)]
    source_predictions = [0, 1, 1]

    with_baselines = audit._add_baseline_rows(
        rows=rows,
        prediction_rows=[],
        source_predictions=source_predictions,
        seed=17,
        split="test",
    )

    by_condition = {condition: audit._condition_rows(with_baselines, condition) for condition in audit.BASELINE_CONDITIONS}
    assert set(by_condition) == set(audit.BASELINE_CONDITIONS)
    assert all(len(condition_rows) == len(rows) for condition_rows in by_condition.values())
    assert [row["prediction_index"] for row in by_condition["source_index_byte"]] == source_predictions
    assert [row["payload_bytes"] for row in by_condition["source_choice_label_text"]] == [1, 1, 1]
    assert by_condition["source_rank_code"][0]["metadata"]["source_score_available"] is False
    assert by_condition["source_index_byte"][0]["metadata"]["forbidden_source_fields"]


def test_seed_summary_emits_paired_source_index_and_text_cis() -> None:
    rows = [_arc_row("r0", 0), _arc_row("r1", 1), _arc_row("r2", 2), _arc_row("r3", 0)]
    prediction_rows = []
    for row in rows:
        source_prediction = row.answer_index
        packet_prediction = row.answer_index if row.row_id != "r3" else 1
        prediction_rows.extend(
            [
                _prediction(condition=arc_gate.MATCHED_CONDITION, row=row, prediction_index=packet_prediction),
                _prediction(condition="target_only", row=row, prediction_index=2),
                _prediction(condition="same_byte_structured_text", row=row, prediction_index=2),
                _prediction(condition="source_index_byte", row=row, prediction_index=source_prediction),
                _prediction(condition="source_choice_label_text", row=row, prediction_index=source_prediction),
                _prediction(condition="source_rank_code", row=row, prediction_index=source_prediction),
                _prediction(condition="entropy_matched_random_index", row=row, prediction_index=1),
                _prediction(condition="candidate_derangement", row=row, prediction_index=2),
            ]
        )

    summary = audit._seed_summary(rows=prediction_rows, seed=11, bootstrap_samples=20)

    assert summary["packet_accuracy"] == pytest.approx(0.75)
    assert summary["source_index_accuracy"] == pytest.approx(1.0)
    assert summary["paired_ci"]["packet_vs_source_index"]["mean"] == pytest.approx(-0.25)
    assert "packet_vs_same_budget_text" in summary["paired_ci"]
    assert "packet_vs_best_destructive" in summary["paired_ci"]


def test_aggregate_records_that_packet_does_not_beat_source_index() -> None:
    seed_rows = [
        {
            "packet_accuracy": 0.75,
            "target_accuracy": 0.25,
            "same_budget_text_accuracy": 0.25,
            "source_index_accuracy": 1.0,
            "source_choice_text_accuracy": 1.0,
            "source_rank_code_accuracy": 1.0,
            "entropy_matched_random_index_accuracy": 0.25,
            "packet_follows_source_index": 0.75,
            "paired_ci": {
                "packet_vs_target": {"mean": 0.5, "ci95_low": 0.25},
                "packet_vs_same_budget_text": {"mean": 0.5, "ci95_low": 0.25},
                "packet_vs_source_index": {"mean": -0.25, "ci95_low": -0.5},
                "packet_vs_source_choice_text": {"mean": -0.25, "ci95_low": -0.5},
                "packet_vs_source_rank_code": {"mean": -0.25, "ci95_low": -0.5},
                "packet_vs_entropy_random_index": {"mean": 0.5, "ci95_low": 0.25},
                "packet_vs_best_destructive": {"mean": 0.5, "ci95_low": 0.25},
            },
        },
        {
            "packet_accuracy": 0.5,
            "target_accuracy": 0.25,
            "same_budget_text_accuracy": 0.25,
            "source_index_accuracy": 0.5,
            "source_choice_text_accuracy": 0.5,
            "source_rank_code_accuracy": 0.5,
            "entropy_matched_random_index_accuracy": 0.25,
            "packet_follows_source_index": 1.0,
            "paired_ci": {
                "packet_vs_target": {"mean": 0.25, "ci95_low": 0.0},
                "packet_vs_same_budget_text": {"mean": 0.25, "ci95_low": 0.0},
                "packet_vs_source_index": {"mean": 0.0, "ci95_low": 0.0},
                "packet_vs_source_choice_text": {"mean": 0.0, "ci95_low": 0.0},
                "packet_vs_source_rank_code": {"mean": 0.0, "ci95_low": 0.0},
                "packet_vs_entropy_random_index": {"mean": 0.25, "ci95_low": 0.0},
                "packet_vs_best_destructive": {"mean": 0.25, "ci95_low": 0.0},
            },
        },
    ]

    aggregate = audit._aggregate(seed_rows)

    assert aggregate["packet_beats_source_index_all_seeds"] is False
    assert aggregate["packet_ties_or_loses_source_index"] is True
    assert aggregate["packet_vs_source_index_ci95_low_min"] == pytest.approx(-0.5)


def test_parse_budgets_rejects_empty_and_nonpositive_values() -> None:
    assert audit._parse_budgets("2, 3,8") == [2, 3, 8]
    with pytest.raises(argparse.ArgumentTypeError):
        audit._parse_budgets("")
    with pytest.raises(argparse.ArgumentTypeError):
        audit._parse_budgets("1,2,4")
