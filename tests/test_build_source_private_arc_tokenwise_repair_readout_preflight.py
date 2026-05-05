from __future__ import annotations

import numpy as np

from scripts import build_source_private_arc_tokenwise_repair_readout_preflight as repair
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate


def _rows() -> list[arc_gate.ArcRow]:
    return [
        arc_gate.ArcRow(
            row_id="r0",
            content_id="c0",
            question="Which object is magnetic?",
            choices=("wood", "iron"),
            choice_labels=("A", "B"),
            answer_index=1,
            answer_label="B",
        ),
        arc_gate.ArcRow(
            row_id="r1",
            content_id="c1",
            question="What do plants need?",
            choices=("sunlight", "sand"),
            choice_labels=("A", "B"),
            answer_index=0,
            answer_label="A",
        ),
        arc_gate.ArcRow(
            row_id="r2",
            content_id="c2",
            question="What is used to see tiny cells?",
            choices=("microscope", "hammer"),
            choice_labels=("A", "B"),
            answer_index=0,
            answer_label="A",
        ),
        arc_gate.ArcRow(
            row_id="r3",
            content_id="c3",
            question="Which material conducts electricity?",
            choices=("copper", "rubber"),
            choice_labels=("A", "B"),
            answer_index=0,
            answer_label="A",
        ),
    ]


def test_repair_readout_evaluates_matched_and_destructive_controls() -> None:
    rows = _rows()
    source_rows = np.asarray(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )
    public_candidates = np.zeros((8, 3), dtype=np.float64)
    source_predictions = [0, 1, 1, 1]
    audit = {
        row.content_id: {
            "target_only": {"scores": [0.4, 0.6]},
            "same_byte_visible_text": {"scores": [0.5, 0.4]},
        }
        for row in rows
    }

    result, prediction_rows = repair.evaluate_repair_readout(
        rows,
        source_rows=source_rows,
        public_candidates=public_candidates,
        source_predictions=source_predictions,
        fit_indices=[0, 1],
        eval_indices=[2, 3],
        audit_rows=audit,
        ridge=1e-6,
        seed=7,
        bootstrap_samples=50,
    )

    assert result["condition_metrics"][repair.MATCHED_CONDITION]["accuracy"] == 1.0
    assert "wrong_row_source_control" in result["condition_metrics"]
    assert "target_only" in result["condition_metrics"]
    assert result["headline"]["matched_accuracy"] == 1.0
    assert {row["condition"] for row in prediction_rows} >= {
        repair.MATCHED_CONDITION,
        "zero_source_control",
        "packet_only_source_index",
        "same_byte_visible_text",
    }
