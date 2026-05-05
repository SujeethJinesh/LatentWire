from __future__ import annotations

import numpy as np

from scripts import build_source_private_arc_candidate_local_source_evidence_repair_preflight as candidate
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate


def _rows() -> list[arc_gate.ArcRow]:
    return [
        arc_gate.ArcRow("r0", "c0", "magnetic?", ("wood", "iron"), ("A", "B"), 1, "B"),
        arc_gate.ArcRow("r1", "c1", "plant need?", ("sun", "sand"), ("A", "B"), 0, "A"),
        arc_gate.ArcRow("r2", "c2", "tiny cells?", ("scope", "hammer"), ("A", "B"), 0, "A"),
        arc_gate.ArcRow("r3", "c3", "conducts?", ("copper", "rubber"), ("A", "B"), 0, "A"),
    ]


def test_candidate_local_readout_uses_candidate_source_signal() -> None:
    rows = _rows()
    source_candidates = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
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

    result, prediction_rows = candidate.evaluate_candidate_local_readout(
        rows,
        source_candidates=source_candidates,
        public_candidates=public_candidates,
        source_predictions=source_predictions,
        fit_indices=[0, 1],
        eval_indices=[2, 3],
        audit_rows=audit,
        ridge=1e-6,
        seed=11,
        bootstrap_samples=50,
    )

    assert result["condition_metrics"][candidate.MATCHED_CONDITION]["accuracy"] == 1.0
    assert result["condition_metrics"]["candidate_source_roll_control"]["accuracy"] == 0.0
    assert result["headline"]["matched_accuracy"] >= result["headline"]["best_control_accuracy"]
    assert {row["condition"] for row in prediction_rows} >= {
        candidate.MATCHED_CONDITION,
        "candidate_source_roll_control",
        "packet_only_source_index",
        "same_byte_visible_text",
    }
