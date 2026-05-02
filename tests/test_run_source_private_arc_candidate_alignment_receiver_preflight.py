from __future__ import annotations

import numpy as np

from scripts import run_source_private_arc_candidate_alignment_receiver_preflight as gate
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate


def _rows() -> list[arc_gate.ArcRow]:
    return [
        arc_gate.ArcRow(
            row_id="r0",
            content_id="c0",
            question="Which option is first?",
            choices=("alpha", "beta", "gamma"),
            choice_labels=("A", "B", "C"),
            answer_index=0,
            answer_label="A",
        ),
        arc_gate.ArcRow(
            row_id="r1",
            content_id="c1",
            question="Which option is second?",
            choices=("alpha", "beta", "gamma"),
            choice_labels=("A", "B", "C"),
            answer_index=1,
            answer_label="B",
        ),
        arc_gate.ArcRow(
            row_id="r2",
            content_id="c2",
            question="Which option is third?",
            choices=("alpha", "beta", "gamma"),
            choice_labels=("A", "B", "C"),
            answer_index=2,
            answer_label="C",
        ),
        arc_gate.ArcRow(
            row_id="r3",
            content_id="c3",
            question="Which option repeats first?",
            choices=("alpha", "beta", "gamma"),
            choice_labels=("A", "B", "C"),
            answer_index=0,
            answer_label="A",
        ),
    ]


def _source_rows() -> list[np.ndarray]:
    rows = []
    for answer in (0, 1, 2, 0):
        row = np.full((3, 2), -1.0, dtype=np.float64)
        row[answer] = np.asarray([4.0, 2.0], dtype=np.float64)
        rows.append(row)
    return rows


def test_sign_sketch_and_packet_bytes_are_fixed_size() -> None:
    source_rows = _source_rows()
    projection = np.eye(2, dtype=np.float64)

    sketches = gate._sketch_rows(source_rows, projection, quantization="sign")

    assert set(np.unique(sketches[0])) <= {-1.0, 1.0}
    assert gate._packet_bytes(max_candidate_count=5, sketch_dim=16, quantization="sign") == 10
    assert gate._packet_bytes(max_candidate_count=5, sketch_dim=16, quantization="int8") == 82
    assert gate._packet_bytes(max_candidate_count=5, sketch_dim=16, quantization="none") == 320


def test_candidate_alignment_receiver_uses_matched_slots() -> None:
    rows = _rows()
    source_rows = _source_rows()
    public_rows = [np.zeros((3, 2), dtype=np.float64) for _ in rows]
    fit_indices = [0, 1, 2]
    eval_indices = [3]
    receiver = gate._fit_receiver(
        rows,
        source_rows,
        public_rows,
        fit_indices,
        max_candidate_count=3,
        target_only=False,
        label_shuffle=False,
        l2_grid=(0.01, 0.1, 1.0),
    )

    matched = gate._scores_by_row(
        rows,
        receiver,
        source_rows,
        public_rows,
        eval_indices,
        max_candidate_count=3,
    )[3]
    rolled_source = gate._source_control_rows(
        source_rows,
        fit_indices=fit_indices,
        eval_indices=eval_indices,
        control="candidate_roll_source",
        seed=7,
    )
    rolled = gate._scores_by_row(
        rows,
        receiver,
        rolled_source,
        public_rows,
        eval_indices,
        max_candidate_count=3,
    )[3]

    assert gate._prediction(matched) == 0
    assert gate._prediction(rolled) != 0


def test_source_control_rows_are_destructive_and_deterministic() -> None:
    source_rows = _source_rows()

    zero = gate._source_control_rows(
        source_rows,
        fit_indices=[0, 1, 2],
        eval_indices=[3],
        control="zero_source",
        seed=11,
    )
    noise_a = gate._source_control_rows(
        source_rows,
        fit_indices=[0, 1, 2],
        eval_indices=[3],
        control="same_norm_noise",
        seed=11,
    )
    noise_b = gate._source_control_rows(
        source_rows,
        fit_indices=[0, 1, 2],
        eval_indices=[3],
        control="same_norm_noise",
        seed=11,
    )

    assert np.allclose(zero[0], 0.0)
    assert np.allclose(noise_a[3], noise_b[3])
    assert np.allclose(np.linalg.norm(noise_a[3]), np.linalg.norm(source_rows[3]), atol=1e-6)
    assert not np.allclose(noise_a[3], source_rows[3])


def test_condition_metrics_tracks_paired_control_deltas() -> None:
    rows = []
    for condition in gate.REPORT_CONDITIONS:
        rows.append(
            {
                "content_id": "c0",
                "condition": condition,
                "correct": condition == gate.MATCHED_CONDITION,
                "margin": 1.0 if condition == gate.MATCHED_CONDITION else -1.0,
            }
        )

    metrics = gate._condition_metrics(rows, seed=3, bootstrap_samples=20)

    assert metrics[gate.MATCHED_CONDITION]["accuracy"] == 1.0
    assert metrics["target_public_only"]["accuracy"] == 0.0
    assert metrics[gate.MATCHED_CONDITION]["paired_accuracy_vs_target_public_only"]["mean"] == 1.0
    assert metrics[gate.MATCHED_CONDITION]["mean_margin_delta_vs_target_public_only"] == 2.0
