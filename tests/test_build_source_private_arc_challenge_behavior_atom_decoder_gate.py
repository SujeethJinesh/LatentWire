from __future__ import annotations

import numpy as np

from scripts import build_source_private_arc_challenge_behavior_atom_decoder_gate as gate
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate


def _rows() -> list[arc_gate.ArcRow]:
    return [
        arc_gate.ArcRow(
            row_id="r0",
            content_id="c0",
            question="Which material conducts electricity?",
            choices=("rubber", "copper", "glass"),
            choice_labels=("A", "B", "C"),
            answer_index=1,
            answer_label="B",
        ),
        arc_gate.ArcRow(
            row_id="r1",
            content_id="c1",
            question="Which object is a tool?",
            choices=("hammer", "cloud", "river"),
            choice_labels=("A", "B", "C"),
            answer_index=0,
            answer_label="A",
        ),
        arc_gate.ArcRow(
            row_id="r2",
            content_id="c2",
            question="Which item is magnetic?",
            choices=("paper", "iron", "sand"),
            choice_labels=("A", "B", "C"),
            answer_index=1,
            answer_label="B",
        ),
    ]


def test_behavior_target_matrix_has_candidate_rows_and_multiple_targets() -> None:
    scores = [[0.1, 0.2, -0.1], [0.8, 0.1, 0.0], [0.0, 0.4, 0.2]]

    targets = gate._behavior_target_matrix(_rows(), scores)

    assert targets.shape == (9, 10)
    assert np.any(targets[:, 0] > 0.0)
    assert np.any(targets[:, 0] < 0.0)


def test_behavior_atom_packet_is_sparse_and_quantized() -> None:
    source = np.asarray(
        [
            [2.0, 0.0, 1.0],
            [1.5, 0.1, 0.9],
            [-1.0, 2.0, 0.0],
            [-0.8, 2.2, 0.1],
            [0.1, -1.5, 2.0],
            [0.0, -1.2, 2.4],
        ],
        dtype=np.float64,
    )
    behavior = np.asarray(
        [
            [1.0, 0.5],
            [0.9, 0.4],
            [-1.0, 0.2],
            [-0.8, 0.1],
            [0.0, -1.0],
            [0.1, -0.9],
        ],
        dtype=np.float64,
    )

    packet, meta = gate._fit_behavior_atom_packet_from_features(
        source,
        behavior,
        fit_flat_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        rank=2,
        top_k=1,
        quant_bits=3,
    )

    assert packet.shape == (6, 2)
    assert meta["kind"] == "train_fit_behavior_supervised_hidden_atom_packet_coordinates"
    assert meta["packet_rank"] == 2
    assert np.max(np.count_nonzero(packet, axis=1)) <= 1
    assert meta["packet_bits_per_candidate"] == 1 * (1 + 3)


def test_same_source_choice_shuffle_prefers_same_choice_same_shape() -> None:
    rows = _rows()
    index = gate._same_choice_shuffle_index(
        row_index=0,
        eval_indices=[0, 1, 2],
        rows=rows,
        source_selected=[1, 0, 1],
    )

    assert index == 2


def test_same_shape_shuffle_ignores_source_choice() -> None:
    rows = _rows()
    index = gate._same_shape_shuffle_index(row_index=0, eval_indices=[0, 1, 2], rows=rows)

    assert index == 1
