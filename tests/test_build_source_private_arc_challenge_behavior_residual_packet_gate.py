from __future__ import annotations

import numpy as np

from scripts import build_source_private_arc_challenge_behavior_residual_packet_gate as gate
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate


def _rows() -> list[arc_gate.ArcRow]:
    return [
        arc_gate.ArcRow(
            row_id="r0",
            content_id="c0",
            question="Which object is magnetic?",
            choices=("wood", "iron", "paper"),
            choice_labels=("A", "B", "C"),
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
    ]


def test_candidate_targets_are_gold_minus_target_probability() -> None:
    targets = gate._candidate_targets(_rows(), [[0.0, 2.0, 0.0], [1.0, 0.0]])

    assert targets.shape == (5,)
    assert targets[1] > 0.0
    assert targets[0] < 0.0
    assert targets[3] > 0.0
    assert np.isclose(float(targets[:3].sum()), 0.0, atol=1e-7)
    assert np.isclose(float(targets[3:].sum()), 0.0, atol=1e-7)


def test_ridge_scalar_map_predicts_fit_targets() -> None:
    features = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0],
        ],
        dtype=np.float64,
    )
    targets = np.asarray([1.0, -1.0, 0.0, 2.0], dtype=np.float64)

    model = gate._fit_ridge_scalar_map(
        features,
        targets,
        fit_indices=np.asarray([0, 1, 2], dtype=np.int64),
        ridge=0.1,
    )
    pred = model.predict(features)

    assert pred.shape == (4,)
    assert model.fit_r2 > 0.5
    assert pred[0] > pred[1]


def test_sparse_row_packet_quantization_reports_bytes_and_topk() -> None:
    rows = [
        np.asarray([1.0, -0.5, 0.2], dtype=np.float64),
        np.asarray([-0.1, 0.7], dtype=np.float64),
    ]

    decoded, metadata = gate._quantize_sparse_row_packets(
        rows,
        fit_row_count=1,
        top_k=1,
        quant_bits=3,
    )

    assert len(decoded) == 2
    assert metadata["kind"] == "candidate_local_behavior_residual_packet"
    assert metadata["packet_bits_per_row"] > 0
    assert metadata["packet_bytes_per_row"] > 0.0
    assert np.count_nonzero(decoded[0]) <= 1
    assert np.count_nonzero(decoded[1]) <= 1


def test_residual_weight_prefers_helpful_direction() -> None:
    rows = _rows()
    target_scores = [[0.0, -1.0, -2.0], [-1.0, 0.0]]
    residuals = [
        np.asarray([-1.0, 2.0, -1.0], dtype=np.float64),
        np.asarray([2.0, -1.0], dtype=np.float64),
    ]

    selected = gate._choose_residual_weight(rows, target_scores, residuals)

    assert selected["weight"] > 0.0
    assert selected["accuracy"] == 1.0


def test_top_atom_knockout_removes_largest_magnitude() -> None:
    knocked = gate._top_atom_knockout([0.2, -3.0, 1.0])

    assert knocked.tolist() == [0.2, 0.0, 1.0]
