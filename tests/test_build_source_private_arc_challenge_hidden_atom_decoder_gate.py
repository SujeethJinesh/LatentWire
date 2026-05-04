from __future__ import annotations

import numpy as np

from scripts import build_source_private_arc_challenge_hidden_atom_decoder_gate as gate
from scripts import build_source_private_arc_challenge_behavior_residual_packet_gate as behavior_gate
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
            choices=("hammer", "cloud"),
            choice_labels=("A", "B"),
            answer_index=0,
            answer_label="A",
        ),
    ]


def test_ridge_matrix_map_predicts_multitarget_fit() -> None:
    x = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float64)
    y = np.asarray([[1.0, -1.0], [-1.0, 1.0], [0.0, 0.0], [2.0, -2.0]], dtype=np.float64)

    model = gate._fit_ridge_matrix_map(x, y, fit_indices=np.asarray([0, 1, 2]), ridge=0.1)
    pred = model.predict(x)

    assert pred.shape == y.shape
    assert model.fit_r2 > 0.5
    assert pred[0, 0] > pred[1, 0]


def test_decoder_features_include_target_packet_and_interaction() -> None:
    target = np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]], dtype=np.float64)
    packet = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    features = gate._decoder_features(target, packet)

    assert features.shape == (2, 9)
    assert np.allclose(features[:, :5], target)
    assert np.allclose(features[:, 5:7], packet)


def test_packet_mutations_change_expected_axes() -> None:
    packet = np.asarray([[1.0, 2.0, 0.0], [0.5, -3.0, 1.0]], dtype=np.float64)

    assert np.allclose(gate._atom_shuffle_packet(packet), np.asarray([[0.0, 1.0, 2.0], [1.0, 0.5, -3.0]]))
    assert np.allclose(gate._coefficient_shuffle_packet(packet), np.asarray([[0.0, 2.0, 1.0], [1.0, -3.0, 0.5]]))
    assert np.allclose(gate._candidate_roll_packet(packet, shift=1)[0], packet[1])
    assert np.count_nonzero(gate._top_atom_knockout_packet(packet)) == np.count_nonzero(packet) - 1


def test_gate_rule_prefers_helpful_source_residual() -> None:
    rows = _rows()
    target_scores = [[0.4, 0.3, 0.2], [0.8, 0.1]]
    residuals = [np.asarray([-0.5, 1.0, -0.5]), np.asarray([0.2, -0.2])]

    rule = gate._choose_gate_rule(rows, target_scores, residuals)
    fused, fired = gate._fused_scores(target_scores[0], residuals[0], rule=rule)

    assert rule["train_accuracy"] >= 0.5
    assert fired
    assert behavior_gate._prediction(fused) == 1


def test_rows_from_candidate_values_are_row_centered() -> None:
    values = np.asarray([1.0, 3.0, 2.0, 5.0, 7.0], dtype=np.float64)

    rows = gate._rows_from_candidate_values(_rows(), values)

    assert len(rows) == 2
    assert np.isclose(float(rows[0].sum()), 0.0)
    assert np.isclose(float(rows[1].sum()), 0.0)
