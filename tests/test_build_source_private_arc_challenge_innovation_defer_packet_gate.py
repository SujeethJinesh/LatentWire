from __future__ import annotations

import numpy as np

from scripts import build_source_private_arc_challenge_confidence_ecoc_packet_gate as ecoc_gate
from scripts import build_source_private_arc_challenge_innovation_defer_packet_gate as gate
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate


def _packet(source_scores: list[float]) -> dict:
    return ecoc_gate._source_packet_from_scores(
        source_scores,
        margin_edges=[0.05, 0.2, 0.5],
        entropy_edges=[0.5, 1.0, 1.5],
    )


def test_packet_features_include_disagreement_and_decoded_candidate_signal() -> None:
    packet = _packet([0.1, 0.9, 0.2, 0.0])
    features = gate._packet_features([0.8, 0.7, 0.1, 0.0], packet)

    assert features.shape[0] >= 10
    assert features[8] == 1.0
    assert features[13] == 1.0


def test_row_value_targets_are_source_minus_target_correctness() -> None:
    rows = [
        arc_gate.ArcRow(
            row_id="r0",
            content_id="c0",
            question="q",
            choices=("a", "b", "c"),
            choice_labels=("A", "B", "C"),
            answer_index=1,
            answer_label="B",
        )
    ]
    targets = gate._row_value_targets(rows, [[0.8, 0.7, 0.0]], [_packet([0.1, 0.9, 0.2])])

    assert targets.tolist() == [1.0]


def test_apply_packet_uses_value_threshold_to_fire() -> None:
    packet = _packet([0.1, 0.9, 0.2, 0.0])
    features = np.asarray(
        [
            gate._packet_features([0.8, 0.7, 0.1, 0.0], packet),
            gate._packet_features([0.8, 0.1, 0.7, 0.0], packet),
        ],
        dtype=np.float64,
    )
    values = np.asarray([1.0, -1.0], dtype=np.float64)
    model = gate._fit_value_model(features, values, fit_indices=np.asarray([0, 1]), ridge=0.1)

    scores, fired, decoded, predicted_gain = gate._apply_packet(
        [0.8, 0.7, 0.1, 0.0],
        packet,
        value_model=model,
        threshold=0.0,
        require_disagree=True,
        require_parity=True,
    )

    assert fired
    assert decoded == 1
    assert predicted_gain > 0.0
    assert scores[1] == max(scores)


def test_choose_defer_rule_prefers_helpful_packet_on_calibration_rows() -> None:
    rows = [
        arc_gate.ArcRow("r0", "c0", "q", ("a", "b", "c"), ("A", "B", "C"), 1, "B"),
        arc_gate.ArcRow("r1", "c1", "q", ("a", "b", "c"), ("A", "B", "C"), 0, "A"),
    ]
    packets = [_packet([0.1, 0.9, 0.2]), _packet([0.1, 0.9, 0.2])]
    scores = [[0.8, 0.7, 0.1], [0.9, 0.8, 0.0]]
    features = np.asarray([gate._packet_features(s, p) for s, p in zip(scores, packets, strict=True)])
    values = np.asarray([1.0, -1.0], dtype=np.float64)
    model = gate._fit_value_model(features, values, fit_indices=np.asarray([0, 1]), ridge=0.1)

    rule = gate._choose_defer_rule(rows, scores, packets, value_model=model)

    assert "threshold" in rule
    assert rule["require_parity"]
