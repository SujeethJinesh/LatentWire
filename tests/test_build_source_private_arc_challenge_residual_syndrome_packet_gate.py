from __future__ import annotations

import numpy as np

from scripts import build_source_private_arc_challenge_residual_syndrome_packet_gate as gate
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate


def _rows() -> list[arc_gate.ArcRow]:
    return [
        arc_gate.ArcRow(
            row_id="r0",
            content_id="c0",
            question="Which material conducts electricity?",
            choices=("rubber", "copper", "glass", "paper"),
            choice_labels=("A", "B", "C", "D"),
            answer_index=1,
            answer_label="B",
        ),
        arc_gate.ArcRow(
            row_id="r1",
            content_id="c1",
            question="Which object is a tool?",
            choices=("hammer", "cloud", "river", "moon"),
            choice_labels=("A", "B", "C", "D"),
            answer_index=0,
            answer_label="A",
        ),
    ]


def test_packet_syndrome_and_parity_are_consistent() -> None:
    packet = gate._source_packet_from_scores(
        [0.1, 0.9, 0.4, -0.2],
        margin_edges=[0.1, 0.3, 0.6],
        entropy_edges=[0.5, 1.0, 1.5],
        syndrome_bits=4,
    )

    assert packet["candidate_count"] == 4
    assert packet["pair_bit_count"] == 6
    assert len(packet["syndrome"]) == 4
    assert gate._parity_ok(packet)


def test_decoding_uses_target_side_information_to_pick_coset_member() -> None:
    source_scores = [0.1, 0.9, 0.4, -0.2]
    target_scores = [0.15, 0.3, 0.8, -0.1]
    packet = gate._source_packet_from_scores(
        source_scores,
        margin_edges=[0.1, 0.3, 0.6],
        entropy_edges=[0.5, 1.0, 1.5],
        syndrome_bits=4,
    )

    decoded = gate._decode_pair_bits(packet, target_scores, use_target_side_info=True)
    residual = gate._pair_residual_votes(decoded, target_scores)

    assert decoded.shape == gate._pair_bits(target_scores).shape
    assert residual.shape == (4,)
    assert np.isclose(float(residual.sum()), 0.0)


def test_parity_flip_prevents_packet_application_when_required() -> None:
    packet = gate._source_packet_from_scores(
        [0.1, 0.9, 0.4, -0.2],
        margin_edges=[0.1, 0.3, 0.6],
        entropy_edges=[0.5, 1.0, 1.5],
        syndrome_bits=4,
    )
    rule = {
        "residual_weight": 1.0,
        "min_margin_bin": 0,
        "max_entropy_bin": 3,
        "max_target_margin": float("inf"),
        "require_innovation": False,
        "require_parity": True,
    }

    _scores, fired, _decoded = gate._apply_packet([0.15, 0.3, 0.8, -0.1], gate._mutate_packet(packet, parity_flip=True), rule)

    assert not fired


def test_gate_rule_can_select_helpful_residual_direction() -> None:
    rows = _rows()
    target_scores = [[0.2, 0.1, 0.8, -0.2], [0.8, 0.1, 0.0, -0.1]]
    packets = [
        gate._source_packet_from_scores(
            [0.1, 0.9, 0.4, -0.2],
            margin_edges=[0.1, 0.3, 0.6],
            entropy_edges=[0.5, 1.0, 1.5],
            syndrome_bits=4,
        ),
        gate._source_packet_from_scores(
            [1.0, 0.2, 0.1, 0.0],
            margin_edges=[0.1, 0.3, 0.6],
            entropy_edges=[0.5, 1.0, 1.5],
            syndrome_bits=4,
        ),
    ]

    rule = gate._choose_gate_rule(rows, target_scores, packets, target_margin_edges=[0.25, 0.5, 1.0])

    assert rule["train_accuracy"] >= 0.5
    assert rule["residual_weight"] >= 0.0


def test_mutating_syndrome_changes_bits_but_keeps_header_parity_consistent() -> None:
    packet = gate._source_packet_from_scores(
        [0.1, 0.9, 0.4, -0.2],
        margin_edges=[0.1, 0.3, 0.6],
        entropy_edges=[0.5, 1.0, 1.5],
        syndrome_bits=4,
    )
    shuffled = gate._mutate_packet(packet, syndrome_roll=1)

    assert shuffled["syndrome"] != packet["syndrome"]
    assert gate._parity_ok(shuffled)
