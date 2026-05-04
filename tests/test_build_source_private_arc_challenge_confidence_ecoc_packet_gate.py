from __future__ import annotations

from scripts import build_source_private_arc_challenge_confidence_ecoc_packet_gate as gate


def test_packet_bins_codeword_and_parity_are_consistent() -> None:
    margin_edges = gate._fit_bin_edges([0.1, 0.2, 0.3, 0.4], bins=4)
    entropy_edges = gate._fit_bin_edges([0.5, 0.6, 0.7, 0.8], bins=4)

    packet = gate._source_packet_from_scores(
        [0.1, 0.7, 0.4, -0.1],
        margin_edges=margin_edges,
        entropy_edges=entropy_edges,
    )

    assert packet["top1_index"] == 1
    assert packet["top2_index"] == 2
    assert gate._decode_ecoc_candidate(packet) == 1
    assert gate._parity_ok(packet)


def test_gate_respects_target_uncertainty_reliability_and_parity() -> None:
    packet = gate._source_packet_from_scores(
        [0.1, 0.8, 0.4, -0.1],
        margin_edges=[0.1, 0.2, 0.3],
        entropy_edges=[0.5, 1.5, 2.5],
    )
    rule = {
        "min_margin_bin": 1,
        "max_entropy_bin": 3,
        "max_target_margin": 0.2,
        "require_disagree": True,
        "require_parity": True,
    }

    assert gate._gate_fires(packet, [0.3, 0.29, 0.2, 0.0], rule)
    assert not gate._gate_fires(packet, [0.9, 0.1, 0.0, -0.1], rule)
    assert not gate._gate_fires(gate._mutate_packet(packet, parity_flip=True), [0.3, 0.29, 0.2, 0.0], rule)


def test_apply_packet_forces_decoded_candidate_only_when_gate_fires() -> None:
    packet = gate._source_packet_from_scores(
        [0.1, 0.8, 0.4, -0.1],
        margin_edges=[0.1, 0.2, 0.3],
        entropy_edges=[0.5, 1.5, 2.5],
    )
    rule = {
        "min_margin_bin": 1,
        "max_entropy_bin": 3,
        "max_target_margin": 0.2,
        "require_disagree": True,
        "require_parity": True,
    }

    scores, fired, decoded = gate._apply_packet([0.3, 0.29, 0.2, 0.0], packet, rule)

    assert fired
    assert decoded == 1
    assert scores[1] == max(scores)


def test_mutating_codeword_changes_decoded_candidate_or_parity_surface() -> None:
    packet = gate._source_packet_from_scores(
        [0.1, 0.8, 0.4, -0.1],
        margin_edges=[0.1, 0.2, 0.3],
        entropy_edges=[0.5, 1.5, 2.5],
    )
    shuffled = gate._mutate_packet(packet, code_roll=1)

    assert gate._parity_ok(shuffled)
    assert shuffled["codeword"] != packet["codeword"]


def test_source_score_quantized_control_preserves_best_candidate() -> None:
    scores = gate._source_score_quantized_control([0.1, 0.8, 0.4, -0.1], bits=4)

    assert max(range(len(scores)), key=lambda index: scores[index]) == 1
    assert abs(sum(scores)) < 1e-7
