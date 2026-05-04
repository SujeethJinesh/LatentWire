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


def test_batchtopk_behavior_atom_packet_is_sparse_and_quantized() -> None:
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

    packet, meta = gate._fit_batchtopk_behavior_atom_packet_from_features(
        source,
        behavior,
        fit_flat_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        rank=4,
        top_k=1,
        quant_bits=3,
        epochs=40,
        learning_rate=0.03,
        batch_size=4,
        reconstruction_weight=0.01,
        l1_weight=0.001,
        seed=11,
    )

    assert packet.shape == (6, 4)
    assert meta["kind"] == "train_fit_batchtopk_behavior_hidden_atom_packet_coordinates"
    assert meta["packet_rank"] == 4
    assert np.max(np.count_nonzero(packet, axis=1)) <= 1
    assert meta["packet_bits_per_candidate"] == 1 * (2 + 3)
    assert meta["batchtopk_active_atoms_max"] <= 1
    assert meta["batchtopk_behavior_fit_r2"] > -1.0


def test_packet_innovation_decoder_is_zero_for_zero_packet() -> None:
    target = np.asarray(
        [
            [0.1, 0.2, 0.3, 1.0, 0.4],
            [0.0, -0.5, 0.2, 0.5, -0.1],
            [0.4, 0.8, 0.6, 1.0, 0.2],
        ],
        dtype=np.float64,
    )
    packet = np.asarray([[1.0, 0.0], [0.0, 2.0], [1.0, -1.0]], dtype=np.float64)
    targets = np.asarray([0.5, -0.5, 0.25], dtype=np.float64)

    features = gate._innovation_decoder_features(target, packet)
    model = gate._fit_ridge_no_intercept_scalar_map(
        features,
        targets,
        fit_indices=np.asarray([0, 1, 2], dtype=np.int64),
        ridge=0.1,
    )
    zero_features = gate._innovation_decoder_features(target, np.zeros_like(packet))

    assert features.shape == (3, 12)
    assert np.allclose(model.predict(zero_features), 0.0)
    assert model.fit_r2 > 0.0


def test_event_triggered_gate_can_accept_helpful_packet_and_reject_zero_packet() -> None:
    target = np.asarray([0.0, 1.0], dtype=np.float64)
    helpful_residual = np.asarray([2.0, -2.0], dtype=np.float64)
    zero_residual = np.zeros_like(helpful_residual)
    helpful_packet = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
    zero_packet = np.zeros_like(helpful_packet)

    features = np.vstack(
        [
            gate._event_gate_features(target, helpful_residual, helpful_packet, residual_weight=1.0),
            gate._event_gate_features(target, zero_residual, zero_packet, residual_weight=1.0),
        ]
    )
    model = gate.behavior_gate._fit_ridge_scalar_map(
        features,
        np.asarray([1.0, 0.0], dtype=np.float64),
        fit_indices=np.asarray([0, 1], dtype=np.int64),
        ridge=0.01,
    )
    scores = model.predict(features)
    rule = gate.EventGateRule(
        residual_weight=1.0,
        threshold=float(np.mean(scores)),
        event_model=model,
        require_prediction_change=True,
        metadata={"gate_mode": "event_triggered"},
    )

    helpful_scores, helpful_fired, helpful_event_score = gate._event_triggered_fused_scores(
        target,
        helpful_residual,
        helpful_packet,
        rule=rule,
    )
    zero_scores, zero_fired, zero_event_score = gate._event_triggered_fused_scores(
        target,
        zero_residual,
        zero_packet,
        rule=rule,
    )

    assert helpful_fired
    assert int(np.argmax(helpful_scores)) == 0
    assert not zero_fired
    assert int(np.argmax(zero_scores)) == 1
    assert helpful_event_score > zero_event_score


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


def test_corruption_noop_decoder_trains_corrupt_packets_toward_noop() -> None:
    rows = _rows()
    target_features = np.zeros((9, 5), dtype=np.float64)
    source_packet = np.asarray(
        [
            [2.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, -2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    target_derived_packet = np.zeros_like(source_packet)
    behavior_targets = np.asarray([1.0, -1.0, 0.5, 1.0, -1.0, 0.5, 1.0, -1.0, 0.5], dtype=np.float64)

    decoder, meta = gate._fit_corruption_noop_decoder(
        rows=rows,
        train_indices=[0, 1, 2],
        target_features=target_features,
        source_packet_flat=source_packet,
        target_derived_packet_flat=target_derived_packet,
        behavior_targets=behavior_targets,
        source_selected=[1, 0, 1],
        decoder_mode="target_conditioned",
        corruption_loss_weight=0.1,
        corruption_condition_weights=None,
        ridge=0.01,
    )

    matched = decoder.predict(gate.hidden_gate._decoder_features(target_features, source_packet))
    zero = decoder.predict(gate.hidden_gate._decoder_features(target_features, np.zeros_like(source_packet)))

    assert meta["receiver_training_mode"] == "corruption_noop"
    assert meta["matched_training_examples"] == 9
    assert meta["corruption_training_examples"] == 9 * len(gate.EVENT_GATE_CORRUPTION_CONDITIONS)
    assert float(np.mean(np.abs(matched))) > 2.0 * float(np.mean(np.abs(zero)))
    assert decoder.fit_r2 > 0.0


def test_parse_args_accepts_corruption_noop_receiver() -> None:
    args = gate.parse_args(
        [
            "--receiver-training-mode",
            "corruption_noop",
            "--corruption-loss-weight",
            "0.25",
            "--corruption-condition-weights",
            "candidate_roll=0.5,top_atom_knockout=0.75",
            "--packet-integrity-mode",
            "candidate_atom",
            "--atom-basis-mode",
            "batchtopk_behavior",
            "--batchtopk-epochs",
            "12",
        ]
    )

    assert args.receiver_training_mode == "corruption_noop"
    assert args.corruption_loss_weight == 0.25
    assert args.packet_integrity_mode == "candidate_atom"
    assert args.atom_basis_mode == "batchtopk_behavior"
    assert args.batchtopk_epochs == 12
    assert gate._parse_condition_weight_spec(args.corruption_condition_weights) == {
        "candidate_roll": 0.5,
        "top_atom_knockout": 0.75,
    }


def test_parse_condition_weight_spec_rejects_unknown_control() -> None:
    try:
        gate._parse_condition_weight_spec("not_a_control=0.5")
    except ValueError as exc:
        assert "unknown corruption condition" in str(exc)
    else:  # pragma: no cover - defensive assertion.
        raise AssertionError("expected ValueError")


def test_packet_integrity_rule_rejects_candidate_roll() -> None:
    rows = _rows()
    target_features = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.1],
            [0.0, 1.0, 0.1],
            [0.0, 0.0, 1.1],
            [1.0, 0.1, 0.0],
            [0.0, 1.1, 0.0],
            [0.0, 0.1, 1.0],
        ],
        dtype=np.float64,
    )
    source_packet = target_features.copy()
    target_derived_packet = np.zeros_like(source_packet)
    fit_indices = np.arange(source_packet.shape[0], dtype=np.int64)

    rule = gate._choose_packet_integrity_rule(
        rows=rows,
        train_indices=[0, 1, 2],
        target_features=target_features,
        source_packet_flat=source_packet,
        target_derived_packet_flat=target_derived_packet,
        fit_candidate_indices=fit_indices,
        source_selected=[1, 0, 1],
        ridge=0.01,
    )
    row_packets = gate.hidden_gate._row_packet_arrays(rows, source_packet)
    matched_accept, matched_score = gate._packet_integrity_accept(
        packet=row_packets[0],
        target_features=target_features[:3],
        rule=rule,
    )
    rolled_accept, rolled_score = gate._packet_integrity_accept(
        packet=gate.hidden_gate._candidate_roll_packet(row_packets[0], shift=1),
        target_features=target_features[:3],
        rule=rule,
    )

    assert matched_accept
    assert not rolled_accept
    assert matched_score is not None and rolled_score is not None
    assert matched_score > rolled_score
