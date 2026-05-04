from __future__ import annotations

import numpy as np

from scripts import (
    build_source_private_hellaswag_decision_sparse_common_basis_hidden_innovation_packet_gate as gate,
)


def _toy_params() -> dict[str, object]:
    return {
        "mean": np.zeros(2, dtype=np.float64),
        "scale": np.ones(2, dtype=np.float64),
        "encoder_weight": np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        "encoder_bias": np.zeros(3, dtype=np.float64),
        "transmit_atom_ids": np.asarray([0, 1], dtype=np.int64),
    }


def test_sparse_atom_packet_preserves_candidate_low_bits_and_reserves_zero_slot() -> None:
    shared = np.zeros((3, 4, 2), dtype=np.float64)
    packet = np.asarray([0, 1, 2], dtype=np.int64)
    shared[0, 0] = [3.0, 1.0]
    shared[1, 1] = [0.0, 4.0]
    shared[2, 2] = [-2.0, -1.0]

    encoded = gate._encode_sparse_atom_packet(
        source_shared=shared,
        packet=packet,
        params=_toy_params(),
        sae_topk=1,
    )

    assert np.array_equal(encoded["code"] % gate.CANDIDATE_COUNT, packet)
    assert encoded["atom_slot"].tolist() == [1, 2, 0]
    assert encoded["code"].tolist() == [4, 9, 2]
    assert int(encoded["code"].max()) < (2 + 1) * gate.CANDIDATE_COUNT


def test_atom_slot_permutation_preserves_candidate_bits_and_zero_slot() -> None:
    code = np.asarray([0, 5, 10, 11], dtype=np.int64)
    rng = np.random.default_rng(123)

    permuted = gate._permute_atom_slots(code, slot_count=2, rng=rng)

    assert np.array_equal(permuted % gate.CANDIDATE_COUNT, code % gate.CANDIDATE_COUNT)
    assert int(permuted[0]) == 0
    assert np.all(permuted // gate.CANDIDATE_COUNT <= 2)


def test_decoder_features_with_sae_includes_selected_target_atom_value() -> None:
    qwen_scores = np.asarray([[1.0, 0.0, -1.0, 0.5]], dtype=np.float64)
    source_code_values = np.asarray([5], dtype=np.int64)  # atom slot 1, candidate low bits 1
    target_atom_values = np.zeros((1, 4, 2), dtype=np.float64)
    target_atom_values[0, :, 0] = [0.1, 0.2, 0.3, 0.4]

    features = gate._candidate_decoder_features_with_sae(
        qwen_scores=qwen_scores,
        qwen_target=np.asarray([1], dtype=np.int64),
        qwen_mean=np.asarray([0], dtype=np.int64),
        qwen_hybrid=np.asarray([1], dtype=np.int64),
        source_code_values=source_code_values,
        codebook_size=12,
        target_atom_values=target_atom_values,
    )

    assert features.shape[0] == 1
    assert features.shape[1] == gate.CANDIDATE_COUNT
    # Penultimate appended channel is the selected target atom value.
    assert np.allclose(features[0, :, -2], [0.1, 0.2, 0.3, 0.4])
    # Final appended channel marks that a nonzero atom slot was transmitted.
    assert np.allclose(features[0, :, -1], 1.0)


def test_ambiguity_code_packs_source_pair_and_four_bit_atom_slot() -> None:
    code = gate._pack_ambiguity_code(
        source_top1=np.asarray([0, 2, 3], dtype=np.int64),
        source_top2=np.asarray([1, 3, 0], dtype=np.int64),
        atom_slot=np.asarray([0, 7, 22], dtype=np.int64),
    )

    top1, top2, slot = gate._decode_ambiguity_code(code)

    assert code.tolist() == [4, 126, 3]
    assert top1.tolist() == [0, 2, 3]
    assert top2.tolist() == [1, 3, 0]
    assert slot.tolist() == [0, 7, 0]


def test_ambiguity_action_features_use_rowwise_target_atoms() -> None:
    qwen_scores = np.asarray([[1.0, 0.0, -1.0, 0.5], [0.0, 1.0, 0.5, -0.5]], dtype=np.float64)
    code = gate._pack_ambiguity_code(
        source_top1=np.asarray([0, 1], dtype=np.int64),
        source_top2=np.asarray([3, 2], dtype=np.int64),
        atom_slot=np.asarray([1, 1], dtype=np.int64),
    )
    target_atoms = np.zeros((2, 4, gate.AMBIGUITY_ATOM_SLOTS), dtype=np.float64)
    target_atoms[0, 0, 0] = 10.0
    target_atoms[1, 1, 0] = 20.0

    features, actions, diagnostics = gate._ambiguity_action_features(
        qwen_scores=qwen_scores,
        qwen_target=np.asarray([0, 1], dtype=np.int64),
        qwen_mean=np.asarray([3, 2], dtype=np.int64),
        qwen_hybrid=np.asarray([0, 1], dtype=np.int64),
        ambiguity_code=code,
        target_atom_values=target_atoms,
    )

    assert features.shape[0] == 2
    assert features.shape[1] == len(gate.AMBIGUITY_ACTION_NAMES)
    assert actions[:, 0].tolist() == [0, 1]
    assert diagnostics["atom_slot"].tolist() == [1, 1]
    assert 10.0 in features[0, 0].tolist()
    assert 20.0 in features[1, 0].tolist()
