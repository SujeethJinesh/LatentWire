from __future__ import annotations

import numpy as np

from scripts import build_source_private_hellaswag_crosscoder_hidden_code_scout as gate


def test_flat_candidate_indices() -> None:
    indices = gate._flat_candidate_indices(np.asarray([0, 2], dtype=np.int64))

    assert indices.tolist() == [0, 1, 2, 3, 8, 9, 10, 11]


def test_linear_crosscoder_projects_paired_candidates() -> None:
    source = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
            [[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]],
            [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]],
        ],
        dtype=np.float64,
    )
    target = source * 2.0

    crosscoder = gate._fit_linear_crosscoder(
        source_candidate_features=source,
        target_candidate_features=target,
        fit_indices=np.asarray([0, 1, 2], dtype=np.int64),
        pca_dims=2,
        shared_dims=2,
    )
    source_shared, target_shared = gate._apply_linear_crosscoder(
        source_candidate_features=source,
        target_candidate_features=target,
        crosscoder=crosscoder,
    )

    assert source_shared.shape == (3, 4, 2)
    assert target_shared.shape == (3, 4, 2)
    assert len(crosscoder["singular_values"]) == 2


def test_crosscoder_kmeans_codes_preserve_candidate_low_bits() -> None:
    train_shared = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [4.0, 4.0],
            [4.1, 4.0],
            [-3.0, 2.0],
            [-3.1, 2.0],
        ],
        dtype=np.float64,
    )
    eval_shared = np.asarray([[0.05, 0.0], [4.05, 4.0]], dtype=np.float64)
    train_packet = np.asarray([0, 1, 2, 3, 0, 1], dtype=np.int64)
    eval_packet = np.asarray([3, 2], dtype=np.int64)

    encoded = gate._encode_crosscoder_codes(
        config={
            "name": "cca_kmeans3",
            "kind": "crosscoder_kmeans",
            "clusters": 3,
            "seed": 5,
            "iterations": 5,
        },
        train_source_packet_shared=train_shared,
        eval_source_packet_shared=eval_shared,
        train_packet=train_packet,
        eval_packet=eval_packet,
        train_answers=np.asarray([0, 1, 2, 3, 0, 1], dtype=np.int64),
        fit_indices=np.arange(len(train_shared), dtype=np.int64),
    )

    assert encoded["codebook_size"] == 12
    assert np.all(encoded["train_code"] % 4 == train_packet)
    assert np.all(encoded["eval_code"] % 4 == eval_packet)
    assert encoded["eval_code"].max() < encoded["codebook_size"]


def test_candidate_decoder_features_append_shared_coordinates() -> None:
    target_shared = np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3)
    codes = np.asarray([4, 9], dtype=np.int64)

    features = gate._candidate_decoder_features_with_shared(
        qwen_scores=np.zeros((2, 4), dtype=np.float64),
        qwen_target=np.zeros(2, dtype=np.int64),
        qwen_mean=np.zeros(2, dtype=np.int64),
        qwen_hybrid=np.zeros(2, dtype=np.int64),
        source_code_values=codes,
        codebook_size=12,
        target_shared=target_shared,
    )

    assert features.shape[0] == 2
    assert features.shape[1] == 4
    assert np.all(features[:, :, -4:-1] == target_shared)
    assert features[0, 2, -1] == target_shared[0, 2, 1]
    assert features[1, 3, -1] == target_shared[1, 3, 2]
