from __future__ import annotations

import numpy as np

from scripts import build_source_private_hellaswag_anchor_relative_hidden_code_scout as gate


def test_candidate_hidden_feature_tensor_shape_and_norms() -> None:
    hidden = np.asarray(
        [
            [
                [[1.0, 0.0]],
                [[0.0, 1.0]],
                [[-1.0, 0.0]],
                [[0.0, -1.0]],
            ],
            [
                [[2.0, 0.0]],
                [[0.0, 2.0]],
                [[1.0, 1.0]],
                [[-1.0, -1.0]],
            ],
        ],
        dtype=np.float64,
    )
    scores = np.asarray([[0.0, 1.0, 3.0, 2.0], [1.0, 4.0, 2.0, 0.0]], dtype=np.float64)
    features = gate._candidate_hidden_feature_tensor(
        hidden=hidden,
        scores=scores,
        reference_prediction=np.asarray([2, 1], dtype=np.int64),
    )

    assert features.shape == (2, 4, 5)
    assert np.allclose(np.linalg.norm(features, axis=2), 1.0)


def test_select_anchor_indices_stratifies_and_is_deterministic() -> None:
    fit_indices = np.arange(8, dtype=np.int64)
    packet = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    qwen = np.asarray([0, 1, 1, 2, 2, 3, 3, 0], dtype=np.int64)
    answers = np.asarray([0, 1, 1, 3, 2, 0, 3, 1], dtype=np.int64)

    first = gate._select_anchor_indices(
        fit_indices=fit_indices,
        train_packet=packet,
        qwen_target=qwen,
        train_answers=answers,
        anchor_count=6,
        seed=17,
    )
    second = gate._select_anchor_indices(
        fit_indices=fit_indices,
        train_packet=packet,
        qwen_target=qwen,
        train_answers=answers,
        anchor_count=6,
        seed=17,
    )

    assert first.tolist() == second.tolist()
    assert len(set(first.tolist())) == 6
    assert set(first.tolist()).issubset(set(fit_indices.tolist()))


def test_nearest_anchor_codes_preserve_candidate_low_bits() -> None:
    train_relative = np.asarray(
        [
            [1.0, 0.0, -1.0],
            [0.0, 1.0, -1.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    eval_relative = np.asarray([[0.9, 0.0, -0.5], [-1.0, 0.1, 0.8]], dtype=np.float64)
    train_packet = np.asarray([0, 1, 2], dtype=np.int64)
    eval_packet = np.asarray([3, 1], dtype=np.int64)

    encoded = gate._encode_relative_codes(
        config={"name": "anchor3_nearest", "kind": "nearest_anchor", "anchor_count": 3},
        train_relative=train_relative,
        eval_relative=eval_relative,
        train_packet=train_packet,
        eval_packet=eval_packet,
        train_answers=np.asarray([0, 0, 2], dtype=np.int64),
        fit_indices=np.arange(3, dtype=np.int64),
    )

    assert encoded["codebook_size"] == 12
    assert np.all(encoded["train_code"] % 4 == train_packet)
    assert np.all(encoded["eval_code"] % 4 == eval_packet)
    assert encoded["eval_code"].tolist() == [3, 9]


def test_candidate_decoder_features_append_relative_and_selected_similarity() -> None:
    qwen_scores = np.zeros((2, 4), dtype=np.float64)
    codes = np.asarray([4, 9], dtype=np.int64)
    target_relative = np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3)
    features = gate._candidate_decoder_features_with_relative(
        qwen_scores=qwen_scores,
        qwen_target=np.zeros(2, dtype=np.int64),
        qwen_mean=np.zeros(2, dtype=np.int64),
        qwen_hybrid=np.zeros(2, dtype=np.int64),
        source_code_values=codes,
        codebook_size=12,
        target_relative=target_relative,
    )

    assert features.shape[0] == 2
    assert features.shape[1] == 4
    assert features.shape[2] > target_relative.shape[2]
    assert np.all(features[:, :, -4:-1] == target_relative)
    assert features[0, 2, -1] == target_relative[0, 2, 1]
    assert features[1, 3, -1] == target_relative[1, 3, 2]
