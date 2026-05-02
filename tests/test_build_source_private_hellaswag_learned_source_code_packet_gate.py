from __future__ import annotations

import numpy as np

from scripts import build_source_private_hellaswag_learned_source_code_packet_gate as gate


def test_packet_bytes_for_codebook() -> None:
    assert gate._packet_bytes_for_codebook(4) == 1
    assert gate._packet_bytes_for_codebook(256) == 1
    assert gate._packet_bytes_for_codebook(257) == 2


def test_source_feature_matrix_shape_and_packet_features() -> None:
    scores = np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    packet = np.asarray([3, 0], dtype=np.int64)
    features = gate._source_feature_matrix(scores, packet)

    assert features.shape == (2, 9)
    assert np.all(features[:, 6] == 0.0)
    assert np.all(features[:, 7] == 1.0)


def test_quantile_source_codes_preserve_candidate_low_bits() -> None:
    train_features = np.asarray(
        [
            [0.0] * 9,
            [1.0] * 9,
            [2.0] * 9,
            [3.0] * 9,
        ],
        dtype=np.float64,
    )
    eval_features = np.asarray([[0.2] * 9, [2.7] * 9], dtype=np.float64)
    train_packet = np.asarray([0, 1, 2, 3], dtype=np.int64)
    eval_packet = np.asarray([2, 1], dtype=np.int64)
    encoded = gate._encode_source_codes(
        config={
            "name": "packet_z_quantile_2",
            "kind": "quantile",
            "feature_name": "packet_z",
            "feature_index": 4,
            "bins": 2,
        },
        train_source_features=train_features,
        eval_source_features=eval_features,
        train_packet=train_packet,
        eval_packet=eval_packet,
        fit_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
    )

    assert encoded["codebook_size"] == 8
    assert np.all(encoded["train_code"] % 4 == train_packet)
    assert np.all(encoded["eval_code"] % 4 == eval_packet)
    assert encoded["train_code"].max() < encoded["codebook_size"]


def test_candidate_decoder_code_interaction_can_map_codes_to_answers() -> None:
    answers = np.asarray([0, 1, 2, 3] * 8, dtype=np.int64)
    code = answers.copy()
    qwen_scores = np.zeros((len(answers), 4), dtype=np.float64)
    features = gate._candidate_decoder_features(
        qwen_scores=qwen_scores,
        qwen_target=np.zeros(len(answers), dtype=np.int64),
        qwen_mean=np.zeros(len(answers), dtype=np.int64),
        qwen_hybrid=np.zeros(len(answers), dtype=np.int64),
        source_code=code,
        codebook_size=4,
    )
    coef = gate._fit_candidate_decoder(
        train_features=features,
        train_answers=answers,
        fit_indices=np.arange(len(answers), dtype=np.int64),
        ridge=0.01,
    )
    predictions = gate._predict_candidate_decoder(features, coef)

    assert predictions.tolist() == answers.tolist()
