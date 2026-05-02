from __future__ import annotations

import numpy as np

from scripts import build_source_private_hellaswag_conditional_selector_syndrome_gate as gate


def test_packet_bytes_for_codebook_keeps_one_byte_until_256_symbols() -> None:
    assert gate._packet_bytes_for_codebook(4) == 1
    assert gate._packet_bytes_for_codebook(256) == 1
    assert gate._packet_bytes_for_codebook(257) == 2


def test_source_code_preserves_candidate_low_bits() -> None:
    train_scores = np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
            [0.0, 3.0, 2.0, 1.0],
            [1.0, 0.0, 3.0, 2.0],
        ],
        dtype=np.float64,
    )
    eval_scores = train_scores[::-1].copy()
    train_packet = np.asarray([3, 0, 1, 2], dtype=np.int64)
    eval_packet = np.asarray([2, 1, 0, 3], dtype=np.int64)

    encoded = gate._fit_source_code(
        kind="quantile",
        feature_name="packet_z",
        bins=4,
        train_scores=train_scores,
        eval_scores=eval_scores,
        train_packet=train_packet,
        eval_packet=eval_packet,
        fit_indices=np.arange(4, dtype=np.int64),
    )

    assert encoded["codebook_size"] == 16
    assert np.all(encoded["train_code"] % 4 == train_packet)
    assert np.all(encoded["eval_code"] % 4 == eval_packet)


def test_selector_feature_matrix_changes_with_source_code() -> None:
    qwen_scores = np.asarray([[1.0, 0.0, 2.0, 3.0], [0.0, 4.0, 1.0, 2.0]], dtype=np.float64)
    packet = np.asarray([0, 1], dtype=np.int64)
    alternative = np.asarray([3, 2], dtype=np.int64)
    code_a = np.asarray([0, 1], dtype=np.int64)
    code_b = np.asarray([4, 5], dtype=np.int64)

    features_a = gate._selector_feature_matrix(
        qwen_scores=qwen_scores,
        packet=packet,
        alternative=alternative,
        source_code=code_a,
        codebook_size=8,
    )
    features_b = gate._selector_feature_matrix(
        qwen_scores=qwen_scores,
        packet=packet,
        alternative=alternative,
        source_code=code_b,
        codebook_size=8,
    )

    assert features_a.shape == features_b.shape
    assert not np.allclose(features_a, features_b)


def test_ridge_selector_can_learn_override_benefit() -> None:
    qwen_scores = np.zeros((8, 4), dtype=np.float64)
    packet = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    alternative = np.asarray([2, 2, 2, 2, 3, 3, 3, 3], dtype=np.int64)
    source_code = np.asarray([0, 4, 0, 4, 1, 5, 1, 5], dtype=np.int64)
    answers = np.asarray([0, 2, 0, 2, 1, 3, 1, 3], dtype=np.int64)
    targets = (alternative == answers).astype(np.float64) - (packet == answers).astype(np.float64)
    features = gate._selector_feature_matrix(
        qwen_scores=qwen_scores,
        packet=packet,
        alternative=alternative,
        source_code=source_code,
        codebook_size=8,
    )
    coef = gate._fit_ridge(features, targets, np.arange(8, dtype=np.int64), ridge=0.01)
    scores = gate._predict_score(features, coef)
    predictions = gate._threshold_predictions(
        packet=packet,
        alternative=alternative,
        benefit_scores=scores,
        threshold=0.0,
    )

    assert predictions.tolist() == answers.tolist()
