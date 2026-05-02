from __future__ import annotations

import numpy as np

from scripts import build_source_private_hellaswag_wyner_ziv_residual_packet_gate as gate


def test_packet_bit_budget_for_four_candidate_residual_sketch() -> None:
    assert gate._packet_bits_per_request(candidate_count=4, quantizer_bins=2) == 6
    assert gate._packet_bits_per_request(candidate_count=4, quantizer_bins=4) == 10
    assert gate._packet_bits_per_request(candidate_count=4, quantizer_bins=8) == 14
    assert gate._packet_bits_per_request(candidate_count=4, quantizer_bins=8) <= 16


def test_quantizer_is_fit_only_and_returns_codes_in_range() -> None:
    train = np.asarray(
        [
            [-1.0, -0.5, 0.5, 1.0],
            [-0.8, -0.2, 0.2, 0.8],
            [-2.0, -1.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )
    quantizer = gate._quantizer_from_fit(
        train_source_z=train,
        fit_indices=np.asarray([0, 1], dtype=np.int64),
        bins=4,
    )
    codes, decoded = gate._apply_quantizer(train, quantizer)

    assert codes.shape == train.shape
    assert decoded.shape == train.shape
    assert codes.min() >= 0
    assert codes.max() < 4
    assert np.all(np.diff(quantizer["edges"]) >= 0)


def test_candidate_feature_views_have_expected_relative_widths() -> None:
    packet = np.asarray([0, 2, 1], dtype=np.int64)
    sketch = np.asarray(
        [
            [-1.0, 0.0, 0.5, 1.0],
            [0.2, -0.3, 1.2, -1.1],
            [1.0, 0.4, -0.2, -0.8],
        ],
        dtype=np.float64,
    )
    qwen_scores = np.asarray(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
            [0.1, 0.5, 0.4, 0.2],
        ],
        dtype=np.float64,
    )
    common = {
        "source_packet": packet,
        "source_score_sketch": sketch,
        "qwen_scores": qwen_scores,
        "qwen_target": np.asarray([3, 0, 1], dtype=np.int64),
        "qwen_mean": np.asarray([3, 0, 1], dtype=np.int64),
        "qwen_hybrid": np.asarray([3, 0, 1], dtype=np.int64),
    }

    target = gate._candidate_feature_tensor(feature_view="target_side_only", **common)
    packet_only = gate._candidate_feature_tensor(feature_view="packet_only", **common)
    sketch_only = gate._candidate_feature_tensor(feature_view="sketch_only", **common)
    full = gate._candidate_feature_tensor(feature_view="full", **common)

    assert target.shape[:2] == (3, 4)
    assert full.shape[2] == target.shape[2] + 7
    assert packet_only.shape[2] == target.shape[2] + 5
    assert sketch_only.shape[2] == target.shape[2] + 2
    assert len(gate._candidate_feature_names("full")) == full.shape[2]


def test_candidate_decoder_learns_source_packet_rule_on_synthetic_data() -> None:
    rng = np.random.default_rng(7)
    answers = np.asarray([0, 1, 2, 3] * 8, dtype=np.int64)
    source_packet = answers.copy()
    sketch = rng.normal(size=(len(answers), 4))
    qwen_scores = rng.normal(size=(len(answers), 4))
    features = gate._candidate_feature_tensor(
        source_packet=source_packet,
        source_score_sketch=sketch,
        qwen_scores=qwen_scores,
        qwen_target=np.argmax(qwen_scores, axis=1),
        qwen_mean=np.argmax(qwen_scores, axis=1),
        qwen_hybrid=np.argmax(qwen_scores, axis=1),
        feature_view="packet_only",
    )
    fit_indices = np.arange(len(answers), dtype=np.int64)
    coef = gate._fit_candidate_decoder(
        train_features=features,
        train_answers=answers,
        fit_indices=fit_indices,
        ridge=0.01,
    )

    predictions = gate._predict_candidate_decoder(features, coef)
    assert predictions.tolist() == answers.tolist()
