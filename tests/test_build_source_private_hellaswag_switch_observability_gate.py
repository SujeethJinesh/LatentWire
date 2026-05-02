from __future__ import annotations

import numpy as np

from scripts import build_source_private_hellaswag_switch_observability_gate as gate


def test_roc_auc_handles_perfect_and_reversed_scores() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    perfect = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    reversed_scores = -perfect

    assert gate._roc_auc(perfect, labels) == 1.0
    assert gate._roc_auc(reversed_scores, labels) == 0.0


def test_roc_auc_handles_ties() -> None:
    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    scores = np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float64)

    assert gate._roc_auc(scores, labels) == 0.5


def test_average_precision_prefers_positive_first() -> None:
    labels = np.asarray([1, 0, 1, 0], dtype=np.int64)
    scores = np.asarray([0.9, 0.8, 0.7, 0.1], dtype=np.float64)

    assert np.isclose(gate._average_precision(scores, labels), (1.0 + 2.0 / 3.0) / 2.0)


def test_feature_matrix_views_have_expected_ordering() -> None:
    source_scores = np.asarray([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    qwen_scores = np.asarray([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
    packet = np.asarray([3, 0], dtype=np.int64)
    alternative = np.asarray([0, 3], dtype=np.int64)

    packet_only, packet_names = gate._feature_matrix(
        view="packet_id_only",
        source_scores=source_scores,
        qwen_scores=qwen_scores,
        packet=packet,
        alternative=alternative,
    )
    source_only, source_names = gate._feature_matrix(
        view="source_score_only",
        source_scores=source_scores,
        qwen_scores=qwen_scores,
        packet=packet,
        alternative=alternative,
    )
    combined, combined_names = gate._feature_matrix(
        view="source_plus_qwen",
        source_scores=source_scores,
        qwen_scores=qwen_scores,
        packet=packet,
        alternative=alternative,
    )

    assert packet_only.shape[1] == len(packet_names)
    assert source_only.shape[1] == len(source_names)
    assert combined.shape[1] == len(combined_names)
    assert packet_only.shape[1] < source_only.shape[1] < combined.shape[1]


def test_benefit_targets_mark_help_and_harm() -> None:
    packet = np.asarray([0, 0, 1, 2], dtype=np.int64)
    alternative = np.asarray([1, 0, 2, 3], dtype=np.int64)
    answers = np.asarray([1, 0, 1, 0], dtype=np.int64)

    assert gate._benefit_targets(packet=packet, alternative=alternative, answers=answers).tolist() == [
        1.0,
        0.0,
        -1.0,
        0.0,
    ]


def test_select_threshold_finds_helpful_override() -> None:
    scores = np.asarray([1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    packet = np.asarray([0, 0, 1, 1], dtype=np.int64)
    alternative = np.asarray([2, 2, 3, 3], dtype=np.int64)
    answers = np.asarray([2, 0, 1, 3], dtype=np.int64)

    selected = gate._select_threshold(
        scores=scores,
        packet=packet,
        alternative=alternative,
        answers=answers,
        indices=np.arange(4, dtype=np.int64),
    )

    assert selected["delta_vs_packet"] == 0.5
    assert selected["help_count"] == 2
    assert selected["harm_count"] == 0
