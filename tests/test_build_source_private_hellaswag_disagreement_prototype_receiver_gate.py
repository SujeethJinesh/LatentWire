from __future__ import annotations

import numpy as np
import pytest

from scripts import build_source_private_hellaswag_disagreement_prototype_receiver_gate as gate


def test_spherical_prototypes_recover_two_clusters() -> None:
    points = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [-1.0, 0.0],
            [-0.9, -0.1],
        ],
        dtype=np.float64,
    )
    prototypes = gate._fit_spherical_prototypes(points, count=2, seed=7, iterations=5)
    sims = gate._normalize_rows(points) @ prototypes.T

    assert prototypes.shape == (2, 2)
    assert np.min(np.max(sims, axis=1)) > 0.98


def test_prototype_receiver_overrides_helpful_disagreement() -> None:
    train_features = np.asarray(
        [
            [2.0, 0.0],
            [2.2, 0.1],
            [-2.0, 0.0],
            [-2.1, -0.1],
            [2.1, -0.1],
            [-2.2, 0.1],
        ],
        dtype=np.float64,
    )
    eval_features = np.asarray(
        [
            [2.3, 0.0],
            [-2.3, 0.0],
        ],
        dtype=np.float64,
    )
    train_packet = np.asarray([0, 0, 0, 0, 0, 0], dtype=np.int64)
    train_alt = np.asarray([1, 1, 1, 1, 1, 1], dtype=np.int64)
    train_answers = np.asarray([1, 1, 0, 0, 1, 0], dtype=np.int64)
    benefit = np.asarray([1.0, 1.0, -1.0, -1.0, 1.0, -1.0], dtype=np.float64)
    validation_packet = np.asarray([0, 0], dtype=np.int64)
    validation_alt = np.asarray([1, 1], dtype=np.int64)
    validation_answers = np.asarray([1, 0], dtype=np.int64)
    target_predictions = validation_packet.copy()

    result = gate._run_prototype_receiver(
        train_features=train_features,
        eval_features=eval_features,
        train_benefit=benefit,
        train_packet=train_packet,
        train_alt=train_alt,
        train_answers=train_answers,
        validation_packet=validation_packet,
        validation_alt=validation_alt,
        validation_answers=validation_answers,
        fit_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        dev_indices=np.asarray([4, 5], dtype=np.int64),
        eval_indices=np.asarray([0, 1], dtype=np.int64),
        positive_count=1,
        negative_count=1,
        seed=11,
        iterations=4,
        aggregation="max",
        bootstrap_samples=50,
        bootstrap_seed=13,
        target_predictions=target_predictions,
    )

    assert result["predictions"].tolist() == [1, 0]
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["delta_vs_packet_only"] == pytest.approx(0.5)
    assert result["prototype_meta"]["positive_fit_rows"] == 2
    assert result["prototype_meta"]["negative_fit_rows"] == 2


def test_roll_predictions_stays_in_candidate_range() -> None:
    assert gate._roll_predictions(np.asarray([0, 1, 2, 3]), width=4).tolist() == [1, 2, 3, 0]
