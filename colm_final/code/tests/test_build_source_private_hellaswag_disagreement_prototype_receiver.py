from __future__ import annotations

import numpy as np

from scripts import build_source_private_hellaswag_disagreement_prototype_receiver as gate


def test_disagreement_prototype_receiver_uses_local_beneficial_cluster() -> None:
    train_features = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [4.0, 4.0],
            [4.1, 4.0],
            [4.0, 4.1],
            [4.1, 4.1],
        ],
        dtype=np.float64,
    )
    eval_features = np.asarray([[4.05, 4.05], [0.05, 0.05]], dtype=np.float64)
    packet_train = np.asarray([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    alt_train = np.asarray([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    answers_train = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    packet_eval = np.asarray([0, 0], dtype=np.int64)
    alt_eval = np.asarray([1, 1], dtype=np.int64)

    receiver = gate._run_prototype_receiver(
        train_features=train_features,
        eval_features=eval_features,
        packet_train=packet_train,
        packet_eval=packet_eval,
        alt_train=alt_train,
        alt_eval=alt_eval,
        answers_train=answers_train,
        fit_indices=np.asarray([0, 1, 2, 4, 5, 6], dtype=np.int64),
        dev_indices=np.asarray([3, 7], dtype=np.int64),
        prototype_count=2,
        neighbor_k=3,
        top_k=1,
    )

    assert receiver["predictions"].tolist() == [1, 0]
    assert receiver["eval_override_rate"] == 0.5
    assert receiver["actual_prototype_count"] == 2


def test_random_indices_are_deterministic_and_bounded() -> None:
    pool = np.arange(10, dtype=np.int64)
    first = gate._random_indices(pool, 4, seed=7)
    second = gate._random_indices(pool, 4, seed=7)

    assert first.tolist() == second.tolist()
    assert len(first) == 4
    assert set(first.tolist()).issubset(set(pool.tolist()))


def test_farthest_first_indices_respects_requested_count() -> None:
    features = np.asarray([[0.0], [1.0], [2.0], [10.0]], dtype=np.float64)
    selected = gate._farthest_first_indices(features, np.arange(4, dtype=np.int64), 2)

    assert len(selected) == 2
    assert set(selected.tolist()).issubset({0, 1, 2, 3})
