from __future__ import annotations

import numpy as np

from scripts import build_source_private_hellaswag_sparse_query_receiver as gate


def test_top_abs_sparse_keeps_largest_entries_per_row() -> None:
    values = np.asarray([[1.0, -4.0, 2.0], [0.1, -0.2, 0.3]], dtype=np.float32)
    sparse = gate._top_abs_sparse(values, 2)

    assert np.allclose(sparse, np.asarray([[0.0, -4.0, 2.0], [0.0, -0.2, 0.3]]))


def test_sparse_query_basis_has_requested_normalized_rows() -> None:
    rng = np.random.default_rng(7)
    train_design = rng.normal(size=(12, 5)).astype(np.float32)
    benefit = np.asarray([1, 1, -1, -1, 0, 0, 1, -1, 0, 1, -1, 0], dtype=np.float64)
    packet = np.zeros(12, dtype=np.int64)
    alt = np.ones(12, dtype=np.int64)
    answers = np.asarray([1, 1, 0, 0, 2, 2, 1, 0, 2, 1, 0, 2], dtype=np.int64)

    basis = gate._build_sparse_query_basis(
        train_design=train_design,
        benefit=benefit,
        packet_predictions=packet,
        alt_predictions=alt,
        answers=answers,
        fit_indices=np.arange(10, dtype=np.int64),
        max_query_count=4,
        seed=11,
    )

    assert basis.shape == (4, 5)
    assert np.allclose(np.linalg.norm(basis, axis=1), 1.0, atol=1e-5)


def test_sparse_query_receiver_can_learn_synthetic_override_rule() -> None:
    train_query = np.asarray(
        [[2.0], [1.5], [-2.0], [-1.5], [1.8], [-1.8]],
        dtype=np.float64,
    )
    eval_query = np.asarray([[1.7], [-1.7]], dtype=np.float64)
    train_scalar = np.zeros((6, 2), dtype=np.float64)
    eval_scalar = np.zeros((2, 2), dtype=np.float64)
    packet_train = np.zeros(6, dtype=np.int64)
    alt_train = np.ones(6, dtype=np.int64)
    answers = np.asarray([1, 1, 0, 0, 1, 0], dtype=np.int64)
    benefit = np.asarray([1, 1, -1, -1, 1, -1], dtype=np.float64)

    receiver = gate._run_sparse_query_receiver(
        train_scalar_features=train_scalar,
        eval_scalar_features=eval_scalar,
        train_query_features=train_query,
        eval_query_features=eval_query,
        benefit=benefit,
        packet_train=packet_train,
        packet_eval=np.asarray([0, 0], dtype=np.int64),
        alt_train=alt_train,
        alt_eval=np.asarray([1, 1], dtype=np.int64),
        answers_train=answers,
        fit_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        dev_indices=np.asarray([4, 5], dtype=np.int64),
        ridges=(0.01, 0.1),
    )

    assert receiver["predictions"].tolist() == [1, 0]
