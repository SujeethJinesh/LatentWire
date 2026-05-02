import numpy as np

from scripts import build_source_private_arc_challenge_transport_common_basis_gate as gate


def test_qjl_sign_transform_reuses_matrix_and_normalizes_rows() -> None:
    values = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [-2.0, 0.5, 1.0],
            [0.0, -1.0, 4.0],
        ],
        dtype=np.float64,
    )

    projected, matrix = gate._qjl_sign_transform(values, output_dim=8, seed=11)
    projected_again, matrix_again = gate._qjl_sign_transform(
        values,
        output_dim=8,
        seed=999,
        matrix=matrix,
    )

    assert projected.shape == (3, 8)
    assert matrix.shape == (3, 8)
    assert np.allclose(matrix_again, matrix)
    assert np.allclose(projected_again, projected)
    assert np.allclose(np.linalg.norm(projected, axis=1), 1.0)
    assert set(np.unique(np.sign(projected))).issubset({-1.0, 1.0})


def test_knn_transport_recovers_training_pairs() -> None:
    source = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=np.float64,
    )
    target = np.asarray(
        [
            [10.0, 0.0, 1.0],
            [0.0, 20.0, 1.0],
            [-10.0, 0.0, 1.0],
            [0.0, -20.0, 1.0],
        ],
        dtype=np.float64,
    )

    mapper = gate._fit_knn_transport(
        source_features=source,
        target_features=target,
        fit_indices=np.arange(4),
        k=1,
        temperature=0.05,
    )
    predicted = gate._apply_knn_transport(source, mapper)

    assert predicted.shape == target.shape
    assert np.allclose(predicted, target)


def test_procrustes_transport_recovers_rotated_low_rank_map() -> None:
    rng = np.random.default_rng(13)
    latent = rng.normal(size=(24, 3))
    rotation_source, _ = np.linalg.qr(rng.normal(size=(6, 3)))
    rotation_target, _ = np.linalg.qr(rng.normal(size=(7, 3)))
    source = latent @ rotation_source.T
    target = latent @ rotation_target.T

    mapper = gate._fit_procrustes_transport(
        source_features=source,
        target_features=target,
        fit_indices=np.arange(source.shape[0]),
        dim=3,
    )
    predicted = gate._apply_procrustes_transport(source, mapper)

    cosine = np.sum(predicted * target, axis=1) / (
        np.linalg.norm(predicted, axis=1) * np.linalg.norm(target, axis=1)
    )
    assert predicted.shape == target.shape
    assert float(np.mean(cosine)) > 0.99
