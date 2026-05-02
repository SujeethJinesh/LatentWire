import numpy as np

from scripts import build_source_private_arc_challenge_sparse_query_cache_bottleneck_gate as gate


def test_rff_parameters_are_reusable_and_shape_stable() -> None:
    values = np.asarray([[0.0, 1.0], [1.0, -1.0], [2.0, 0.5]], dtype=np.float64)
    params = gate._rff_parameters(input_dim=2, components=7, seed=5, gamma=0.75)

    first = gate._apply_rff(values, params)
    second = gate._apply_rff(values, params)

    assert params["omega"].shape == (2, 7)
    assert params["phase"].shape == (7,)
    assert first.shape == (3, 7)
    assert np.allclose(first, second)


def test_topk_sparse_keeps_largest_absolute_components() -> None:
    values = np.asarray([[1.0, -3.0, 2.0, 0.5], [-0.1, 0.2, -0.3, 0.4]], dtype=np.float64)

    sparse = gate._topk_sparse(values, active_components=2)

    assert np.allclose(sparse[0], [0.0, -3.0, 2.0, 0.0])
    assert np.allclose(sparse[1], [0.0, 0.0, -0.3, 0.4])


def test_sparse_query_map_learns_synthetic_nonlinear_target() -> None:
    rng = np.random.default_rng(19)
    source = rng.normal(size=(80, 5))
    target = np.column_stack(
        [
            np.cos(source[:, 0] - 0.5 * source[:, 1]),
            np.cos(0.25 + source[:, 2]),
            np.cos(source[:, 3] + source[:, 4]),
        ]
    )
    mapper = gate._fit_sparse_query_map(
        source_features=source,
        target_features=target,
        fit_indices=np.arange(source.shape[0]),
        pca_dim=5,
        rff_components=64,
        active_components=64,
        gamma=1.0,
        ridge=1e-3,
        seed=23,
    )
    predicted = gate._apply_sparse_query_map(source, mapper)
    corr = np.corrcoef(predicted.ravel(), target.ravel())[0, 1]

    assert mapper["sparsity_fraction"] == 1.0
    assert predicted.shape == target.shape
    assert corr > 0.95
