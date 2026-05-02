from __future__ import annotations

import numpy as np

from scripts import build_source_private_hellaswag_nonlinear_selector_syndrome_gate as gate


def test_standardize_fit_eval_uses_fit_rows_only() -> None:
    train = np.asarray(
        [
            [1.0, 10.0],
            [3.0, 10.0],
            [100.0, 10.0],
        ],
        dtype=np.float64,
    )
    eval_features = np.asarray([[5.0, 10.0]], dtype=np.float64)

    train_std, eval_std, standardizer = gate._standardize_fit_eval(
        train,
        eval_features,
        np.asarray([0, 1], dtype=np.int64),
    )

    assert np.allclose(train_std[:2, 0], [-1.0, 1.0])
    assert np.allclose(train_std[:2, 1], [0.0, 0.0])
    assert np.allclose(eval_std[0, 0], 3.0)
    assert np.allclose(gate._standardize_apply(eval_features, standardizer), eval_std)


def test_rff_parameters_are_deterministic_and_seed_sensitive() -> None:
    params_a = gate._rff_parameters(3, components=5, gamma=0.7, seed=11)
    params_b = gate._rff_parameters(3, components=5, gamma=0.7, seed=11)
    params_c = gate._rff_parameters(3, components=5, gamma=0.7, seed=12)

    assert np.allclose(params_a["weights"], params_b["weights"])
    assert np.allclose(params_a["bias"], params_b["bias"])
    assert not np.allclose(params_a["weights"], params_c["weights"])


def test_nonlinear_feature_matrix_adds_bias_and_rff_components() -> None:
    features = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    params = gate._rff_parameters(2, components=7, gamma=1.0, seed=5)

    nonlinear = gate._nonlinear_feature_matrix(features, params)

    assert nonlinear.shape == (2, 1 + 2 + 7)
    assert np.allclose(nonlinear[:, 0], 1.0)


def test_benefit_targets_weight_harms() -> None:
    packet = np.asarray([0, 0, 1, 2], dtype=np.int64)
    alternative = np.asarray([1, 0, 2, 3], dtype=np.int64)
    answers = np.asarray([1, 0, 1, 0], dtype=np.int64)

    targets = gate._benefit_targets(
        alternative=alternative,
        packet=packet,
        answers=answers,
        harm_weight=2.0,
    )

    assert targets.tolist() == [1.0, 0.0, -2.0, 0.0]


def test_nonlinear_features_can_fit_xor_like_override_rule() -> None:
    base = np.asarray(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    params = gate._rff_parameters(2, components=64, gamma=1.0, seed=17)
    features = gate._nonlinear_feature_matrix(base, params)
    targets = np.asarray([-1.0, 1.0, 1.0, -1.0], dtype=np.float64)

    coef = gate.linear._fit_ridge(features, targets, np.arange(4, dtype=np.int64), ridge=1e-4)
    scores = gate.linear._predict_score(features, coef)

    assert np.sign(scores).tolist() == np.sign(targets).tolist()
