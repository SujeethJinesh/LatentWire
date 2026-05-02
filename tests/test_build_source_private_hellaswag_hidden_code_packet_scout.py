from __future__ import annotations

import pathlib

import numpy as np

from scripts import build_source_private_hellaswag_hidden_code_packet_scout as gate


def test_hidden_source_feature_matrix_normalizes_packet_and_top_differences() -> None:
    hidden = np.asarray(
        [
            [
                [[1.0, 0.0]],
                [[0.0, 1.0]],
                [[-1.0, 0.0]],
                [[0.0, -1.0]],
            ],
            [
                [[2.0, 0.0]],
                [[0.0, 2.0]],
                [[1.0, 1.0]],
                [[-1.0, -1.0]],
            ],
        ],
        dtype=np.float64,
    )
    scores = np.asarray(
        [
            [0.0, 1.0, 3.0, 2.0],
            [1.0, 4.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    packet = np.asarray([3, 0], dtype=np.int64)

    features = gate._hidden_source_feature_matrix(hidden=hidden, scores=scores, packet=packet)

    assert features.shape == (2, 4)
    assert np.allclose(np.linalg.norm(features, axis=1), 1.0)


def test_encode_hidden_codes_preserves_candidate_low_bits() -> None:
    train_features = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [5.0, 5.0, 0.0],
            [5.2, 5.0, 0.0],
            [-4.0, 3.0, 1.0],
            [-4.1, 3.1, 1.0],
        ],
        dtype=np.float64,
    )
    eval_features = np.asarray([[0.05, 0.0, 0.0], [5.1, 5.0, 0.0]], dtype=np.float64)
    train_packet = np.asarray([0, 1, 2, 3, 0, 1], dtype=np.int64)
    eval_packet = np.asarray([3, 2], dtype=np.int64)

    encoded = gate._encode_hidden_codes(
        config={
            "name": "hidden_pca2_kmeans3",
            "kind": "pca_kmeans_hidden_code",
            "pca_dims": 2,
            "clusters": 3,
            "seed": 7,
            "iterations": 5,
        },
        train_hidden_features=train_features,
        eval_hidden_features=eval_features,
        train_packet=train_packet,
        eval_packet=eval_packet,
        train_answers=np.asarray([0, 1, 2, 3, 0, 1], dtype=np.int64),
        fit_indices=np.arange(len(train_features), dtype=np.int64),
    )

    assert encoded["codebook_size"] == 12
    assert np.all(encoded["train_code"] % 4 == train_packet)
    assert np.all(encoded["eval_code"] % 4 == eval_packet)
    assert encoded["train_code"].max() < encoded["codebook_size"]
    assert encoded["eval_code"].max() < encoded["codebook_size"]


def test_reliability_hidden_codes_preserve_candidate_low_bits() -> None:
    train_features = np.asarray(
        [
            [1.0, 0.0],
            [1.1, 0.0],
            [-1.0, 0.0],
            [-1.1, 0.0],
            [0.9, 0.1],
            [-0.9, 0.1],
        ],
        dtype=np.float64,
    )
    eval_features = np.asarray([[1.05, 0.0], [-1.05, 0.0]], dtype=np.float64)
    train_packet = np.asarray([0, 1, 2, 3, 0, 1], dtype=np.int64)
    train_answers = np.asarray([0, 1, 0, 0, 0, 0], dtype=np.int64)
    eval_packet = np.asarray([3, 2], dtype=np.int64)

    encoded = gate._encode_hidden_codes(
        config={
            "name": "hidden_reliability_q2",
            "kind": "reliability_quantile_hidden_code",
            "bins": 2,
            "reliability_ridge": 0.1,
        },
        train_hidden_features=train_features,
        eval_hidden_features=eval_features,
        train_packet=train_packet,
        eval_packet=eval_packet,
        train_answers=train_answers,
        fit_indices=np.arange(len(train_features), dtype=np.int64),
    )

    assert encoded["codebook_size"] == 8
    assert np.all(encoded["train_code"] % 4 == train_packet)
    assert np.all(encoded["eval_code"] % 4 == eval_packet)
    assert encoded["encoder_audit"]["kind"] == "reliability_quantile_hidden_code"


def test_pca_is_fit_only_and_projects_eval_with_train_mean() -> None:
    train_features = np.asarray(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [4.0, 0.0],
            [100.0, 100.0],
        ],
        dtype=np.float64,
    )
    pca = gate._fit_pca(train_features, np.asarray([0, 1, 2], dtype=np.int64), dims=1)
    projected = gate._project_pca(np.asarray([[2.0, 0.0]], dtype=np.float64), pca)

    assert pca["dims"] == 1
    assert np.allclose(pca["mean"], np.asarray([2.0, 0.0]))
    assert np.allclose(projected, np.asarray([[0.0]]), atol=1e-8)


def test_slice_jsonl_writes_exact_contiguous_slice(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "slice.jsonl"
    source.write_text("\n".join(f'{{"row": {index}}}' for index in range(5)) + "\n", encoding="utf-8")

    meta = gate._slice_jsonl(source_path=source, output_path=output, start=1, count=3)

    assert output.read_text(encoding="utf-8").splitlines() == [
        '{"row": 1}',
        '{"row": 2}',
        '{"row": 3}',
    ]
    assert meta["slice_start"] == 1
    assert meta["slice_end_exclusive"] == 4
    assert meta["slice_rows"] == 3
