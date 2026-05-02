from __future__ import annotations

import numpy as np

from scripts import build_source_private_hellaswag_pq_hidden_innovation_codec_gate as gate


def test_subspace_slices_cover_dimension_once() -> None:
    slices = gate._subspace_slices(dim=7, subspaces=3)

    assert [(item.start, item.stop) for item in slices] == [(0, 2), (2, 4), (4, 7)]


def test_product_quantizer_codes_subspaces_independently() -> None:
    train = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.1],
            [4.0, 4.0, 0.0, 0.0],
            [4.1, 4.0, 0.0, 0.1],
            [0.0, 0.0, 5.0, 5.0],
            [0.1, 0.0, 5.1, 5.0],
            [4.0, 4.0, 5.0, 5.0],
            [4.1, 4.0, 5.1, 5.0],
        ],
        dtype=np.float64,
    )
    quantizer = gate._fit_product_quantizer(
        train_projected=train,
        fit_indices=np.arange(len(train), dtype=np.int64),
        subspaces=2,
        clusters=2,
        seed=5,
        iterations=8,
    )
    codes = gate._apply_product_quantizer(train, quantizer)

    assert codes.shape == (len(train),)
    assert int(codes.max()) < 4
    assert len(np.unique(codes)) >= 3


def test_encode_pq_hidden_codes_preserves_candidate_low_bits() -> None:
    rng = np.random.default_rng(9)
    base = rng.normal(size=(12, 6))
    train = np.vstack([base[:6], base[:6] + 3.0])
    eval_values = np.asarray([base[0] + 0.01, base[7] + 0.01], dtype=np.float64)
    train_packet = np.asarray([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    eval_packet = np.asarray([3, 2], dtype=np.int64)

    encoded = gate._encode_pq_hidden_codes(
        config={
            "name": "pq_pca4_m2_k2_identity",
            "pca_dims": 4,
            "subspaces": 2,
            "clusters": 2,
            "rotation": "identity",
            "rotation_seed": 11,
            "seed": 13,
            "iterations": 5,
        },
        train_hidden_features=train,
        eval_hidden_features=eval_values,
        train_packet=train_packet,
        eval_packet=eval_packet,
        fit_indices=np.arange(len(train), dtype=np.int64),
    )

    assert encoded["codebook_size"] == 16
    assert np.all(encoded["train_code"] % 4 == train_packet)
    assert np.all(encoded["eval_code"] % 4 == eval_packet)
    assert encoded["train_code"].max() < encoded["codebook_size"]
    assert encoded["eval_code"].max() < encoded["codebook_size"]


def test_orthogonal_rotation_is_norm_preserving() -> None:
    rotation = gate._orthogonal_rotation(dim=5, seed=3)
    values = np.asarray([[1.0, -2.0, 0.5, 0.0, 4.0]], dtype=np.float64)

    assert np.allclose(rotation.T @ rotation, np.eye(5), atol=1e-8)
    assert np.allclose(np.linalg.norm(values @ rotation), np.linalg.norm(values), atol=1e-8)
