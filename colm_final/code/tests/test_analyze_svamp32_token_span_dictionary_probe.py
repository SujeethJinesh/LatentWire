from __future__ import annotations

import torch

from scripts import analyze_svamp32_token_span_dictionary_probe as probe


def test_spherical_kmeans_and_dictionary_encoding_are_deterministic() -> None:
    vectors = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=torch.float32,
    )
    dictionary = probe._spherical_kmeans(vectors, atoms=2, iters=4)
    dictionary_again = probe._spherical_kmeans(vectors, atoms=2, iters=4)

    assert torch.allclose(dictionary, dictionary_again)
    assert dictionary.shape == (2, 2)
    assert torch.allclose(dictionary.norm(dim=1), torch.ones(2))

    tokens = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    mask = torch.tensor([True, True])
    code = probe._encode_dictionary(tokens, mask, dictionary, topk_atoms=1)
    assert code.shape == (2,)
    assert torch.isclose(code.norm(), torch.tensor(1.0))


def test_crossfit_dictionary_predictions_include_destructive_controls() -> None:
    tokens = torch.randn(4, 3, 5, generator=torch.Generator().manual_seed(3))
    mask = torch.ones(4, 3, dtype=torch.bool)
    sidecar = torch.randn(4, 4, generator=torch.Generator().manual_seed(4))
    profile = torch.randn(4, 2, generator=torch.Generator().manual_seed(5))
    labels = {
        2: torch.tensor([0, 1, 0, 1], dtype=torch.long),
        3: torch.tensor([0, 1, 2, 0], dtype=torch.long),
    }
    config = probe.DictionaryProbeConfig(
        moduli=(2, 3),
        outer_folds="2",
        atoms=4,
        topk_atoms=1,
        random_projection_dim=6,
        dictionary_iters=2,
    )

    predictions, metadata = probe._crossfit_dictionary_predictions(
        tokens=tokens,
        mask=mask,
        sidecar=sidecar,
        profile=profile,
        labels_by_modulus=labels,
        config=config,
    )

    assert set(predictions) == {
        "matched",
        "zero_source",
        "shuffled_source",
        "label_shuffled",
        "same_norm_noise",
        "boundary_only",
    }
    assert all(len(values) == 4 for values in predictions.values())
    assert len(metadata) == 2
    assert all("dead_atom_rate" in row for row in metadata)
