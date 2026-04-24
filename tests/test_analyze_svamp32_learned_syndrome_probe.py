from __future__ import annotations

import torch

from scripts import analyze_svamp32_learned_syndrome_probe as probe


def test_make_outer_folds_supports_loo_and_balanced_folds() -> None:
    assert probe._make_outer_folds(3, "loo") == [[0], [1], [2]]
    assert probe._make_outer_folds(5, "2") == [[0, 2, 4], [1, 3]]


def test_crossfit_residue_predictions_returns_control_signatures() -> None:
    tokens = torch.randn(6, 4, 3)
    mask = torch.ones(6, 4, dtype=torch.bool)
    labels = {2: torch.tensor([0, 1, 0, 1, 0, 1])}
    predictions, folds = probe._crossfit_residue_predictions(
        tokens=tokens,
        mask=mask,
        labels_by_modulus=labels,
        config=probe.LearnedProbeConfig(
            moduli=(2,),
            query_count=2,
            hidden_dim=4,
            epochs=1,
            outer_folds="3",
            seed=7,
        ),
        device="cpu",
    )

    assert sorted(predictions) == [
        "label_shuffled",
        "matched",
        "same_norm_noise",
        "shuffled_source",
        "zero_source",
    ]
    assert len(folds) == 3
    assert all(len(values) == 6 for values in predictions.values())
    assert all(len(signature) == 1 for values in predictions.values() for signature in values)


def test_learned_probe_module_emits_factorized_modulus_logits() -> None:
    model = probe.SyndromeQ(input_dim=5, hidden_dim=3, query_count=2, moduli=(2, 3))
    tokens = torch.randn(4, 7, 5)
    mask = torch.ones(4, 7, dtype=torch.bool)
    logits = model(tokens, mask)

    assert sorted(logits) == [2, 3]
    assert logits[2].shape == (4, 2)
    assert logits[3].shape == (4, 3)
