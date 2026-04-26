from __future__ import annotations

import torch

from scripts import analyze_svamp32_target_query_source_bottleneck_probe as probe


def test_target_query_source_bottleneck_emits_modulus_logits() -> None:
    model = probe.TargetQuerySourceBottleneck(
        source_dim=5,
        target_dim=7,
        hidden_dim=3,
        query_count=2,
        moduli=(2, 3),
    )
    source_tokens = torch.randn(4, 6, 5)
    source_mask = torch.ones(4, 6, dtype=torch.bool)
    target_tokens = torch.randn(4, 8, 7)
    target_mask = torch.ones(4, 8, dtype=torch.bool)

    logits = model(source_tokens, source_mask, target_tokens, target_mask)

    assert sorted(logits) == [2, 3]
    assert logits[2].shape == (4, 2)
    assert logits[3].shape == (4, 3)


def test_crossfit_target_query_predictions_include_source_independent_controls() -> None:
    source_tokens = torch.randn(6, 4, 3)
    source_mask = torch.ones(6, 4, dtype=torch.bool)
    target_tokens = torch.randn(6, 5, 2)
    target_mask = torch.ones(6, 5, dtype=torch.bool)
    labels = {2: torch.tensor([0, 1, 0, 1, 0, 1])}

    predictions, folds = probe._crossfit_target_query_predictions(
        source_tokens=source_tokens,
        source_mask=source_mask,
        target_tokens=target_tokens,
        target_mask=target_mask,
        labels_by_modulus=labels,
        config=probe.TargetQueryConfig(
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
        "projected_soft_prompt",
        "same_norm_noise",
        "shuffled_source",
        "target_only_prefix",
        "zero_source",
    ]
    assert len(folds) == 3
    assert all(len(values) == 6 for values in predictions.values())
    assert all(len(signature) == 1 for values in predictions.values() for signature in values)
