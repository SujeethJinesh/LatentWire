import pytest
import torch

from scripts import analyze_svamp32_source_soft_prefix_logprob_probe as probe


def test_source_soft_prefix_connector_shapes() -> None:
    source = torch.randn(5)
    target = torch.randn(7)
    connector = probe.SourceSoftPrefixConnector(
        source_dim=5,
        target_dim=7,
        target_embed_dim=11,
        hidden_dim=3,
        prefix_len=2,
        use_source=True,
        use_target=True,
    )

    prefix = connector(source, target)

    assert prefix.shape == (2, 11)


def test_slots_only_connector_ignores_inputs() -> None:
    source = torch.randn(5)
    target = torch.randn(7)
    connector = probe.SourceSoftPrefixConnector(
        source_dim=5,
        target_dim=7,
        target_embed_dim=11,
        hidden_dim=3,
        prefix_len=2,
        use_source=False,
        use_target=False,
    )

    first = connector(source, target)
    second = connector(source + 100.0, target - 100.0)

    assert first.shape == (2, 11)
    assert torch.equal(first, second)


def test_summarize_counts_clean_matched_only_and_control_leaks() -> None:
    rows = [
        {
            "example_id": "clean-win",
            "matched_margin": 0.4,
            "best_control_margin": -0.1,
            "matched_minus_best_control_margin": 0.5,
        },
        {
            "example_id": "clean-leak",
            "matched_margin": 0.2,
            "best_control_margin": 0.3,
            "matched_minus_best_control_margin": -0.1,
        },
        {
            "example_id": "target-correct",
            "matched_margin": 0.1,
            "best_control_margin": 0.0,
            "matched_minus_best_control_margin": 0.1,
        },
    ]

    summary = probe._summarize(
        rows,
        clean_ids={"clean-win", "clean-leak"},
        target_self_ids={"target-correct"},
        min_margin_delta=0.0,
    )

    assert summary["clean_ids_scored"] == 2
    assert summary["matched_only_clean_count"] == 1
    assert summary["matched_only_clean_ids"] == ["clean-win"]
    assert summary["control_leak_clean_count"] == 1
    assert summary["control_leak_clean_ids"] == ["clean-leak"]
    assert summary["target_self_matched_positive_count"] == 1


def test_target_set_ids_uses_first_nonempty_key() -> None:
    assert probe._target_set_ids(
        {"ids": {"clean_residual_targets": [], "clean_teacher_only": ["a", 2]}},
        "clean_residual_targets",
        "clean_teacher_only",
    ) == ["a", "2"]


def test_answer_continuation_requires_placeholder() -> None:
    with pytest.raises(ValueError, match="must contain"):
        probe._answer_continuation("5", "Answer:")
