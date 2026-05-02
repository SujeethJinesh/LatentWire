import torch

from scripts import analyze_svamp32_source_cross_attention_logprob_probe as probe


def test_cross_attention_connector_shape() -> None:
    connector = probe.SourceCrossAttentionPrefixConnector(
        source_dim=5,
        target_dim=7,
        target_embed_dim=11,
        hidden_dim=3,
        prefix_len=2,
        use_source=True,
        use_target=True,
    )
    source = torch.randn(4, 5)
    source_mask = torch.tensor([True, True, False, False])
    target = torch.randn(6, 7)
    target_mask = torch.tensor([True, True, True, False, False, False])

    prefix = connector(source, source_mask, target, target_mask)

    assert prefix.shape == (2, 11)


def test_cross_attention_target_only_connector_shape() -> None:
    connector = probe.SourceCrossAttentionPrefixConnector(
        source_dim=5,
        target_dim=7,
        target_embed_dim=11,
        hidden_dim=3,
        prefix_len=2,
        use_source=False,
        use_target=True,
    )
    source = torch.randn(4, 5)
    source_mask = torch.tensor([True, True, False, False])
    target = torch.randn(6, 7)
    target_mask = torch.tensor([True, True, True, False, False, False])

    prefix = connector(source, source_mask, target, target_mask)

    assert prefix.shape == (2, 11)


def test_cross_attention_slots_only_ignores_inputs() -> None:
    connector = probe.SourceCrossAttentionPrefixConnector(
        source_dim=5,
        target_dim=7,
        target_embed_dim=11,
        hidden_dim=3,
        prefix_len=2,
        use_source=False,
        use_target=False,
    )
    source = torch.randn(4, 5)
    source_mask = torch.tensor([True, True, False, False])
    target = torch.randn(6, 7)
    target_mask = torch.tensor([True, True, True, False, False, False])

    first = connector(source, source_mask, target, target_mask)
    second = connector(source + 100.0, source_mask, target - 100.0, target_mask)

    assert first.shape == (2, 11)
    assert torch.equal(first, second)


def test_generation_summary_tracks_matched_only_clean_and_control_leak() -> None:
    records = {
        "matched": [
            {"example_id": "a", "correct": True, "normalized_prediction": "1", "prediction": "1"},
            {"example_id": "b", "correct": False, "normalized_prediction": "2", "prediction": "2"},
            {"example_id": "c", "correct": True, "normalized_prediction": "3", "prediction": "3"},
        ],
        "zero_source": [
            {"example_id": "a", "correct": False, "normalized_prediction": "0", "prediction": "0"},
            {"example_id": "b", "correct": True, "normalized_prediction": "2", "prediction": "2"},
            {"example_id": "c", "correct": True, "normalized_prediction": "3", "prediction": "3"},
        ],
    }

    summary = probe._summarize_generation(records, clean_ids={"a", "b"}, target_self_ids={"c"})

    assert summary["matched"]["correct_count"] == 2
    assert summary["matched"]["clean_correct_count"] == 1
    assert summary["matched"]["target_self_correct_count"] == 1
    assert summary["gate"]["matched_only_clean_ids"] == ["a"]
    assert summary["gate"]["control_leak_clean_ids"] == ["b"]
