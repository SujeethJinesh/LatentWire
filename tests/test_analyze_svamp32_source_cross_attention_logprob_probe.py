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
