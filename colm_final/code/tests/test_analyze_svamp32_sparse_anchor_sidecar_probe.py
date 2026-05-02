from __future__ import annotations

import torch

from scripts import analyze_svamp32_sparse_anchor_sidecar_probe as probe


class _FakeTokenizer:
    def __init__(self, step: int) -> None:
        self.step = int(step)

    def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool):
        del add_special_tokens, return_offsets_mapping
        offsets = []
        for start in range(0, len(text), self.step):
            offsets.append((start, min(len(text), start + self.step)))
        return {"offset_mapping": offsets}


def test_sequence_alignment_sidecar_is_deterministic() -> None:
    texts = ["alpha beta", "gamma delta"]
    source = _FakeTokenizer(step=2)
    target = _FakeTokenizer(step=3)

    features, profiles, metadata = probe._sequence_alignment_sidecar(
        texts,
        source_tokenizer=source,
        target_tokenizer=target,
        dim=8,
        token_scale=0.75,
    )
    features_again, profiles_again, metadata_again = probe._sequence_alignment_sidecar(
        texts,
        source_tokenizer=source,
        target_tokenizer=target,
        dim=8,
        token_scale=0.75,
    )

    assert torch.equal(features, features_again)
    assert torch.equal(profiles, profiles_again)
    assert metadata == metadata_again
    assert features.shape == (2, 8)
    assert profiles.shape == (2, 6)
    assert all(0.0 <= row["boundary_f1"] <= 1.0 for row in metadata)


def test_sparse_anchor_code_is_rate_capped_and_deterministic() -> None:
    features = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    code = probe._sparse_anchor_code(features, code_dim=10, topk=3, seed=5)
    code_again = probe._sparse_anchor_code(features, code_dim=10, topk=3, seed=5)

    assert torch.equal(code, code_again)
    assert code.shape == (3, 10)
    assert all(value <= 3 for value in (code != 0).sum(dim=1).tolist())
    nonzero_rows = code.norm(dim=1) > 0
    assert torch.allclose(code[nonzero_rows].norm(dim=1), torch.ones(int(nonzero_rows.sum())))
