from __future__ import annotations

import json

import numpy as np

from scripts import build_source_private_hellaswag_anchor_variant_scout as scout


def _rows(count: int) -> list[scout.arc_gate.ArcRow]:
    return [
        scout.arc_gate.ArcRow(
            row_id=f"row-{index}",
            content_id=f"content-{index}",
            question="q",
            choices=("a", "b", "c", "d"),
            choice_labels=("A", "B", "C", "D"),
            answer_index=index % 4,
            answer_label=("A", "B", "C", "D")[index % 4],
        )
        for index in range(count)
    ]


def _scores(rows: list[scout.arc_gate.ArcRow]) -> list[list[float]]:
    return [[4.0, 3.0, 2.0, 1.0] for _ in rows]


def _hidden(rows: list[scout.arc_gate.ArcRow]) -> np.ndarray:
    hidden = np.zeros((len(rows), 4, 1, 6), dtype=np.float64)
    basis = np.eye(4, 4, dtype=np.float64)
    for index, row in enumerate(rows):
        hidden[index, :, 0, :4] = basis
        hidden[index, row.answer_index, 0, 4] = 2.0
        hidden[index, 0, 0, 5] = 1.0
    return hidden


def test_topk_variant_declares_anchor_order_controls_ineffective() -> None:
    assert scout._anchor_order_controls_expected_effective(
        scout.VariantSpec(name="cosine_full", kind="cosine_full")
    )
    assert not scout._anchor_order_controls_expected_effective(
        scout.VariantSpec(name="cosine_topk2", kind="cosine_topk", dim=2)
    )
    assert not scout._anchor_order_controls_expected_effective(
        scout.VariantSpec(name="rbf_topk2", kind="rbf_topk", dim=2)
    )


def test_sorted_topk_features_ignore_anchor_id_permutation() -> None:
    scores = [[0.4, 0.3, 0.2, 0.1]]
    similarities = np.asarray([[[0.1, 0.8, -0.2], [0.7, 0.2, 0.0], [0.3, 0.4, 0.9], [-0.5, 0.0, 0.6]]])
    spec = scout.VariantSpec(name="cosine_topk2", kind="cosine_topk", dim=2)

    features = scout._variant_from_similarities(
        scores=scores,
        similarities=similarities,
        spec=spec,
        params={},
        permute_anchor_ids=False,
    )
    permuted_features = scout._variant_from_similarities(
        scores=scores,
        similarities=similarities,
        spec=spec,
        params={},
        permute_anchor_ids=True,
    )

    assert np.allclose(features, permuted_features)


def test_signed_topk_hash_features_change_under_anchor_id_permutation() -> None:
    scores = [[0.4, 0.3, 0.2, 0.1]]
    similarities = np.asarray([[[0.1, 0.8, -0.2], [0.7, 0.2, 0.0], [0.3, 0.4, 0.9], [-0.5, 0.0, 0.6]]])
    spec = scout.VariantSpec(name="signed_topk_hash2x8", kind="signed_topk_hash", dim=2, bins=8, seed=811)

    features = scout._variant_from_similarities(
        scores=scores,
        similarities=similarities,
        spec=spec,
        params={},
        permute_anchor_ids=False,
    )
    permuted_features = scout._variant_from_similarities(
        scores=scores,
        similarities=similarities,
        spec=spec,
        params={},
        permute_anchor_ids=True,
    )

    assert not np.allclose(features, permuted_features)


def test_build_scout_writes_parseable_artifacts_with_mocked_inputs(tmp_path, monkeypatch) -> None:
    rows = _rows(12)
    source = tmp_path / "rows.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": row.row_id,
                    "question": row.question,
                    "choices": row.choices,
                    "answer_index": row.answer_index,
                }
            )
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    score_cache = tmp_path / "score.json"
    hidden_cache = tmp_path / "hidden.npz"
    train_score_cache = tmp_path / "train_score.json"
    train_hidden_cache = tmp_path / "train_hidden.npz"
    score_cache.write_text("{}", encoding="utf-8")
    train_score_cache.write_text("{}", encoding="utf-8")
    np.savez_compressed(hidden_cache, hidden=_hidden(rows))
    np.savez_compressed(train_hidden_cache, hidden=_hidden(rows[:8]))

    def fake_load_rows(path):
        return rows

    def fake_load_score_cache(path, *, rows):
        return _scores(rows), None, {"cache_hit": True}

    def fake_load_hidden_cache(path, *, rows):
        return _hidden(rows), {"cache_hit": True}

    def fake_sample_caches(**kwargs):
        train_rows = kwargs["train_rows"]
        return (
            _scores(train_rows),
            _hidden(train_rows),
            {"cache_hit": True},
            {"cache_hit": True},
            train_score_cache,
            train_hidden_cache,
        )

    monkeypatch.setattr(scout.arc_gate, "_load_rows", fake_load_rows)
    monkeypatch.setattr(scout.headroom, "_load_score_cache", fake_load_score_cache)
    monkeypatch.setattr(scout.top2, "_load_hidden_cache", fake_load_hidden_cache)
    monkeypatch.setattr(scout.stress, "_sample_caches", fake_sample_caches)

    payload = scout.build_scout(
        output_dir=tmp_path / "out",
        train_path=source,
        eval_path=source,
        eval_score_cache=score_cache,
        eval_hidden_cache=hidden_cache,
        train_sample_cache_dir=tmp_path,
        train_hidden_rows=8,
        train_sample_seeds=(1729,),
        split_seeds=(1729,),
        anchor_count=4,
        bootstrap_samples=10,
        variants=(
            scout.VariantSpec(name="cosine_full", kind="cosine_full"),
            scout.VariantSpec(name="signed_topk_hash2x8", kind="signed_topk_hash", dim=2, bins=8, seed=811),
            scout.VariantSpec(name="cosine_topk2", kind="cosine_topk", dim=2),
        ),
        source_lm_model="mock",
    )

    assert payload["gate"] == "source_private_hellaswag_anchor_variant_scout"
    assert payload["headline"]["raw_payload_bytes"] == 2
    topk_row = next(row for row in payload["variant_rows"] if row["variant"] == "cosine_topk2")
    assert not topk_row["anchor_order_controls_expected_effective"]
    assert topk_row["anchor_order_controls_ok"]
    signed_row = next(row for row in payload["variant_rows"] if row["variant"] == "signed_topk_hash2x8")
    assert signed_row["anchor_order_controls_expected_effective"]
    assert (tmp_path / "out" / "hellaswag_anchor_variant_scout.json").exists()
    assert (tmp_path / "out" / "variant_rows.csv").exists()
    assert (tmp_path / "out" / "predictions" / "cosine_topk2.jsonl").exists()
