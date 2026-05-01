from __future__ import annotations

import json

import numpy as np

from scripts import build_source_private_hellaswag_dense_residual_sketch_scout as scout


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


def test_rademacher_projection_is_deterministic_and_scaled() -> None:
    projection_a = scout._rademacher_projection(6, 8, seed=47)
    projection_b = scout._rademacher_projection(6, 8, seed=47)

    assert np.allclose(projection_a, projection_b)
    assert projection_a.shape == (6, 8)
    assert set(np.unique(np.abs(projection_a))) == {1 / np.sqrt(8)}


def test_sign_sketch_features_change_under_wrong_projection() -> None:
    rows = _rows(2)
    scores = _scores(rows)
    hidden = _hidden(rows)
    spec = scout.SketchSpec(name="qjl_sign4", kind="sign", dim=4, seed=47)
    projection = scout._rademacher_projection(6, 4, seed=47)
    wrong_projection = scout._rademacher_projection(6, 4, seed=99)

    features = scout._sketch_feature_tensor(
        scores=scores,
        hidden=hidden,
        spec=spec,
        projection=projection,
    )
    wrong_features = scout._sketch_feature_tensor(
        scores=scores,
        hidden=hidden,
        spec=spec,
        projection=wrong_projection,
    )

    assert features.shape == (2, 4, scout.repair._candidate_score_features(scores[0], 0).shape[0] + 4)
    assert not np.allclose(features, wrong_features)


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
        bootstrap_samples=10,
        sketches=(
            scout.SketchSpec(name="qjl_sign4", kind="sign", dim=4, seed=47),
            scout.SketchSpec(name="jl_value4", kind="value", dim=4, seed=53),
        ),
        source_lm_model="mock",
    )

    assert payload["gate"] == "source_private_hellaswag_dense_residual_sketch_scout"
    assert payload["headline"]["raw_payload_bytes"] == 2
    assert payload["packet_contract"]["sketch_vector_transmitted"] is False
    assert (tmp_path / "out" / "hellaswag_dense_residual_sketch_scout.json").exists()
    assert (tmp_path / "out" / "variant_rows.csv").exists()
    assert (tmp_path / "out" / "predictions" / "qjl_sign4.jsonl").exists()
