from __future__ import annotations

import json

import numpy as np

from scripts import build_source_private_hellaswag_learned_residual_basis_scout as scout


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


def test_fit_pca_basis_uses_fit_rows_and_has_expected_shape() -> None:
    rows = _rows(8)
    scores = _scores(rows)
    hidden = _hidden(rows)
    spec = scout.BasisSpec(name="pca3", dim=3)

    params = scout._fit_pca_basis(
        rows=rows,
        scores=scores,
        hidden=hidden,
        fit_indices=[0, 1, 2, 3],
        spec=spec,
        random_seed=47,
    )

    assert params["basis"].shape == (6, 3)
    assert params["random_basis"].shape == (6, 3)
    assert params["candidate_vectors"] >= 4


def test_basis_controls_change_coefficients() -> None:
    rows = _rows(8)
    scores = _scores(rows)
    hidden = _hidden(rows)
    spec = scout.BasisSpec(name="pca4", dim=4)
    params = scout._fit_pca_basis(
        rows=rows,
        scores=scores,
        hidden=hidden,
        fit_indices=[0, 1, 2, 3],
        spec=spec,
        random_seed=47,
    )

    matched = scout._basis_coefficients(scores=scores, hidden=hidden, spec=spec, params=params)
    rolled = scout._basis_coefficients(
        scores=scores,
        hidden=hidden,
        spec=spec,
        params=params,
        control="basis_dim_roll",
    )
    random_basis = scout._basis_coefficients(
        scores=scores,
        hidden=hidden,
        spec=spec,
        params=params,
        control="random_basis_same_dim",
    )

    assert matched.shape == (8, 4, 4)
    assert not np.allclose(matched, rolled)
    assert not np.allclose(matched, random_basis)


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
        bases=(
            scout.BasisSpec(name="pca3", dim=3),
            scout.BasisSpec(name="pca3_whiten", dim=3, whiten=True),
        ),
        source_lm_model="mock",
    )

    assert payload["gate"] == "source_private_hellaswag_learned_residual_basis_scout"
    assert payload["headline"]["raw_payload_bytes"] == 2
    assert payload["packet_contract"]["basis_coefficients_transmitted"] is False
    assert (tmp_path / "out" / "hellaswag_learned_residual_basis_scout.json").exists()
    assert (tmp_path / "out" / "variant_rows.csv").exists()
    assert (tmp_path / "out" / "predictions" / "pca3.jsonl").exists()
