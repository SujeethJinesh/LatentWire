from __future__ import annotations

import json

import numpy as np

from scripts import build_source_private_hellaswag_anchor_relative_hidden_innovation_gate as gate


def _row(index: int) -> object:
    return type(
        "Row",
        (),
        {
            "row_id": f"row-{index}",
            "answer_index": index % 4,
        },
    )()


def test_anchor_relative_features_change_under_anchor_id_permutation() -> None:
    scores = [[0.4, 0.3, 0.2, 0.1]]
    hidden = np.zeros((1, 4, 1, 3), dtype=np.float64)
    hidden[0, :, 0, :] = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    anchors = np.eye(3, dtype=np.float64)

    features = gate._anchor_relative_feature_tensor(scores=scores, hidden=hidden, anchors=anchors)
    score_dim = gate.repair._candidate_score_features(scores[0], 0).shape[0]
    permuted = features.copy()
    permuted[:, :, score_dim:] = np.roll(permuted[:, :, score_dim:], 1, axis=2)

    assert features.shape == (1, 4, score_dim + 3)
    assert not np.allclose(features[:, :, score_dim:], permuted[:, :, score_dim:])


def test_fit_anchor_bank_uses_fit_rows_only_and_writes_metadata() -> None:
    scores = [
        [0.9, 0.1, 0.0, -0.1],
        [0.1, 0.9, 0.0, -0.1],
    ]
    hidden = np.zeros((2, 4, 1, 4), dtype=np.float64)
    for row in range(2):
        for candidate in range(4):
            hidden[row, candidate, 0, candidate] = 1.0 + row
    anchors, meta = gate._fit_anchor_bank(
        rows=[_row(0), _row(1)],
        scores=scores,
        hidden=hidden,
        fit_indices=[0],
        anchor_count=2,
    )

    assert anchors.shape[0] <= 2
    assert anchors.shape[1] == 4
    assert meta["candidate_vectors"] >= 1
    assert meta["selection"] == "norm_descending_then_farthest_first_cosine"


def test_build_gate_writes_parseable_artifacts_with_mocked_inputs(tmp_path, monkeypatch) -> None:
    rows = [
        gate.arc_gate.ArcRow(
            row_id=f"row-{index}",
            content_id=f"content-{index}",
            question="q",
            choices=("a", "b", "c", "d"),
            choice_labels=("A", "B", "C", "D"),
            answer_index=index % 4,
            answer_label=("A", "B", "C", "D")[index % 4],
        )
        for index in range(12)
    ]
    scores = [[4.0, 3.0, 2.0, 1.0] for _ in rows]
    hidden = np.zeros((len(rows), 4, 1, 6), dtype=np.float64)
    for index, row in enumerate(rows):
        hidden[index, row.answer_index, 0, 0] = 2.0
        hidden[index, 0, 0, 1] = 1.0
        hidden[index, :, 0, 2:] = np.eye(4, 4)[:4]

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

    def fake_load_rows(path):
        return rows

    def fake_load_score_cache(path, *, rows):
        return scores, None, {"cache_hit": True}

    def fake_load_hidden_cache(path, *, rows):
        return hidden, {"cache_hit": True}

    def fake_sample_caches(**kwargs):
        n = len(kwargs["train_rows"])
        return scores[:n], hidden[:n], {"cache_hit": True}, {"cache_hit": True}, tmp_path / "s.json", tmp_path / "h.npz"

    monkeypatch.setattr(gate.arc_gate, "_load_rows", fake_load_rows)
    monkeypatch.setattr(gate.headroom, "_load_score_cache", fake_load_score_cache)
    monkeypatch.setattr(gate.top2, "_load_hidden_cache", fake_load_hidden_cache)
    monkeypatch.setattr(gate.stress, "_sample_caches", fake_sample_caches)
    (tmp_path / "s.json").write_text("{}", encoding="utf-8")
    np.savez_compressed(tmp_path / "h.npz", hidden=hidden)
    (tmp_path / "score.json").write_text("{}", encoding="utf-8")
    np.savez_compressed(tmp_path / "hidden.npz", hidden=hidden)

    payload = gate.build_gate(
        output_dir=tmp_path / "out",
        train_path=source,
        eval_path=source,
        eval_score_cache=tmp_path / "score.json",
        eval_hidden_cache=tmp_path / "hidden.npz",
        train_sample_cache_dir=tmp_path,
        train_hidden_rows=8,
        train_sample_seeds=(1729, 2027, 2039),
        split_seeds=(1729,),
        anchor_count=4,
        bootstrap_samples=10,
    )

    assert payload["gate"] == "source_private_hellaswag_anchor_relative_hidden_innovation_gate"
    assert payload["headline"]["raw_payload_bytes"] == 2
    assert (tmp_path / "out" / "hellaswag_anchor_relative_hidden_innovation_gate.json").exists()
    assert (tmp_path / "out" / "anchor_banks.npz").exists()
