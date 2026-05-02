from __future__ import annotations

import json

import numpy as np

from scripts import build_source_private_hellaswag_top2_contrastive_repair_probe as probe
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate


def _write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _rows(prefix: str, count: int) -> list[dict[str, object]]:
    rows = []
    for index in range(count):
        answer_index = 1 if index % 2 else 0
        rows.append(
            {
                "id": f"{prefix}-{index}",
                "content_id": f"{prefix}-content-{index}",
                "question": "Shared context",
                "choices": ("choice A", "choice B", "choice C", "choice D"),
                "choice_labels": ("A", "B", "C", "D"),
                "answer_index": answer_index,
                "answer_label": "B" if answer_index == 1 else "A",
            }
        )
    return rows


def _content_digest(rows: list[arc_gate.ArcRow]) -> str:
    return probe._content_digest(rows)


def _write_score_cache(path, rows: list[arc_gate.ArcRow]) -> None:
    scores = [[4.0, 3.0, 0.0, -1.0] for _ in rows]
    payload = {
        "row_count": len(rows),
        "row_ids": [row.row_id for row in rows],
        "content_digest": _content_digest(rows),
        "source_scores": scores,
        "source_predictions": [0 for _ in rows],
        "source_model": {"kind": "fixture", "latency_s": 0.0},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_hidden_cache(path, rows: list[arc_gate.ArcRow]) -> None:
    features = np.zeros((len(rows), 4, 1, 2), dtype=np.float32)
    for row_index, row in enumerate(rows):
        sign = 1.0 if row.answer_index == 1 else -1.0
        features[row_index, 1, 0, 0] = sign
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=features)
    metadata = {
        "row_count": len(rows),
        "row_ids": [row.row_id for row in rows],
        "content_digest": _content_digest(rows),
        "layers": [-1],
        "hidden_dim": 2,
        "model_path": "fixture",
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _fixture(tmp_path):
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_jsonl(train_path, _rows("train", 12))
    _write_jsonl(eval_path, _rows("eval", 8))
    train_rows = arc_gate._load_rows(train_path)
    eval_rows = arc_gate._load_rows(eval_path)
    train_score = tmp_path / "train_scores.json"
    eval_score = tmp_path / "eval_scores.json"
    train_hidden = tmp_path / "train_hidden.npz"
    eval_hidden = tmp_path / "eval_hidden.npz"
    _write_score_cache(train_score, train_rows)
    _write_score_cache(eval_score, eval_rows)
    _write_hidden_cache(train_hidden, train_rows)
    _write_hidden_cache(eval_hidden, eval_rows)
    return {
        "train_path": train_path,
        "eval_path": eval_path,
        "train_score_cache": train_score,
        "eval_score_cache": eval_score,
        "train_hidden_cache": train_hidden,
        "eval_hidden_cache": eval_hidden,
    }


def test_top2_contrastive_probe_promotes_when_hidden_switch_signal_is_real(tmp_path) -> None:
    paths = _fixture(tmp_path)
    payload = probe.build_probe(
        output_dir=tmp_path / "out",
        train_hidden_rows=12,
        selection_seed=11,
        dev_fraction=0.25,
        hidden_layer_index=-1,
        public_feature_dim=16,
        ridges=(0.1, 1.0),
        bootstrap_samples=50,
        run_date="2026-05-01",
        **paths,
    )

    assert payload["pass_gate"] is True
    assert payload["headline"]["selected_view"].startswith("hidden")
    assert payload["headline"]["selected_eval_accuracy"] == 1.0
    assert payload["headline"]["source_label_copy_eval_accuracy"] == 0.5
    assert payload["headline"]["selected_minus_best_label_copy"] >= 0.02
    assert payload["packet_contract"]["raw_payload_bytes"] == 2
    assert payload["packet_contract"]["source_text_exposed"] is False
    assert payload["packet_contract"]["raw_hidden_vector_transmitted"] is False


def test_top2_contrastive_probe_writes_outputs(tmp_path) -> None:
    paths = _fixture(tmp_path)
    probe.build_probe(
        output_dir=tmp_path / "out",
        train_hidden_rows=12,
        selection_seed=11,
        dev_fraction=0.25,
        hidden_layer_index=-1,
        public_feature_dim=16,
        ridges=(0.1,),
        bootstrap_samples=20,
        run_date="2026-05-01",
        **paths,
    )

    assert (tmp_path / "out" / "hellaswag_top2_contrastive_repair_probe.json").exists()
    assert (tmp_path / "out" / "hellaswag_top2_contrastive_repair_probe.md").exists()
    assert (tmp_path / "out" / "candidate_readouts.jsonl").exists()
    prediction_rows = (tmp_path / "out" / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(prediction_rows) == 8
    assert json.loads(prediction_rows[0])["source_top1"] == 0
