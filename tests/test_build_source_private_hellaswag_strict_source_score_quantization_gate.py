from __future__ import annotations

import json

from scripts import build_source_private_hellaswag_strict_source_score_quantization_gate as gate


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _score_cache(path, row_ids, scores):
    _write_json(
        path,
        {
            "row_count": len(row_ids),
            "row_ids": [str(row_id) for row_id in row_ids],
            "source_scores": scores,
            "source_predictions": [
                max(range(4), key=lambda choice, row=row: row[choice]) for row in scores
            ],
        },
    )


def _slice(tmp_path, *, name, start, row_ids, answers, selected, scores, nested=False):
    root = tmp_path / name
    pred_dir = root / "bagged_gate" if nested else root
    predictions = []
    for row_id, answer, prediction in zip(row_ids, answers, selected, strict=True):
        predictions.append(
            {
                "row_id": str(row_id),
                "answer_index": answer,
                "selected_prediction": prediction,
            }
        )
    _write_jsonl(pred_dir / "predictions.jsonl", predictions)
    score_path = root / "score_cache.json"
    _score_cache(score_path, row_ids, scores)
    if nested:
        _write_json(
            root / "slice_gate.json",
            {
                "eval_cache_metadata": {"score_cache": str(score_path)},
                "bagged_gate_path": str(root / "bagged_gate" / "slice_gate.json"),
            },
        )
        _write_json(
            root / "bagged_gate" / "slice_gate.json",
            {"eval_score_cache": str(score_path)},
        )
    else:
        _write_json(root / "slice_gate.json", {"eval_score_cache": str(score_path)})
    return {
        "artifact_path": str(root / "slice_gate.json"),
        "eval_slice_start": start,
        "eval_slice_end_exclusive": start + len(row_ids),
        "eval_rows": len(row_ids),
    }


def test_source_score_quantization_gate_builds_train_only_score_controls(tmp_path):
    train_rows = [
        {"id": "t0", "answer_index": 0},
        {"id": "t1", "answer_index": 1},
        {"id": "t2", "answer_index": 2},
        {"id": "t3", "answer_index": 3},
        {"id": "t4", "answer_index": 0},
        {"id": "t5", "answer_index": 1},
    ]
    train_rows_path = tmp_path / "hellaswag_train.jsonl"
    _write_jsonl(train_rows_path, train_rows)
    train_cache_path = tmp_path / "train_score_cache.json"
    _score_cache(
        train_cache_path,
        [row["id"] for row in train_rows],
        [
            [4, 1, 0, 0],
            [0, 4, 1, 0],
            [0, 1, 4, 0],
            [0, 0, 1, 4],
            [4, 2, 1, 0],
            [1, 4, 2, 0],
        ],
    )
    first_pred_dir = tmp_path / "left"
    _write_jsonl(
        first_pred_dir / "sample_caches.jsonl",
        [{"train_score_cache": str(train_cache_path)}],
    )
    left = _slice(
        tmp_path,
        name="left",
        start=0,
        row_ids=["e0", "e1", "e2", "e3"],
        answers=[0, 1, 2, 3],
        selected=[0, 1, 1, 3],
        scores=[[4, 1, 0, 0], [0, 4, 1, 0], [0, 4, 3, 0], [0, 0, 1, 4]],
    )
    right = _slice(
        tmp_path,
        name="right",
        start=4,
        row_ids=["e4", "e5", "e6", "e7"],
        answers=[0, 1, 2, 3],
        selected=[0, 0, 2, 3],
        scores=[[4, 1, 0, 0], [4, 3, 1, 0], [0, 1, 4, 0], [0, 0, 1, 4]],
        nested=True,
    )
    source = {
        "pass_gate": True,
        "headline": {
            "total_eval_rows": 8,
            "weighted_selected_eval_accuracy": 0.75,
        },
        "slice_rows": [left, right],
    }
    source_path = tmp_path / "source.json"
    _write_json(source_path, source)

    payload = gate.build_gate(
        source_path=source_path,
        output_dir=tmp_path / "out",
        train_rows_path=train_rows_path,
        bootstrap_samples=20,
        run_date="2026-05-03",
    )

    assert payload["gate"] == "source_private_hellaswag_strict_source_score_quantization_gate"
    assert payload["headline"]["total_eval_rows"] == 8
    assert payload["headline"]["score_quantized_variant_count"] >= 4
    assert payload["variant_rows"][0]["matched_accuracy"] >= 0.75
    assert (tmp_path / "out" / "hellaswag_strict_source_score_quantization_gate.json").exists()
    assert (tmp_path / "out" / "variant_rows.csv").exists()
