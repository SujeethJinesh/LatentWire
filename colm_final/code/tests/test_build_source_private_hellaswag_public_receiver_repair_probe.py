from __future__ import annotations

import json

from scripts import build_source_private_hellaswag_public_receiver_repair_probe as probe
from scripts import build_source_private_hellaswag_score_packet_headroom as headroom
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate


def _row(row_id: str, answer_index: int, *, marker: str = "gold") -> dict[str, object]:
    labels = ["A", "B", "C", "D"]
    choices = [
        f"{row_id} dull ending",
        f"{row_id} {marker} ending",
        f"{row_id} spare ending",
        f"{row_id} distractor ending",
    ]
    return {
        "id": row_id,
        "content_id": row_id,
        "question": f"Context {row_id}",
        "choices": choices,
        "choice_labels": labels,
        "answer_index": answer_index,
        "answer_label": labels[answer_index],
        "source_name": "unit",
    }


def _write_rows(path, rows) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_top2_public_rerank_can_add_signal_when_public_top_is_outside_source_top2() -> None:
    predictions = probe._public_repair_predictions(
        source_scores=[[0.8, 0.7, 0.1, 0.0], [0.5, 0.4, 0.1, 0.0]],
        public_scores=[[0.2, 0.6, 0.1, 1.5], [0.7, 0.2, 0.1, 1.4]],
        public_predictions=[3, 3],
    )

    assert predictions["source_label_copy"] == [0, 0]
    assert predictions["public_target_only"] == [3, 3]
    assert predictions["top2_public_rerank"] == [1, 0]
    assert predictions["public_if_in_source_top2"] == [0, 0]


def test_public_receiver_repair_probe_writes_artifacts(tmp_path) -> None:
    train_rows = [_row(f"train_{index}", 1) for index in range(16)]
    eval_rows = [_row("eval_0", 1), _row("eval_1", 0)]
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_rows(train_path, train_rows)
    _write_rows(eval_path, eval_rows)

    loaded_eval_rows = arc_gate._load_rows(eval_path)
    score_cache = tmp_path / "source_score_cache.json"
    headroom._write_score_cache(
        score_cache,
        rows=loaded_eval_rows,
        source_scores=[[0.8, 0.7, 0.0, -0.1], [0.6, 0.5, 0.0, -0.1]],
        source_predictions=[0, 0],
        source_model={"kind": "unit"},
    )

    payload = probe.build_probe(
        output_dir=tmp_path / "out",
        train_path=train_path,
        eval_path=eval_path,
        score_cache=score_cache,
        dim=256,
        epochs=2,
        split_seed=3,
        dev_rows=4,
        run_date="2026-05-01",
    )

    assert payload["gate"] == "source_private_hellaswag_public_receiver_repair_probe"
    assert payload["train_rows"] == 16
    assert payload["eval_rows"] == 2
    assert "top2_public_rerank" in payload["metrics"]
    assert (tmp_path / "out" / "hellaswag_public_receiver_repair_probe.json").exists()
    assert (tmp_path / "out" / "predictions.jsonl").exists()
