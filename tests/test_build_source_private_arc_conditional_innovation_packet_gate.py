from __future__ import annotations

import json

import pytest

from scripts import build_source_private_arc_conditional_innovation_packet_gate as gate
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate


def _write_arc_rows(path, *, row_count: int = 16) -> list[arc_gate.ArcRow]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_rows = []
    for index in range(row_count):
        answer_index = 1 + (index % 3)
        raw_rows.append(
            {
                "id": f"synthetic_{index}",
                "content_id": f"synthetic_content_{index}",
                "question": f"Which option is supported for item {index}?",
                "choices": ["receiver distractor", "answer one", "answer two", "answer three"],
                "choice_labels": ["A", "B", "C", "D"],
                "answer_index": answer_index,
                "answer_label": ["A", "B", "C", "D"][answer_index],
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in raw_rows) + "\n", encoding="utf-8")
    return arc_gate._load_rows(path)


def _write_score_cache(path, *, rows: list[arc_gate.ArcRow], scores: list[list[float]]) -> None:
    predictions = [max(range(len(row_scores)), key=lambda index: row_scores[index]) for row_scores in scores]
    payload = {
        "row_count": len(rows),
        "row_ids": [row.row_id for row in rows],
        "content_digest": gate._content_digest(rows),
        "source_scores": scores,
        "source_predictions": predictions,
        "source_model": {"model": "synthetic"},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_conditional_innovation_gate_recovers_synthetic_residual_signal(tmp_path):
    eval_path = tmp_path / "arc.jsonl"
    rows = _write_arc_rows(eval_path)
    source_scores = []
    receiver_scores = []
    for row in rows:
        receiver_row = [8.0, -2.0, -2.0, -2.0]
        source_row = [7.0, -4.0, -4.0, -4.0]
        source_row[row.answer_index] = 6.0
        source_scores.append(source_row)
        receiver_scores.append(receiver_row)
    source_cache = tmp_path / "source_scores.json"
    receiver_cache = tmp_path / "receiver_scores.json"
    _write_score_cache(source_cache, rows=rows, scores=source_scores)
    _write_score_cache(receiver_cache, rows=rows, scores=receiver_scores)

    payload = gate.build_gate(
        output_dir=tmp_path / "out",
        eval_path=eval_path,
        source_score_cache=source_cache,
        receiver_score_cache=receiver_cache,
        clips=(1.5, 2.0),
        ridges=(0.01, 0.1, 1.0),
        bootstrap_samples=200,
        bootstrap_seed=11,
        shuffle_seed=13,
        run_date="2026-05-03",
    )

    headline = payload["headline"]
    assert headline["matched_conditional_innovation_packet_heldout_accuracy"] == pytest.approx(1.0)
    assert headline["source_label_text_heldout_accuracy"] == pytest.approx(0.0)
    assert headline["source_index_only_decoder_heldout_accuracy"] <= 0.25
    assert payload["condition_metrics"]["candidate_roll_innovation_control"]["heldout_accuracy"] < 1.0
    assert payload["packet_contract"]["raw_payload_bytes"] == 3
    assert (tmp_path / "out" / "arc_conditional_innovation_packet_gate.json").exists()
    assert (tmp_path / "out" / "arc_conditional_innovation_packet_predictions.jsonl").exists()


def test_conditional_innovation_gate_rejects_cache_row_mismatch(tmp_path):
    eval_path = tmp_path / "arc.jsonl"
    rows = _write_arc_rows(eval_path, row_count=4)
    cache = tmp_path / "scores.json"
    _write_score_cache(cache, rows=rows[:-1], scores=[[1.0, 0.0, 0.0, 0.0] for _ in rows[:-1]])

    with pytest.raises(ValueError, match="row count"):
        gate._load_score_cache(cache, rows=rows, role="source")
