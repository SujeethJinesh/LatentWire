from __future__ import annotations

import datetime as dt
import json

from scripts import build_source_private_hellaswag_score_channel_true_candidate_text_permutation_gate as gate
from scripts import build_source_private_hellaswag_score_packet_headroom as score_headroom


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _canonical_row(index: int, answer: int) -> dict:
    choices = [f"choice {index}-{slot}" for slot in range(4)]
    return {
        "answer_index": answer,
        "answer_label": "ABCD"[answer],
        "choice_labels": ["A", "B", "C", "D"],
        "choices": choices,
        "content_id": f"content-{index}",
        "id": str(index),
        "question": f"context {index}",
        "row_index": index,
        "source_name": "synthetic/hellaswag",
    }


def test_score_channel_true_candidate_text_permutation_uses_canonical_remap(tmp_path):
    eval_rows = [_canonical_row(index, index % 4) for index in range(8)]
    eval_path = tmp_path / "hellaswag_validation.jsonl"
    _write_jsonl(eval_path, eval_rows)

    canonical_rows = gate._slice_rows(eval_path, start=0, rows=len(eval_rows))
    expanded_rows, _ = gate._expanded_rows(canonical_rows, gate._permutations("fixed8"))
    source_scores = []
    source_predictions = []
    for row in expanded_rows:
        scores = [-1.0, -1.0, -1.0, -1.0]
        scores[row.answer_index] = 2.0
        source_scores.append(scores)
        source_predictions.append(row.answer_index)

    score_cache = tmp_path / "score_cache.json"
    _write_json(
        score_cache,
        {
            "created_utc": dt.datetime.now(dt.UTC).isoformat(),
            "row_count": len(expanded_rows),
            "row_ids": [row.row_id for row in expanded_rows],
            "content_digest": score_headroom._content_digest(expanded_rows),
            "source_scores": source_scores,
            "source_predictions": source_predictions,
            "source_model": {"kind": "synthetic_choice_scorer", "latency_s": 0.0},
        },
    )

    payload = gate.build_gate(
        output_dir=tmp_path / "out",
        eval_path=eval_path,
        slice_start=0,
        eval_rows=len(eval_rows),
        permutation_mode="fixed8",
        score_cache=score_cache,
        bootstrap_samples=30,
        min_eval_rows_for_smoke_pass=4,
        run_date="2026-05-03",
    )

    headline = payload["headline"]
    assert payload["gate"] == "source_private_hellaswag_score_channel_true_candidate_text_permutation_gate"
    assert headline["permuted_evaluations"] == 64
    assert headline["identity_accuracy"] == 1.0
    assert headline["remapped_accuracy"] == 1.0
    assert headline["canonical_packet_consistency_rate"] == 1.0
    assert headline["unremapped_accuracy"] < 1.0
    assert headline["wrong_remap_accuracy"] == 0.0
    assert payload["smoke_pass_gate"] is True
    assert payload["promotion_pass_gate"] is False
    assert (tmp_path / "out" / "permutation_prediction_rows.jsonl").exists()
    assert (tmp_path / "out" / "permutation_summary_rows.csv").exists()
    assert (tmp_path / "out" / "answer_position_rows.csv").exists()
