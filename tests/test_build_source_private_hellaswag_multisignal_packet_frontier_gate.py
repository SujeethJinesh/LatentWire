from __future__ import annotations

import json

from scripts.build_source_private_hellaswag_multisignal_packet_frontier_gate import build_gate


def _write_jsonl(path, rows) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_multisignal_packet_frontier_gate_runs_on_synthetic_slice(tmp_path) -> None:
    qwen_path = tmp_path / "qwen.jsonl"
    phi_path = tmp_path / "phi.json"
    source_path = tmp_path / "source.json"
    answers = [0, 1, 2, 3, 0, 1, 2, 3]
    selected = [0, 2, 2, 0, 1, 1, 3, 3]
    hidden = [0, 1, 2, 0, 0, 1, 3, 3]
    score = [1, 2, 2, 3, 1, 1, 2, 0]
    phi_predictions = [1, 1, 0, 3, 0, 2, 2, 1]
    source_scores = [
        [4.0, 1.0, 0.0, -1.0],
        [0.0, 4.0, 1.0, -1.0],
        [0.0, 1.0, 4.0, -1.0],
        [0.0, 1.0, -1.0, 4.0],
        [4.0, 0.0, 1.0, -1.0],
        [0.0, 4.0, 1.0, -1.0],
        [1.0, 0.0, 4.0, -1.0],
        [0.0, 1.0, -1.0, 4.0],
    ]
    phi_scores = [
        [0.0, 3.0, 1.0, -1.0],
        [0.0, 3.0, 1.0, -1.0],
        [3.0, 0.0, 1.0, -1.0],
        [0.0, -1.0, 1.0, 3.0],
        [3.0, 0.0, 1.0, -1.0],
        [0.0, 1.0, 3.0, -1.0],
        [0.0, 1.0, 3.0, -1.0],
        [0.0, 3.0, 1.0, -1.0],
    ]
    rows = []
    for index, answer in enumerate(answers):
        rows.append(
            {
                "answer_index": answer,
                "candidate_roll_hidden_prediction": (hidden[index] + 1) % 4,
                "hidden_mean_prediction": hidden[index],
                "row_id": f"row-{index}",
                "score_channel_roll_hidden_prediction": (score[index] + 1) % 4,
                "score_mean_prediction": score[index],
                "score_only_bagged_prediction": score[index],
                "score_vote_prediction": score[index],
                "selected_margin": 0.5 + 0.1 * index,
                "selected_prediction": selected[index],
                "source_label_prediction": score[index],
                "source_rank_only_bagged_prediction": hidden[index],
                "trained_label_prediction": score[index],
                "vote_prediction": hidden[index] if hidden[index] == score[index] else selected[index],
                "wrong_example_hidden_prediction": (hidden[index] + 2) % 4,
                "zero_hidden_prediction": 0,
            }
        )
    _write_jsonl(qwen_path, rows)
    phi_path.write_text(
        json.dumps(
            {
                "content_digest": "synthetic",
                "created_utc": "2026-05-04T00:00:00+00:00",
                "row_count": len(rows),
                "row_ids": [row["row_id"] for row in rows],
                "source_model": "synthetic-phi",
                "source_predictions": phi_predictions,
                "source_scores": phi_scores,
            }
        ),
        encoding="utf-8",
    )
    source_path.write_text(
        json.dumps(
            {
                "content_digest": "synthetic",
                "created_utc": "2026-05-04T00:00:00+00:00",
                "row_count": len(rows),
                "row_ids": [row["row_id"] for row in rows],
                "source_model": "synthetic-qwen",
                "source_predictions": [max(range(4), key=lambda item: score[item]) for score in source_scores],
                "source_scores": source_scores,
            }
        ),
        encoding="utf-8",
    )
    payload = build_gate(
        output_dir=tmp_path / "out",
        slices=(
            {
                "slice_start": 0,
                "slice_end_exclusive": len(rows),
                "qwen_predictions": str(qwen_path),
                "phi_target_score_cache": str(phi_path),
            },
        ),
        source_score_cache=source_path,
        fit_rows_per_slice=2,
        select_rows_per_slice=2,
        bootstrap_samples=50,
    )
    assert payload["gate"] == "source_private_hellaswag_multisignal_packet_frontier_gate"
    assert payload["headline"]["eval_rows"] == 4
    assert payload["headline"]["framed_record_bytes"] == 5
    assert (tmp_path / "out" / "prediction_rows.jsonl").exists()
