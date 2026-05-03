from __future__ import annotations

import json

from scripts import build_source_private_hellaswag_fixed_hybrid_full_validation_gate as gate


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _row(row_id, answer, selected, vote, hidden, score_mean):
    controls = {
        "source_label_prediction": selected,
        "source_rank_only_bagged_prediction": selected,
        "score_only_bagged_prediction": selected,
        "score_vote_prediction": score_mean,
        "trained_label_prediction": selected,
        "wrong_example_hidden_prediction": (answer + 1) % 4,
        "zero_hidden_prediction": selected,
        "candidate_roll_hidden_prediction": (selected + 1) % 4,
        "score_channel_roll_hidden_prediction": (score_mean + 1) % 4,
    }
    return {
        "row_id": str(row_id),
        "answer_index": int(answer),
        "selected_prediction": int(selected),
        "vote_prediction": int(vote),
        "hidden_mean_prediction": int(hidden),
        "score_mean_prediction": int(score_mean),
        "selected_margin": 0.1,
        **{key: int(value) for key, value in controls.items()},
    }


def test_fixed_hybrid_full_validation_gate_builds_tail_artifacts(tmp_path):
    slice_rows = [
        _row(0, 0, 1, 0, 0, 0),
        _row(1, 1, 1, 1, 1, 1),
        _row(2, 2, 0, 2, 2, 2),
        _row(3, 3, 3, 0, 0, 1),
    ]
    tail_rows = [
        _row(4, 0, 1, 0, 0, 0),
        _row(5, 1, 1, 1, 1, 1),
    ]
    slice_path = tmp_path / "slice" / "predictions.jsonl"
    tail_path = tmp_path / "tail" / "predictions.jsonl"
    _write_jsonl(slice_path, slice_rows)
    _write_jsonl(tail_path, tail_rows)
    audit_path = tmp_path / "audit.json"
    _write_json(
        audit_path,
        {
            "slice_rows": [
                {
                    "eval_slice_start": 0,
                    "eval_slice_end_exclusive": 4,
                    "predictions_path": str(slice_path),
                }
            ]
        },
    )

    payload = gate.build_gate(
        audit_path=audit_path,
        tail_predictions=tail_path,
        output_dir=tmp_path / "out",
        tail_start=4,
        tail_end_exclusive=6,
        bootstrap_samples=20,
        run_date="2026-05-03",
    )

    assert payload["gate"] == "source_private_hellaswag_fixed_hybrid_full_validation_gate"
    assert payload["headline"]["eval_rows"] == 6
    assert payload["headline"]["slice_count"] == 2
    assert payload["headline"]["fixed_hybrid_accuracy"] >= payload["headline"]["candidate_only_accuracy"]
    assert (tmp_path / "out" / "hellaswag_fixed_hybrid_full_validation_gate.json").exists()
    assert (tmp_path / "out" / "method_rows.csv").exists()
