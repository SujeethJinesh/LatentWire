from __future__ import annotations

import json

from scripts import build_source_private_hellaswag_strict_channel_selector_gate as gate


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _prediction_row(row_id, answer, selected, vote, margin=0.5):
    return {
        "row_id": str(row_id),
        "answer_index": int(answer),
        "selected_prediction": int(selected),
        "vote_prediction": int(vote),
        "hidden_mean_prediction": int(vote),
        "trained_label_prediction": int(selected),
        "score_mean_prediction": int(selected),
        "score_vote_prediction": int(selected),
        "wrong_example_hidden_prediction": (int(answer) + 1) % 4,
        "candidate_roll_hidden_prediction": (int(selected) + 1) % 4,
        "selected_margin": float(margin),
    }


def test_channel_selector_gate_builds_fixed_and_prefix_rows(tmp_path):
    left_rows = [
        _prediction_row("0", 0, 1, 0, 0.1),
        _prediction_row("1", 1, 1, 1, 0.2),
        _prediction_row("2", 2, 0, 2, 0.9),
        _prediction_row("3", 3, 3, 0, 1.0),
    ]
    right_rows = [
        _prediction_row("4", 0, 1, 0, 0.1),
        _prediction_row("5", 1, 0, 1, 0.2),
        _prediction_row("6", 2, 2, 2, 0.9),
        _prediction_row("7", 3, 3, 3, 1.0),
    ]
    left_path = tmp_path / "left" / "predictions.jsonl"
    right_path = tmp_path / "right" / "predictions.jsonl"
    _write_jsonl(left_path, left_rows)
    _write_jsonl(right_path, right_rows)
    audit_path = tmp_path / "candidate_audit.json"
    _write_json(
        audit_path,
        {
            "pass_gate": True,
            "slice_rows": [
                {
                    "eval_slice_start": 0,
                    "predictions_path": str(left_path),
                },
                {
                    "eval_slice_start": 4,
                    "predictions_path": str(right_path),
                },
            ],
        },
    )

    payload = gate.build_gate(
        input_path=audit_path,
        output_dir=tmp_path / "out",
        bootstrap_samples=20,
        run_date="2026-05-03",
    )

    assert payload["gate"] == "source_private_hellaswag_strict_channel_selector_gate"
    assert payload["headline"]["total_eval_rows"] == 8
    assert payload["headline"]["method_count"] >= 5
    assert any(
        row["method"] == "fixed_hybrid_vote_on_score_agreement"
        for row in payload["method_rows"]
    )
    assert payload["headline"]["best_non_oracle_method"]["method_accuracy"] >= 0.75
    assert (tmp_path / "out" / "hellaswag_strict_channel_selector_gate.json").exists()
    assert (tmp_path / "out" / "method_rows.csv").exists()
