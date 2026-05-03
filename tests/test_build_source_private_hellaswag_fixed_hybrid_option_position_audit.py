from __future__ import annotations

import json

from scripts import build_source_private_hellaswag_fixed_hybrid_option_position_audit as audit


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


def test_fixed_hybrid_option_position_audit_builds_cached_artifacts(tmp_path):
    rows = [
        _row(0, 0, 1, 0, 0, 0),
        _row(1, 1, 2, 1, 1, 1),
        _row(2, 2, 3, 2, 2, 2),
        _row(3, 3, 0, 3, 3, 3),
        _row(4, 0, 0, 0, 0, 0),
        _row(5, 1, 1, 1, 1, 1),
        _row(6, 2, 2, 2, 2, 2),
        _row(7, 3, 3, 3, 3, 3),
    ]
    predictions_path = tmp_path / "slice" / "predictions.jsonl"
    _write_jsonl(predictions_path, rows)
    audit_path = tmp_path / "candidate_audit.json"
    _write_json(
        audit_path,
        {
            "slice_rows": [
                {
                    "eval_slice_start": 0,
                    "eval_slice_end_exclusive": len(rows),
                    "predictions_path": str(predictions_path),
                }
            ]
        },
    )

    payload = audit.build_audit(
        audit_path=audit_path,
        tail_predictions=None,
        output_dir=tmp_path / "out",
        bootstrap_samples=20,
        run_date="2026-05-03",
    )

    assert payload["gate"] == "source_private_hellaswag_fixed_hybrid_option_position_audit"
    assert payload["headline"]["eval_rows"] == len(rows)
    assert payload["headline"]["positive_answer_position_count"] == 4
    assert len(payload["option_position_rows"]) == 4
    assert len(payload["prediction_distribution_rows"]) == 3
    assert len(payload["roll_control_rows"]) == 6
    assert len(payload["global_packet_permutation_rows"]) == 23
    assert len(payload["rowwise_random_control_rows"]) == 20
    assert len(payload["equivariance_rows"]) == 24
    assert (tmp_path / "out" / "hellaswag_fixed_hybrid_option_position_audit.json").exists()
    assert (tmp_path / "out" / "option_position_rows.csv").exists()
    assert (tmp_path / "out" / "roll_control_rows.csv").exists()
    assert (tmp_path / "out" / "global_packet_permutation_rows.csv").exists()
