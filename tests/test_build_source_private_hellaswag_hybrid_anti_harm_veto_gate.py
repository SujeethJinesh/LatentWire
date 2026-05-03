from __future__ import annotations

import json

from scripts import build_source_private_hellaswag_hybrid_anti_harm_veto_gate as gate


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _prediction_row(row_id, answer, selected, vote, hidden, score_mean, margin=0.1):
    return {
        "row_id": str(row_id),
        "answer_index": int(answer),
        "selected_prediction": int(selected),
        "vote_prediction": int(vote),
        "hidden_mean_prediction": int(hidden),
        "score_mean_prediction": int(score_mean),
        "score_vote_prediction": int(score_mean),
        "trained_label_prediction": int(selected),
        "selected_margin": float(margin),
    }


def test_hybrid_anti_harm_veto_gate_builds_fit_select_artifacts(tmp_path):
    first_rows = []
    for idx in range(1024):
        answer = idx % 4
        selected = answer if idx % 3 else (answer + 1) % 4
        hidden = answer if idx % 5 else selected
        first_rows.append(
            _prediction_row(
                idx,
                answer,
                selected,
                vote=answer,
                hidden=hidden,
                score_mean=hidden,
                margin=(idx % 17) / 17.0,
            )
        )
    second_rows = []
    for idx in range(1024, 2048):
        answer = idx % 4
        selected = answer if idx % 4 else (answer + 1) % 4
        hidden = answer if idx % 7 else selected
        second_rows.append(
            _prediction_row(
                idx,
                answer,
                selected,
                vote=answer,
                hidden=hidden,
                score_mean=hidden,
                margin=(idx % 19) / 19.0,
            )
        )

    first_path = tmp_path / "slice0" / "predictions.jsonl"
    second_path = tmp_path / "slice1" / "predictions.jsonl"
    _write_jsonl(first_path, first_rows)
    _write_jsonl(second_path, second_rows)
    audit_path = tmp_path / "candidate_audit.json"
    _write_json(
        audit_path,
        {
            "pass_gate": True,
            "slice_rows": [
                {
                    "eval_slice_start": 0,
                    "eval_slice_end_exclusive": 1024,
                    "predictions_path": str(first_path),
                },
                {
                    "eval_slice_start": 1024,
                    "eval_slice_end_exclusive": 2048,
                    "predictions_path": str(second_path),
                },
            ],
        },
    )

    payload = gate.build_gate(
        input_path=audit_path,
        output_dir=tmp_path / "out",
        bootstrap_samples=20,
        run_date="2026-05-03",
        cross_family_slices=(),
    )

    assert payload["gate"] == "source_private_hellaswag_hybrid_anti_harm_veto_gate"
    assert payload["headline"]["fit_rows"] == 512
    assert payload["headline"]["selection_rows"] == 512
    assert payload["headline"]["heldout_eval_rows"] == 1024
    assert payload["cross_family"] is None
    assert any(row["method"] == "fit_select_single_rule_anti_harm_veto" for row in payload["method_rows"])
    assert (tmp_path / "out" / "hellaswag_hybrid_anti_harm_veto_gate.json").exists()
    assert (tmp_path / "out" / "method_rows.csv").exists()
