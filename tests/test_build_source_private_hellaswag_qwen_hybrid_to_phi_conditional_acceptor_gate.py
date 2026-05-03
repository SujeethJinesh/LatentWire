from __future__ import annotations

import json

from scripts import build_source_private_hellaswag_qwen_hybrid_to_phi_conditional_acceptor_gate as gate


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _row(row_id, answer, selected, vote, hidden, score_mean, target, target_scores, margin=0.1):
    controls = {
        "source_label_prediction": score_mean,
        "source_rank_only_bagged_prediction": score_mean,
        "score_only_bagged_prediction": score_mean,
        "score_vote_prediction": score_mean,
        "trained_label_prediction": selected,
        "wrong_example_hidden_prediction": (answer + 1) % 4,
        "zero_hidden_prediction": score_mean,
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
        "selected_margin": float(margin),
        "_target_prediction": int(target),
        "_target_scores": [float(value) for value in target_scores],
        **{key: int(value) for key, value in controls.items()},
    }


def test_qwen_hybrid_to_phi_conditional_acceptor_gate_builds_fit_select_eval(tmp_path):
    slice_specs = []
    for slice_id, start in enumerate((100, 200)):
        rows = [
            _row(start + 0, 0, 0, 0, 0, 0, 0, [4.0, 1.0, 0.0, -1.0], 0.05),
            _row(start + 1, 1, 0, 1, 0, 0, 1, [0.0, 3.0, 1.0, -1.0], 0.15),
            _row(start + 2, 2, 2, 2, 2, 2, 2, [0.0, 1.0, 4.0, -1.0], 0.25),
            _row(start + 3, 3, 2, 3, 2, 2, 3, [0.0, 1.0, -1.0, 4.0], 0.35),
            _row(start + 4, 0, 1, 0, 1, 1, 0, [4.0, 2.0, 0.0, -1.0], 0.45),
            _row(start + 5, 1, 1, 1, 1, 1, 2, [0.0, 2.0, 3.0, -1.0], 0.55),
        ]
        qwen_path = tmp_path / f"slice{slice_id}" / "qwen.jsonl"
        target_path = tmp_path / f"slice{slice_id}" / "target.json"
        _write_jsonl(
            qwen_path,
            [{key: value for key, value in row.items() if not key.startswith("_target")} for row in rows],
        )
        _write_json(
            target_path,
            {
                "row_count": len(rows),
                "row_ids": [row["row_id"] for row in rows],
                "source_predictions": [row["_target_prediction"] for row in rows],
                "source_scores": [row["_target_scores"] for row in rows],
            },
        )
        slice_specs.append(
            {
                "slice_start": start,
                "slice_end_exclusive": start + len(rows),
                "qwen_predictions": str(qwen_path),
                "phi_target_score_cache": str(target_path),
            }
        )

    payload = gate.build_gate(
        output_dir=tmp_path / "out",
        slices=tuple(slice_specs),
        fit_rows_per_slice=2,
        select_rows_per_slice=2,
        bootstrap_samples=20,
        run_date="2026-05-03",
    )

    assert payload["gate"] == "source_private_hellaswag_qwen_hybrid_to_phi_conditional_acceptor_gate"
    assert payload["headline"]["fit_rows"] == 4
    assert payload["headline"]["select_rows"] == 4
    assert payload["headline"]["eval_rows"] == 4
    assert "conditional_acceptor_accuracy" in payload["headline"]
    assert any(row["method"] == "conditional_target_acceptor" for row in payload["method_rows"])
    assert any(row["method"].startswith("control_") for row in payload["method_rows"])
    assert (tmp_path / "out" / "hellaswag_qwen_hybrid_to_phi_conditional_acceptor_gate.json").exists()
    assert (tmp_path / "out" / "method_rows.csv").exists()
    assert (tmp_path / "out" / "slice_rows.csv").exists()
