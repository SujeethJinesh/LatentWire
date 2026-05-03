from __future__ import annotations

import json

from scripts import build_source_private_hellaswag_qwen_hybrid_to_phi_cross_family_gate as gate


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _row(row_id, answer, selected, vote, score_mean, target):
    return {
        "row_id": str(row_id),
        "answer_index": int(answer),
        "selected_prediction": int(selected),
        "hidden_mean_prediction": int(selected),
        "vote_prediction": int(vote),
        "score_mean_prediction": int(score_mean),
        "score_vote_prediction": int(score_mean),
        "source_label_prediction": int(score_mean),
        "source_rank_only_bagged_prediction": int(score_mean),
        "score_only_bagged_prediction": int(score_mean),
        "trained_label_prediction": int(score_mean),
        "wrong_example_hidden_prediction": (int(answer) + 1) % 4,
        "zero_hidden_prediction": int(score_mean),
        "candidate_roll_hidden_prediction": (int(selected) + 1) % 4,
        "score_channel_roll_hidden_prediction": (int(score_mean) + 1) % 4,
        "_target_prediction": int(target),
    }


def test_qwen_hybrid_to_phi_gate_builds_from_cached_slices(tmp_path, monkeypatch):
    slice_specs = []
    for slice_id, start in enumerate((100, 104)):
        rows = [
            _row(start + 0, 0, 1, 0, 1, 1),
            _row(start + 1, 1, 1, 1, 1, 0),
            _row(start + 2, 2, 0, 2, 0, 0),
            _row(start + 3, 3, 3, 0, 3, 0),
        ]
        qwen_path = tmp_path / f"slice{slice_id}" / "qwen.jsonl"
        target_path = tmp_path / f"slice{slice_id}" / "target.json"
        _write_jsonl(qwen_path, [{k: v for k, v in row.items() if k != "_target_prediction"} for row in rows])
        _write_json(
            target_path,
            {
                "row_count": len(rows),
                "row_ids": [row["row_id"] for row in rows],
                "source_predictions": [row["_target_prediction"] for row in rows],
                "source_scores": [[0.0, 0.0, 0.0, 0.0] for _ in rows],
            },
        )
        slice_specs.append(
            {
                "slice_start": slice_id * 4,
                "slice_end_exclusive": slice_id * 4 + 4,
                "qwen_predictions": str(qwen_path),
                "phi_target_score_cache": str(target_path),
            }
        )

    monkeypatch.setattr(gate, "DEFAULT_SLICES", tuple(slice_specs))
    payload = gate.build_gate(
        output_dir=tmp_path / "out",
        bootstrap_samples=20,
        train_prefix_rows_per_slice=1,
        run_date="2026-05-03",
    )

    assert payload["gate"] == "source_private_hellaswag_qwen_hybrid_to_phi_cross_family_gate"
    assert payload["headline"]["heldout_eval_rows"] == 6
    assert payload["headline"]["hybrid_accuracy"] >= payload["headline"]["candidate_only_accuracy"]
    assert payload["packet_contract"]["target_family"] == "Phi-3-mini"
    assert (tmp_path / "out" / "hellaswag_qwen_hybrid_to_phi_cross_family_gate.json").exists()
    assert (tmp_path / "out" / "method_rows.csv").exists()
