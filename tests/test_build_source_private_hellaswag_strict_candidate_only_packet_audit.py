from __future__ import annotations

import json

from scripts import build_source_private_hellaswag_strict_candidate_only_packet_audit as audit


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _slice(
    tmp_path,
    *,
    name: str,
    start: int,
    rows: list[tuple[int, int, int, int]],
    nested: bool = False,
):
    slice_dir = tmp_path / name
    predictions_dir = slice_dir / "bagged_gate" if nested else slice_dir
    predictions = []
    for offset, (answer, selected, source_rank, score_only) in enumerate(rows):
        predictions.append(
            {
                "row_id": str(start + offset),
                "answer_index": answer,
                "selected_prediction": selected,
                "source_label_prediction": source_rank,
                "source_rank_only_bagged_prediction": source_rank,
                "score_only_bagged_prediction": score_only,
                "zero_hidden_prediction": source_rank,
                "candidate_roll_hidden_prediction": (selected + 1) % 4,
                "score_channel_roll_hidden_prediction": (score_only + 1) % 4,
                "wrong_example_hidden_prediction": (answer + 1) % 4,
            }
        )
    _write_jsonl(predictions_dir / "predictions.jsonl", predictions)
    selected_acc = sum(answer == selected for answer, selected, _, _ in rows) / len(rows)
    source_rank_acc = sum(answer == source_rank for answer, _, source_rank, _ in rows) / len(rows)
    score_acc = sum(answer == score_only for answer, _, _, score_only in rows) / len(rows)
    if nested:
        _write_json(
            slice_dir / "slice_gate.json",
            {
                "bagged_gate_path": str(slice_dir / "bagged_gate" / "slice_gate.json"),
                "headline": {"selected_eval_accuracy": selected_acc},
            },
        )
        _write_json(
            slice_dir / "bagged_gate" / "slice_gate.json",
            {"headline": {"selected_eval_accuracy": selected_acc}},
        )
    else:
        _write_json(slice_dir / "slice_gate.json", {"headline": {"selected_eval_accuracy": selected_acc}})
    return {
        "artifact_path": str(slice_dir / "slice_gate.json"),
        "eval_slice_start": start,
        "eval_slice_end_exclusive": start + len(rows),
        "eval_rows": len(rows),
        "selected_eval_accuracy": selected_acc,
        "best_label_copy_eval_accuracy": source_rank_acc,
        "source_label_copy_eval_accuracy": source_rank_acc,
        "source_rank_only_bagged_control_accuracy": source_rank_acc,
        "score_only_bagged_control_accuracy": score_acc,
        "zero_hidden_control_accuracy": source_rank_acc,
        "paired_ci95_low_vs_best_label_copy": 0.1,
        "paired_ci95_low_vs_source_rank_only_bagged": 0.1,
        "paired_ci95_low_vs_score_only_bagged": 0.1,
        "rank_score_channel_controls_available": True,
        "source_text_exposed": False,
        "source_kv_exposed": False,
        "raw_hidden_vector_transmitted": False,
        "raw_scores_transmitted": False,
    }


def test_strict_candidate_only_packet_audit_recomputes_and_compacts(tmp_path):
    left = _slice(
        tmp_path,
        name="left",
        start=0,
        rows=[
            (0, 0, 1, 1),
            (1, 1, 1, 2),
            (2, 2, 3, 3),
            (3, 0, 0, 0),
        ],
    )
    right = _slice(
        tmp_path,
        name="right",
        start=4,
        nested=True,
        rows=[
            (0, 0, 1, 1),
            (1, 1, 2, 2),
            (2, 2, 3, 3),
            (3, 3, 0, 0),
        ],
    )
    total_rows = left["eval_rows"] + right["eval_rows"]
    weighted_selected = (
        left["selected_eval_accuracy"] * left["eval_rows"]
        + right["selected_eval_accuracy"] * right["eval_rows"]
    ) / total_rows
    source = {
        "gate": "source_private_hellaswag_hidden_innovation_multi_slice_stress",
        "pass_gate": True,
        "headline": {
            "raw_payload_bytes": 2,
            "framed_record_bytes": 5,
            "strict_delta_required": 0.02,
            "total_eval_rows": total_rows,
            "weighted_selected_eval_accuracy": weighted_selected,
            "weighted_best_label_copy_eval_accuracy": 0.25,
            "min_ci95_low_vs_best_label_copy": 0.1,
            "min_ci95_low_vs_source_rank_only_bagged": 0.1,
            "min_ci95_low_vs_score_only_bagged": 0.1,
        },
        "slice_rows": [left, right],
    }
    source_path = tmp_path / "source.json"
    _write_json(source_path, source)

    payload = audit.build_audit(
        input_path=source_path,
        output_dir=tmp_path / "out",
        run_date="2026-05-03",
    )

    assert payload["pass_gate"] is True
    assert payload["headline"]["candidate_only_raw_payload_bytes"] == 1
    assert payload["headline"]["candidate_only_framed_record_bytes"] == 4
    assert payload["headline"]["weighted_candidate_only_eval_accuracy"] == weighted_selected
    assert payload["packet_contract"]["fields"] == ["selected candidate id packed into 2 bits"]
    assert (tmp_path / "out" / "hellaswag_strict_candidate_only_packet_audit.json").exists()
    assert (tmp_path / "out" / "slice_rows.csv").exists()
