from __future__ import annotations

import json

from scripts import build_source_private_hellaswag_qwen_to_phi_error_repair_audit as audit


def _qwen_row(row_id: int, *, answer: int, hybrid: int, selected: int) -> dict[str, object]:
    return {
        "answer_index": answer,
        "candidate_roll_hidden_prediction": (hybrid + 1) % 4,
        "hidden_mean_prediction": hybrid,
        "row_id": str(row_id),
        "score_channel_roll_hidden_prediction": (hybrid + 2) % 4,
        "score_mean_prediction": hybrid,
        "score_only_bagged_prediction": hybrid,
        "score_vote_prediction": hybrid,
        "selected_margin": 0.25 + 0.1 * (row_id % 4),
        "selected_prediction": selected,
        "source_label_prediction": selected,
        "source_rank_only_bagged_prediction": selected,
        "trained_label_prediction": selected,
        "vote_prediction": hybrid,
        "wrong_example_hidden_prediction": (hybrid + 3) % 4,
        "zero_hidden_prediction": 0,
    }


def _write_slice(tmp_path, *, name: str, start_row_id: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    qwen_path = tmp_path / f"{name}_qwen.jsonl"
    phi_path = tmp_path / f"{name}_phi.json"
    rows: list[dict[str, object]] = []
    phi_predictions: list[int] = []
    phi_scores: list[list[float]] = []
    for offset in range(8):
        answer = offset % 4
        hybrid = answer if offset % 3 else (answer + 1) % 4
        selected = hybrid if offset % 4 else (answer + 2) % 4
        row_id = start_row_id + offset
        rows.append(_qwen_row(row_id, answer=answer, hybrid=hybrid, selected=selected))
        phi_prediction = answer if offset % 5 == 1 else (answer + 1) % 4
        phi_predictions.append(phi_prediction)
        scores = [-3.0, -3.5, -4.0, -4.5]
        scores[phi_prediction] = -1.0
        scores[(answer + 2) % 4] = -1.5
        phi_scores.append(scores)
    qwen_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    phi_path.write_text(
        json.dumps(
            {
                "row_count": len(rows),
                "row_ids": [str(row["row_id"]) for row in rows],
                "source_predictions": phi_predictions,
                "source_scores": phi_scores,
                "source_model": {"kind": "synthetic_phi"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return (
        {
            "slice_start": start_row_id,
            "slice_end_exclusive": start_row_id + len(rows),
            "qwen_predictions": qwen_path,
            "phi_target_score_cache": phi_path,
        },
        rows,
    )


def test_error_repair_audit_writes_decision_surface(tmp_path) -> None:
    slice_a, rows_a = _write_slice(tmp_path, name="a", start_row_id=1000)
    slice_b, rows_b = _write_slice(tmp_path, name="b", start_row_id=2000)
    all_rows = rows_a + rows_b
    score_path = tmp_path / "qwen_source_scores.json"
    source_scores: list[list[float]] = []
    for row in all_rows:
        answer = int(row["answer_index"])
        scores = [-4.0, -4.5, -5.0, -5.5]
        scores[answer] = -1.0
        scores[(answer + 1) % 4] = -1.25
        source_scores.append(scores)
    score_path.write_text(
        json.dumps(
            {
                "row_count": len(all_rows),
                "row_ids": [str(row["row_id"]) for row in all_rows],
                "source_predictions": [int(row["answer_index"]) for row in all_rows],
                "source_scores": source_scores,
                "source_model": {"kind": "synthetic_qwen"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = audit.build_audit(
        output_dir=tmp_path / "out",
        slices=(slice_a, slice_b),
        source_score_cache=score_path,
        fit_rows_per_slice=2,
        select_rows_per_slice=2,
        bootstrap_samples=50,
        run_date="2026-05-04",
    )

    assert payload["audit_only"] is True
    assert payload["headline"]["eval_rows"] == 8
    assert payload["packet_contract"]["source_text_exposed"] is False
    assert payload["packet_contract"]["raw_scores_or_logits_transmitted"] is False
    assert payload["source_score_metadata"]["source_score_cache_rows"] == len(all_rows)
    method_names = {row["method"] for row in payload["method_rows"]}
    assert "fixed_hybrid_or_qwen_top2_oracle_diagnostic" in method_names
    assert "source_row_shuffle_top2_oracle_control" in method_names
    partitions = {row["partition"] for row in payload["partition_rows"]}
    assert "fixed_hybrid_wrong_source_unique_top2" in partitions
    assert (tmp_path / "out" / "hellaswag_qwen_to_phi_error_repair_audit.json").exists()
    assert (tmp_path / "out" / "method_rows.csv").exists()
    assert (tmp_path / "out" / "partition_rows.csv").exists()
    assert (tmp_path / "out" / "manifest.json").exists()
