from __future__ import annotations

import json
import pathlib

from scripts import analyze_svamp32_syndrome_sidecar_probe as probe


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_syndrome_probe_separates_matched_from_controls(tmp_path: pathlib.Path) -> None:
    target_rows = [
        {
            "example_id": "clean_a",
            "method": "target_alone",
            "answer": "6",
            "prediction": "Answer: 1",
            "normalized_prediction": "1",
            "correct": False,
        },
        {
            "example_id": "clean_b",
            "method": "target_alone",
            "answer": "7",
            "prediction": "Answer: 2",
            "normalized_prediction": "2",
            "correct": False,
        },
        {
            "example_id": "self",
            "method": "target_alone",
            "answer": "8",
            "prediction": "Answer: 0",
            "normalized_prediction": "0",
            "correct": False,
        },
    ]
    teacher_rows = [
        {
            **row,
            "method": "c2c",
            "prediction": f"Answer: {row['answer']}",
            "normalized_prediction": row["answer"],
            "correct": True,
        }
        for row in target_rows
    ]
    target_self_rows = [
        {
            **row,
            "method": "target_self_repair",
            "prediction": "candidates mention 6 and 7; final answer: 0"
            if row["example_id"] != "self"
            else "final answer: 8",
            "normalized_prediction": "0" if row["example_id"] != "self" else "8",
            "correct": row["example_id"] == "self",
        }
        for row in target_rows
    ]
    target_path = tmp_path / "target.jsonl"
    teacher_path = tmp_path / "teacher.jsonl"
    target_self_path = tmp_path / "target_self.jsonl"
    target_set_path = tmp_path / "target_set.json"
    _write_jsonl(target_path, target_rows)
    _write_jsonl(teacher_path, teacher_rows)
    _write_jsonl(target_self_path, target_self_rows)
    target_set_path.write_text(
        json.dumps(
            {
                "ids": {
                    "teacher_only": ["clean_a", "clean_b", "self"],
                    "clean_residual_targets": ["clean_a", "clean_b"],
                    "target_self_repair": ["self"],
                }
            }
        ),
        encoding="utf-8",
    )

    payload = probe.analyze(
        target_spec=probe.RowSpec("target_alone", target_path, "target_alone"),
        teacher_spec=probe.RowSpec("c2c", teacher_path, "c2c_generate"),
        candidate_specs=[
            probe.RowSpec("target_self_repair", target_self_path, "target_self_repair")
        ],
        target_set_path=target_set_path,
        moduli_sets=[[11]],
        fallback_label="target_self_repair",
        shuffle_offset=1,
        min_correct=3,
        min_clean_source_necessary=2,
        min_numeric_coverage=3,
        run_date="2026-04-24",
    )

    run = payload["runs"][0]
    assert payload["status"] == "syndrome_sidecar_bound_clears_gate_not_method"
    assert run["condition_summaries"]["matched"]["correct_count"] == 3
    assert run["condition_summaries"]["zero_source"]["clean_correct_count"] == 0
    assert run["condition_summaries"]["shuffled_source"]["clean_correct_count"] == 0
    assert run["condition_summaries"]["target_only"]["clean_correct_count"] == 0
    assert run["source_necessary_clean_ids"] == ["clean_a", "clean_b"]


def test_parse_moduli_set_rejects_invalid_values() -> None:
    assert probe._parse_moduli_set("2,3,5") == [2, 3, 5]
    try:
        probe._parse_moduli_set("1")
    except Exception as exc:
        assert "moduli must be >1" in str(exc)
    else:
        raise AssertionError("invalid modulus should raise")
