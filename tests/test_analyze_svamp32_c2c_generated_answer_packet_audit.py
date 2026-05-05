from __future__ import annotations

import json
import pathlib

from scripts import analyze_svamp32_c2c_generated_answer_packet_audit as audit


def _record(example_id: str, answer: str, prediction: str, method: str) -> dict:
    return {
        "example_id": example_id,
        "answer": [answer],
        "prediction": prediction,
        "normalized_prediction": prediction,
        "method": method,
    }


def test_generated_answer_packet_audit_marks_visible_answer_as_same_control(tmp_path: pathlib.Path) -> None:
    target_path = tmp_path / "target.jsonl"
    c2c_path = tmp_path / "c2c.jsonl"
    source_path = tmp_path / "source.jsonl"
    text_path = tmp_path / "text.jsonl"

    rows = ["row-a", "row-b"]
    target_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                _record("row-a", "5", "3", "target_alone"),
                _record("row-b", "8", "9", "target_alone"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    c2c_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                _record("row-a", "5", "5", "c2c"),
                _record("row-b", "8", "8", "c2c"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                _record("row-a", "5", "4", "source_alone"),
                _record("row-b", "8", "8", "source_alone"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    text_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                _record("row-a", "5", "7", "text_to_text"),
                _record("row-b", "8", "6", "text_to_text"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    target_set = tmp_path / "target_set.json"
    target_set.write_text(
        json.dumps(
            {
                "reference_ids": rows,
                "ids": {
                    "teacher_only": rows,
                    "clean_residual_targets": rows,
                },
                "artifacts": {
                    "source": {"label": "c2c", "method": "c2c_generate", "path": str(c2c_path)},
                    "target": {"label": "target", "method": "target_alone", "path": str(target_path)},
                    "controls": [
                        {"label": "source_alone", "method": "source_alone", "path": str(source_path)},
                        {"label": "text_to_text", "method": "text_to_text", "path": str(text_path)},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    payload = audit.analyze(
        target_set_path=target_set,
        run_date="2026-05-05",
        output_json=tmp_path / "audit.json",
        output_md=tmp_path / "audit.md",
    )

    assert payload["status"] == "generated_answer_packet_is_answer_leak_not_method"
    assert payload["condition_summaries"]["generated_answer_value_packet"]["correct_count"] == 2
    assert payload["condition_summaries"]["same_byte_visible_answer_text"]["correct_count"] == 2
    assert payload["condition_summaries"]["target_only"]["correct_count"] == 0
    assert payload["packet_contract"]["source_private"] is False
    assert (tmp_path / "manifest.json").exists()


def test_same_index_wrong_row_falls_back_to_row_shuffle_when_needed() -> None:
    rows = [
        audit.RowAudit(
            index=0,
            example_id="a",
            answer=("1",),
            gold="1",
            candidate_values=("1", "2"),
            c2c_value="1",
            c2c_index=0,
            target_value="2",
            source_value="2",
            text_value="2",
        ),
        audit.RowAudit(
            index=1,
            example_id="b",
            answer=("3",),
            gold="3",
            candidate_values=("3", "4"),
            c2c_value="4",
            c2c_index=1,
            target_value="4",
            source_value="4",
            text_value="4",
        ),
    ]

    assert audit._same_index_wrong_row_indices(rows) == [1, 0]


def test_generated_prediction_number_prefers_final_answer_over_first_rationale_number() -> None:
    row = {
        "prediction": "There were 7 bags, then 12 more. #### 19",
        "normalized_prediction": "there were 7 bags then 12 more #### 19",
    }

    assert audit._generated_prediction_number(row) == "19"
