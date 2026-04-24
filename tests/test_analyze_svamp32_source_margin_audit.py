from __future__ import annotations

import json

from scripts import analyze_svamp32_source_margin_audit as audit


def test_source_margin_summary_separates_final_answer_and_margin_signal() -> None:
    rows = [
        {
            "example_id": "clean_source_margin",
            "source_alone_correct": False,
            "text_to_text_correct": False,
            "source_margin": 2.0,
            "target_margin": 0.5,
            "source_minus_target_margin": 1.5,
        },
        {
            "example_id": "clean_final_answer",
            "source_alone_correct": True,
            "text_to_text_correct": False,
            "source_margin": -1.0,
            "target_margin": -2.0,
            "source_minus_target_margin": 1.0,
        },
        {
            "example_id": "target_self",
            "source_alone_correct": False,
            "text_to_text_correct": False,
            "source_margin": 0.25,
            "target_margin": 0.0,
            "source_minus_target_margin": 0.25,
        },
    ]

    summary = audit._summarize_rows(
        rows,
        clean_ids={"clean_source_margin", "clean_final_answer"},
        target_self_ids={"target_self"},
        min_margin_delta=0.0,
    )

    assert summary["clean_ids_scored"] == 2
    assert summary["target_self_ids_scored"] == 1
    assert summary["source_final_clean_correct_ids"] == ["clean_final_answer"]
    assert summary["source_margin_positive_clean_ids"] == ["clean_source_margin"]
    assert summary["source_margin_advantage_clean_ids"] == [
        "clean_source_margin",
        "clean_final_answer",
    ]
    assert summary["source_margin_positive_advantage_clean_ids"] == [
        "clean_source_margin"
    ]
    assert summary["mean_source_minus_target_margin_clean"] == 1.25


def test_method_records_accepts_c2c_alias_and_checks_methods(tmp_path) -> None:
    path = tmp_path / "preds.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"example_id": "a", "method": "c2c"},
                {"example_id": "b", "method": "source_alone"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert audit._method_records(path, "c2c_generate")[0]["example_id"] == "a"
    assert audit._method_records(path, "source_alone")[0]["example_id"] == "b"


def test_answer_continuation_requires_answer_placeholder() -> None:
    assert audit._answer_continuation("42", " answer={answer}") == " answer=42"
    try:
        audit._answer_continuation("42", "answer")
    except ValueError as exc:
        assert "must contain {answer}" in str(exc)
    else:
        raise AssertionError("missing placeholder should raise")
