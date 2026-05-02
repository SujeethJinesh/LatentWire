from __future__ import annotations

import pytest

from scripts.analyze_svamp32_teacher_forced_connector_diagnostic import (
    _answer_continuation,
    _summarize_clean_rows,
)


def test_answer_continuation_requires_answer_placeholder() -> None:
    assert _answer_continuation("42", " Final answer: {answer}") == " Final answer: 42"
    with pytest.raises(ValueError, match="must contain"):
        _answer_continuation("42", " Final answer:")


def test_summarize_clean_rows_counts_matched_only_and_control_leak() -> None:
    rows = [
        {
            "example_id": "clean_a",
            "matched_margin": 1.5,
            "best_control_margin": 0.25,
            "matched_minus_best_control_margin": 1.25,
        },
        {
            "example_id": "clean_b",
            "matched_margin": 0.5,
            "best_control_margin": 0.75,
            "matched_minus_best_control_margin": -0.25,
        },
        {
            "example_id": "self_a",
            "matched_margin": 2.0,
            "best_control_margin": 1.0,
            "matched_minus_best_control_margin": 1.0,
        },
    ]

    summary = _summarize_clean_rows(
        rows,
        clean_ids={"clean_a", "clean_b"},
        target_self_ids={"self_a"},
        min_margin_delta=0.0,
    )

    assert summary["clean_ids_scored"] == 2
    assert summary["target_self_ids_scored"] == 1
    assert summary["matched_positive_clean_count"] == 2
    assert summary["matched_only_clean_count"] == 1
    assert summary["matched_only_clean_ids"] == ["clean_a"]
    assert summary["control_leak_clean_count"] == 1
    assert summary["control_leak_clean_ids"] == ["clean_b"]
