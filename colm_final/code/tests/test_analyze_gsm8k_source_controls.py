from __future__ import annotations

from scripts import analyze_gsm8k_source_controls as source_controls


def _record(example_id: str, method: str, prediction: str, correct: bool, index: int) -> dict:
    return {
        "example_id": example_id,
        "method": method,
        "prediction": prediction,
        "answer": ["1"],
        "correct": correct,
        "index": index,
    }


def test_apply_target_fallback_only_replaces_empty_or_nonnumeric_outputs() -> None:
    target = [
        _record("a", "target_alone", "answer is 1", True, 0),
        _record("b", "target_alone", "answer is 2", False, 1),
    ]
    control = [
        _record("a", "rotalign_kv", "", False, 0),
        _record("b", "rotalign_kv", "answer is 7", False, 1),
    ]

    patched = source_controls._apply_target_fallback(control, target)

    assert patched[0]["target_fallback_used"] is True
    assert patched[0]["prediction"] == "answer is 1"
    assert patched[0]["correct"] is True
    assert patched[0]["method"] == "rotalign_kv"
    assert patched[0]["original_prediction_before_target_fallback"] == ""
    assert patched[1]["target_fallback_used"] is False
    assert patched[1]["prediction"] == "answer is 7"


def test_source_control_gate_allows_target_fallback_valid_controls() -> None:
    target_summary = {"correct": 1}
    live_summary = {"correct": 2}
    control_summaries = [
        {
            "label": "shuffled_source_salt0",
            "n": 2,
            "correct": 1,
            "empty_predictions": 0,
            "numeric_extraction_coverage": 2,
            "ordered_id_parity": True,
            "paired_vs_target": {"win": 0, "loss": 0, "tie": 2},
            "live_win_retention_count": 0,
        }
    ]

    gate = source_controls._source_control_gate(
        target_correct=target_summary["correct"],
        live_summary=live_summary,
        control_summaries=control_summaries,
    )

    assert gate["status"] == "source_controls_support_matched_source_signal"
    assert gate["decisive_control_labels"] == ["shuffled_source_salt0"]


def test_target_fallback_preserves_id_parity_and_target_records() -> None:
    target = [
        _record("a", "target_alone", "answer is 1", True, 0),
        _record("b", "target_alone", "answer is 2", False, 1),
    ]
    target_before = [dict(row) for row in target]
    control = [
        _record("a", "rotalign_kv", "", False, 0),
        _record("b", "rotalign_kv", "no numeric answer", False, 1),
    ]

    patched = source_controls._apply_target_fallback(control, target)
    summary = source_controls._row_summary(
        label="control",
        records=patched,
        reference_ids=["a", "b"],
        target_records=target,
        live_records=target,
    )

    assert target == target_before
    assert summary["ordered_id_parity"] is True
    assert summary["set_id_parity"] is True
    assert summary["empty_predictions"] == 0
    assert summary["numeric_extraction_coverage"] == 2
    assert summary["telemetry"]["target_fallback_used"] == 2
