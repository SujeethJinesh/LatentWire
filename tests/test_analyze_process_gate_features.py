from __future__ import annotations

import json
from pathlib import Path

from scripts import analyze_process_gate_features as audit


def _row(index: int, method: str, correct: bool, **extra: object) -> dict[str, object]:
    row: dict[str, object] = {
        "index": index,
        "method": method,
        "correct": correct,
        "example_id": f"ex-{index}",
    }
    row.update(extra)
    return row


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def test_extract_process_features_scores_valid_equations_and_finished_answers() -> None:
    row = {
        "prediction": "We compute 16 - 7 = 9.\nThen 9 * 2 = 18.\nFinal answer: 18",
        "normalized_prediction": "18",
        "candidate_format_score": 3.0,
    }

    features = audit.extract_process_features(row)

    assert features["equation_count"] == 2.0
    assert features["valid_equation_count"] == 2.0
    assert features["equation_valid_fraction"] == 1.0
    assert features["answer_marker_score"] == 1.0
    assert features["finished_tail_score"] == 1.0
    assert features["prediction_tail_match_score"] == 1.0
    assert features["format_plus_process_score"] > features["process_completeness_score"]


def test_audit_source_finds_process_gate(tmp_path: Path) -> None:
    source = tmp_path / "repair.jsonl"
    rows: list[dict[str, object]] = []
    selected_correct = [True, True, False, False]
    repair_correct = [True, True, True, False]
    predictions = [
        "1 + 1 = 2. Final answer: 2",
        "3 + 4 = 7. Answer: 7",
        "We have 9 - 2 = 7. Remaining",
        "The result is unclear",
    ]
    for idx in range(4):
        rows.append(_row(idx, "target_alone", correct=False))
        rows.append(
            _row(
                idx,
                "selected_route_no_repair",
                correct=selected_correct[idx],
                prediction=predictions[idx],
                normalized_prediction=str([2, 7, 7, 0][idx]),
                candidate_format_score=[4.0, 3.5, 1.0, 1.0][idx],
            )
        )
        rows.append(_row(idx, "process_repair_selected_route", correct=repair_correct[idx]))
    _write_jsonl(source, rows)

    result = audit.audit_source(source, features=["process_completeness_score"])

    assert result.source == source.name
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row["feature"] == "process_completeness_score"
    assert row["n"] == 4
    assert row["selected_correct_auroc"] == 1.0
    assert row["best_gate_accuracy"] == 0.75
    assert row["best_gate_saved_repair_rate"] == 0.5
    assert row["best_gate_missed_help"] == 0


def test_main_writes_json_and_markdown(tmp_path: Path) -> None:
    source = tmp_path / "repair.jsonl"
    output_json = tmp_path / "process_gate.json"
    output_md = tmp_path / "process_gate.md"
    rows = [
        _row(0, "target_alone", correct=False),
        _row(
            0,
            "selected_route_no_repair",
            correct=True,
            prediction="2 + 2 = 4. Final answer: 4",
            normalized_prediction="4",
        ),
        _row(0, "process_repair_selected_route", correct=True),
    ]
    _write_jsonl(source, rows)

    payload = audit.main(
        [
            "--inputs",
            str(source),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--features",
            "process_completeness_score",
        ]
    )

    assert json.loads(output_json.read_text()) == payload
    markdown = output_md.read_text()
    assert "# Process Gate Feature Audit" in markdown
    assert "process_completeness_score" in markdown
