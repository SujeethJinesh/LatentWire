from __future__ import annotations

import json
from pathlib import Path

from scripts import analyze_repair_gate_features as audit


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


def _fixture_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    selected_correct = [True, True, False, False]
    repair_correct = [True, True, True, False]
    feature_values = [4.0, 3.5, 1.0, 2.0]
    for idx in range(4):
        rows.append(_row(idx, "target_alone", correct=False))
        rows.append(
            _row(
                idx,
                "selected_route_no_repair",
                correct=selected_correct[idx],
                candidate_format_score=feature_values[idx],
                candidate_completion_score=feature_values[idx] + 0.25,
            )
        )
        rows.append(_row(idx, "process_repair_selected_route", correct=repair_correct[idx]))
    return rows


def test_audit_source_scores_features_and_best_gate(tmp_path: Path) -> None:
    source = tmp_path / "repair.jsonl"
    _write_jsonl(source, _fixture_rows())

    result = audit.audit_source(source, features=["candidate_format_score", "missing"])

    assert result.source == source.name
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row["feature"] == "candidate_format_score"
    assert row["n"] == 4
    assert row["selected_correct_auroc"] == 1.0
    assert row["repair_help_count"] == 1
    assert row["best_gate_accuracy"] == 0.75
    assert row["best_gate_saved_repair_rate"] == 0.75
    assert row["best_gate_missed_help"] == 0


def test_main_writes_json_and_markdown(tmp_path: Path) -> None:
    source = tmp_path / "repair.jsonl"
    output_json = tmp_path / "audit.json"
    output_md = tmp_path / "audit.md"
    _write_jsonl(source, _fixture_rows())

    payload = audit.main(
        [
            "--inputs",
            str(source),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--features",
            "candidate_format_score",
        ]
    )

    assert json.loads(output_json.read_text()) == payload
    markdown = output_md.read_text()
    assert "# Repair Gate Feature Audit" in markdown
    assert "candidate_format_score" in markdown
