from __future__ import annotations

import json
from pathlib import Path

from scripts import analyze_test_before_repair_policy as tbr


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
    selected_correct = [True, False, False, True]
    repair_correct = [True, True, False, True]
    target_correct = [False, False, False, True]
    target_self_correct = [True, False, False, True]
    format_scores = [2.0, -1.0, 2.0, 2.0]
    completion_scores = [2.0, 0.0, 2.0, 2.0]
    vote_margins = [2, 0, 1, 2]
    for idx in range(4):
        rows.append(_row(idx, "target_alone", target_correct[idx]))
        rows.append(
            _row(
                idx,
                "selected_route_no_repair",
                selected_correct[idx],
                candidate_format_score=format_scores[idx],
                candidate_completion_score=completion_scores[idx],
                selected_candidate_format_delta_vs_target=0.0 if idx != 1 else -1.0,
                candidate_vote_margin=vote_margins[idx],
                prediction=(
                    "1 + 1 = 2. Final answer: 2"
                    if selected_correct[idx]
                    else "Incomplete equation 1 +"
                ),
                normalized_prediction="2" if selected_correct[idx] else "1",
            )
        )
        rows.append(_row(idx, "target_self_repair", target_self_correct[idx]))
        rows.append(_row(idx, "process_repair_selected_route", repair_correct[idx]))
    return rows


def test_summarize_source_includes_budget_and_missed_help(tmp_path: Path) -> None:
    source = tmp_path / "repair_controls.jsonl"
    _write_jsonl(source, _fixture_rows())

    summary = tbr.summarize_source(source)
    rows = {row["policy"]: row for row in summary.rows if row["threshold"] is None}

    assert rows["never_repair_selected"]["accuracy"] == 0.5
    assert rows["never_repair_selected"]["repair_application_rate"] == 0.0
    assert rows["repair_all_selected"]["accuracy"] == 0.75
    assert rows["repair_all_selected"]["repair_application_rate"] == 1.0
    assert rows["oracle_precheck_analysis_only"]["accuracy"] == 0.75
    assert rows["oracle_precheck_analysis_only"]["repair_application_rate"] == 0.5
    assert rows["oracle_precheck_analysis_only"]["repair_saved_rate_vs_repair_all"] == 0.5

    format_rows = [row for row in summary.rows if row["policy"] == "format_gate"]
    assert format_rows
    best_format = max(format_rows, key=lambda row: row["accuracy"])
    assert best_format["missed_help_count"] == 0
    assert best_format["repaired_help_count"] == 1

    process_rows = [row for row in summary.rows if row["policy"] == "process_gate"]
    assert process_rows
    assert any(row["missed_help_count"] == 0 for row in process_rows)


def test_main_writes_json_and_markdown(tmp_path: Path) -> None:
    source = tmp_path / "repair_controls.jsonl"
    output_json = tmp_path / "tbr.json"
    output_md = tmp_path / "tbr.md"
    _write_jsonl(source, _fixture_rows())

    payload = tbr.main(
        [
            "--inputs",
            str(source),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--top-k",
            "5",
        ]
    )

    loaded = json.loads(output_json.read_text())
    assert loaded == payload
    assert loaded["sources"][0]["source"] == source.name
    markdown = output_md.read_text()
    assert "# Test-Before-Repair Policy Analysis" in markdown
    assert "Delta vs target self" in markdown
