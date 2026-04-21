from __future__ import annotations

import json
from pathlib import Path

from scripts import analyze_process_repair_attribution as attribution


def _row(
    *,
    index: int,
    method: str,
    correct: bool,
    answer: str = "gold",
    normalized_prediction: str | None = None,
    prediction: str | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "index": index,
        "method": method,
        "correct": correct,
        "answer": [answer],
        "normalized_prediction": normalized_prediction or f"{method}_{index}",
        "prediction": prediction or f"{method} prediction {index}",
        "generated_tokens": 16,
        "example_id": f"ex-{index}",
    }
    if extra:
        row.update(extra)
    return row


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _source_rows() -> list[dict[str, object]]:
    rows = [
        _row(index=0, method="target_alone", correct=False, normalized_prediction="0"),
        _row(index=1, method="target_alone", correct=True, normalized_prediction="1"),
        _row(index=2, method="target_alone", correct=True, normalized_prediction="2"),
        _row(index=3, method="target_alone", correct=False, normalized_prediction="3"),
        _row(
            index=0,
            method="selected_route_no_repair",
            correct=True,
            normalized_prediction="10",
            extra={
                "repair_pre_correct": True,
                "repair_changed_answer": False,
                "repair_selected_candidate_source": "target",
                "repair_full_oracle_correct": True,
            },
        ),
        _row(
            index=1,
            method="selected_route_no_repair",
            correct=False,
            normalized_prediction="11",
            extra={
                "repair_pre_correct": False,
                "repair_changed_answer": False,
                "repair_selected_candidate_source": "seed_0",
                "repair_full_oracle_correct": False,
            },
        ),
        _row(
            index=2,
            method="selected_route_no_repair",
            correct=True,
            normalized_prediction="12",
            extra={
                "repair_pre_correct": True,
                "repair_changed_answer": False,
                "repair_selected_candidate_source": "target",
                "repair_full_oracle_correct": True,
            },
        ),
        _row(
            index=3,
            method="selected_route_no_repair",
            correct=False,
            normalized_prediction="13",
            extra={
                "repair_pre_correct": False,
                "repair_changed_answer": False,
                "repair_selected_candidate_source": "seed_1",
                "repair_full_oracle_correct": True,
            },
        ),
        _row(
            index=0,
            method="target_self_repair",
            correct=True,
            normalized_prediction="20",
            extra={
                "repair_pre_correct": False,
                "repair_changed_answer": True,
                "repair_selected_candidate_source": "target",
                "repair_full_oracle_correct": True,
            },
        ),
        _row(
            index=1,
            method="target_self_repair",
            correct=False,
            normalized_prediction="21",
            extra={
                "repair_pre_correct": False,
                "repair_changed_answer": False,
                "repair_selected_candidate_source": "target",
                "repair_full_oracle_correct": False,
            },
        ),
        _row(
            index=2,
            method="target_self_repair",
            correct=False,
            normalized_prediction="22",
            extra={
                "repair_pre_correct": True,
                "repair_changed_answer": True,
                "repair_selected_candidate_source": "target",
                "repair_full_oracle_correct": True,
            },
        ),
        _row(
            index=3,
            method="target_self_repair",
            correct=True,
            normalized_prediction="23",
            extra={
                "repair_pre_correct": True,
                "repair_changed_answer": False,
                "repair_selected_candidate_source": "target",
                "repair_full_oracle_correct": False,
            },
        ),
        _row(
            index=0,
            method="process_repair_selected_route",
            correct=True,
            normalized_prediction="30",
            extra={
                "repair_pre_correct": False,
                "repair_changed_answer": True,
                "repair_selected_candidate_source": "seed_0",
                "repair_full_oracle_correct": True,
            },
        ),
        _row(
            index=1,
            method="process_repair_selected_route",
            correct=True,
            normalized_prediction="31",
            extra={
                "repair_pre_correct": True,
                "repair_changed_answer": False,
                "repair_selected_candidate_source": "target",
                "repair_full_oracle_correct": False,
            },
        ),
        _row(
            index=2,
            method="process_repair_selected_route",
            correct=False,
            normalized_prediction="32",
            extra={
                "repair_pre_correct": False,
                "repair_changed_answer": True,
                "repair_selected_candidate_source": "seed_1",
                "repair_full_oracle_correct": True,
            },
        ),
        _row(
            index=3,
            method="process_repair_selected_route",
            correct=False,
            normalized_prediction="33",
            extra={
                "repair_pre_correct": True,
                "repair_changed_answer": False,
                "repair_selected_candidate_source": "seed_0",
                "repair_full_oracle_correct": False,
            },
        ),
    ]
    return rows


def test_summarize_source_reports_method_and_pair_attribution(tmp_path) -> None:
    path = tmp_path / "qwen_gsm70_process_repair_controls_strict_selector_telemetry.jsonl"
    _write_jsonl(path, _source_rows())

    summary = attribution.summarize_source(path, n_bootstrap=32)

    assert summary.source == path.name
    assert summary.total_rows == 16
    assert summary.method_summaries["target_alone"]["accuracy"] == 0.5
    assert summary.method_summaries["selected_route_no_repair"]["pre_repair_accuracy"] == 0.5
    assert summary.method_summaries["selected_route_no_repair"]["changed_answer_rate"] == 0.0
    assert summary.method_summaries["target_self_repair"]["repair_help_rate"] == 0.25
    assert summary.method_summaries["target_self_repair"]["repair_harm_rate"] == 0.25
    assert summary.method_summaries["process_repair_selected_route"]["target_selection_rate"] == 0.25
    assert summary.method_summaries["process_repair_selected_route"]["full_oracle"] == 0.5

    paired = summary.paired_summaries["process_repair_selected_route"]
    assert paired["paired_n"] == 4
    assert paired["baseline_method"] == "target_alone"
    assert paired["method_only"] == 1
    assert paired["baseline_only"] == 1
    assert paired["both_correct"] == 1
    assert paired["both_wrong"] == 1
    assert paired["delta_accuracy"] == 0.0
    assert paired["delta_accuracy_ci_low"] <= paired["delta_accuracy_ci_high"]

    target_self_paired = summary.target_self_paired_summaries["process_repair_selected_route"]
    assert target_self_paired["baseline_method"] == "target_self_repair"
    assert target_self_paired["paired_n"] == 4
    assert target_self_paired["delta_accuracy"] == 0.0


def test_main_writes_markdown_and_json_for_multiple_sources(tmp_path) -> None:
    source_a = tmp_path / "qwen_gsm70_process_repair_controls_strict_selector_telemetry.jsonl"
    source_b = tmp_path / "qwen_svamp70_process_repair_controls_strict_selector_telemetry.jsonl"
    _write_jsonl(source_a, _source_rows())
    _write_jsonl(
        source_b,
        [
            _row(index=0, method="target_alone", correct=True, normalized_prediction="0"),
            _row(index=1, method="target_alone", correct=False, normalized_prediction="1"),
            _row(
                index=0,
                method="process_repair_selected_route",
                correct=True,
                normalized_prediction="30",
                extra={
                    "repair_pre_correct": False,
                    "repair_changed_answer": True,
                    "repair_selected_candidate_source": "target",
                    "repair_full_oracle_correct": True,
                },
            ),
            _row(
                index=1,
                method="process_repair_selected_route",
                correct=False,
                normalized_prediction="31",
                extra={
                    "repair_pre_correct": True,
                    "repair_changed_answer": False,
                    "repair_selected_candidate_source": "seed_0",
                    "repair_full_oracle_correct": False,
                },
            ),
        ],
    )

    output_json = tmp_path / "process_repair_attribution.json"
    output_md = tmp_path / "process_repair_attribution.md"
    attribution.main(
        [
            "--inputs",
            str(source_a),
            str(source_b),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--n-bootstrap",
            "32",
        ]
    )

    payload = json.loads(output_json.read_text())
    assert [item["source"] for item in payload["sources"]] == [source_a.name, source_b.name]
    assert payload["sources"][0]["method_summaries"]["target_alone"]["accuracy"] == 0.5
    assert payload["sources"][1]["method_summaries"]["process_repair_selected_route"]["accuracy"] == 0.5
    assert "target_self_paired_summaries" in payload["sources"][0]

    markdown = output_md.read_text()
    assert "# Process Repair Attribution" in markdown
    assert source_a.name in markdown
    assert source_b.name in markdown
    assert "Delta vs target" in markdown
    assert "Delta vs target self-repair" in markdown
