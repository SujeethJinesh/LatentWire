from __future__ import annotations

import json
from dataclasses import dataclass

from scripts import process_repair_routes as repair


@dataclass
class _Example:
    prompt: str
    answers: list[str]


def _record(index: int, method: str, prediction: str, normalized: str, correct: bool) -> dict[str, object]:
    return {
        "index": index,
        "method": method,
        "prediction": prediction,
        "normalized_prediction": normalized,
        "correct": correct,
        "answer": [normalized if correct else "gold"],
        "generated_tokens": 20,
    }


def test_build_repair_prompt_contains_problem_candidate_and_final_contract() -> None:
    prompt = repair.build_repair_prompt(
        problem="What is 2+2?",
        candidate={"prediction": "2+2=5. Final answer: 5", "normalized_prediction": "5"},
        max_candidate_chars=20,
    )

    assert "What is 2+2?" in prompt
    assert "Candidate extracted final answer: '5'" in prompt
    assert "Final answer: <number>" in prompt
    assert "2+2=5" in prompt


def test_process_repair_records_logs_pre_post_and_help() -> None:
    baseline = [_record(0, "target_alone", "target says 5", "5", False)]
    seed0 = [_record(0, "bridge", "candidate says 5", "5", False)]
    seed1 = [_record(0, "bridge", "candidate says 4", "4", True)]
    examples = [_Example(prompt="What is 2+2?", answers=["4"])]

    def repair_to_four(_idx: int, _prompt: str, selected: dict[str, object], _candidates: list[dict[str, object]]) -> str:
        assert selected["candidate_source"] == "target"
        return "The first arithmetic step should be corrected. Final answer: 4"

    records = repair.process_repair_records(
        [baseline + seed0, baseline + seed1],
        examples=examples,
        method="bridge",
        response_fn=repair_to_four,
        selection_policy="target_on_strict_format",
    )
    selected = next(row for row in records if row["method"] == "process_repair_selected_route")

    assert selected["repair_pre_correct"] is False
    assert selected["correct"] is True
    assert selected["normalized_prediction"] == "4"
    assert selected["repair_changed_answer"] is True
    assert selected["repair_selected_candidate_source"] == "target"
    assert selected["repair_full_oracle_correct"] is True

    results = repair.summarize_results(records)
    assert results["process_repair_selected_route"] == 1.0
    assert results["process_repair_selected_route_pre_repair_accuracy"] == 0.0
    assert results["process_repair_selected_route_repair_help_rate"] == 1.0
    assert results["process_repair_selected_route_repair_harm_rate"] == 0.0


def test_process_repair_summary_and_jsonl(tmp_path) -> None:
    baseline = [_record(0, "target_alone", "target says 4", "4", True)]
    seed = [_record(0, "bridge", "candidate says 5", "5", False)]
    examples = [_Example(prompt="What is 2+2?", answers=["4"])]
    records = repair.process_repair_records(
        [baseline + seed],
        examples=examples,
        method="bridge",
        response_fn=lambda _idx, _prompt, _selected, _candidates: "Already correct. Final answer: 4",
    )
    results = repair.summarize_results(records)
    output = tmp_path / "repair.jsonl"
    markdown = tmp_path / "repair.md"

    repair.write_prediction_records(output, records)
    repair.write_markdown_summary(results, markdown)

    loaded = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(loaded) == 2
    assert "Process Repair Route Summary" in markdown.read_text()
