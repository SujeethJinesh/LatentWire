from __future__ import annotations

import json
from dataclasses import dataclass

from scripts import pairwise_verifier_rerank_routes as pairwise


@dataclass
class _Example:
    prompt: str


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


def test_parse_pairwise_winner_handles_common_outputs() -> None:
    assert pairwise.parse_pairwise_winner("L") == 0
    assert pairwise.parse_pairwise_winner("right") == 1
    assert pairwise.parse_pairwise_winner("I choose the left candidate.") == 0
    assert pairwise.parse_pairwise_winner("Candidate right is better") == 1
    assert pairwise.parse_pairwise_winner("Maybe neither") is None


def test_build_pairwise_prompt_contains_candidates() -> None:
    left = {"candidate_source": "target", "normalized_prediction": "1", "prediction": "answer is 1"}
    right = {"candidate_source": "seed_0", "normalized_prediction": "2", "prediction": "answer is 2"}

    prompt = pairwise.build_pairwise_prompt(problem="What is 1+1?", left_candidate=left, right_candidate=right)

    assert "What is 1+1?" in prompt
    assert "Left candidate:" in prompt
    assert "source=target" in prompt
    assert "Right candidate:" in prompt
    assert "source=seed_0" in prompt
    assert "Reply with exactly one letter" in prompt


def test_order_pairwise_candidates_is_seeded_and_can_move_target() -> None:
    candidates = [
        {"candidate_source": "target", "normalized_prediction": "1"},
        {"candidate_source": "seed_0", "normalized_prediction": "2"},
        {"candidate_source": "seed_1", "normalized_prediction": "3"},
        {"candidate_source": "seed_2", "normalized_prediction": "4"},
    ]

    original = pairwise.order_pairwise_candidates(
        candidates,
        example_index=0,
        pair_order_seed=7,
    )
    shuffled = pairwise.order_pairwise_candidates(
        candidates,
        example_index=0,
        pair_order_seed=7,
    )

    assert [row["candidate_source"] for row in original] == [row["candidate_source"] for row in shuffled]
    assert sorted(row["candidate_source"] for row in shuffled) == ["seed_0", "seed_1", "seed_2", "target"]
    assert [row["candidate_source"] for row in shuffled] != [row["candidate_source"] for row in candidates]


def test_pairwise_tournament_selects_best_source_and_logs_telemetry() -> None:
    baseline = [_record(0, "target_alone", "target says 7", "7", False)]
    seed0 = [_record(0, "bridge", "candidate A says 5", "5", False)]
    seed1 = [_record(0, "bridge", "candidate B says 8", "8", True)]
    seed2 = [_record(0, "bridge", "candidate C says 6", "6", False)]
    examples = [_Example(prompt="What is 4+4?")]
    observed_sources: list[list[str]] = []

    def choose_correct(_idx: int, _prompt: str, left: dict[str, object], right: dict[str, object]) -> str:
        observed_sources.append([str(left["candidate_source"]), str(right["candidate_source"])])
        if bool(left.get("correct")) and not bool(right.get("correct")):
            return "L"
        if bool(right.get("correct")) and not bool(left.get("correct")):
            return "R"
        return "L"

    records = pairwise.pairwise_verifier_rerank_records(
        [baseline + seed0, baseline + seed1, baseline + seed2],
        examples=examples,
        method="bridge",
        response_fn=choose_correct,
        pair_order_seed=11,
    )
    selected = next(row for row in records if row["method"] == "pairwise_verifier_tournament")

    assert selected["selected_candidate_source"] == "seed_1"
    assert selected["pairwise_selected_candidate_source"] == "seed_1"
    assert selected["pairwise_total_comparisons"] == 3
    assert selected["pairwise_budget_exhausted"] is False
    assert selected["pairwise_win_counts"]["seed_1"] >= 1
    assert len(selected["pairwise_comparisons"]) == 3
    assert all("compared_sources" in row for row in selected["pairwise_comparisons"])
    assert all("raw_response" in row for row in selected["pairwise_comparisons"])
    assert observed_sources != [["target", "seed_0"], ["seed_1", "seed_2"]]


def test_pairwise_tournament_fallback_and_summary(tmp_path) -> None:
    baseline = [_record(0, "target_alone", "target says 7", "7", True)]
    seed = [_record(0, "bridge", "candidate says 8", "8", False)]
    examples = [_Example(prompt="What is 4+3?")]

    records = pairwise.pairwise_verifier_rerank_records(
        [baseline + seed],
        examples=examples,
        method="bridge",
        response_fn=lambda _idx, _prompt, _left, _right: "not sure",
    )
    selected = next(row for row in records if row["method"] == "pairwise_verifier_tournament")

    assert selected["selected_candidate_source"] == "target"
    assert selected["pairwise_fallback_rate"] == 1.0
    assert selected["pairwise_target_was_left_rate"] in {0.0, 1.0}

    results = pairwise.summarize_results(records)
    output = tmp_path / "pairwise.jsonl"
    markdown = tmp_path / "pairwise.md"
    pairwise.write_prediction_records(str(output), records)
    pairwise.write_markdown_summary(results, markdown)

    loaded = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(loaded) == 2
    assert results["pairwise_verifier_tournament"] == 1.0
    assert results["pairwise_verifier_tournament_fallback_rate"] == 1.0
    assert results["pairwise_verifier_tournament_target_selection_rate"] == 1.0
    assert "Pairwise Verifier Tournament Summary" in markdown.read_text(encoding="utf-8")
