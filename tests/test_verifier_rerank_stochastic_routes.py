from __future__ import annotations

import json
from dataclasses import dataclass

from scripts import verifier_rerank_stochastic_routes as verifier


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


def test_parse_verifier_choice_handles_common_outputs() -> None:
    assert verifier.parse_verifier_choice("B", candidate_count=4) == 1
    assert verifier.parse_verifier_choice("I choose candidate C.", candidate_count=4) == 2
    assert verifier.parse_verifier_choice("Option D is best", candidate_count=4) == 3
    assert verifier.parse_verifier_choice("E", candidate_count=4) is None
    assert verifier.parse_verifier_choice("", candidate_count=4) is None


def test_build_verifier_prompt_contains_candidates() -> None:
    candidates = [
        {"candidate_source": "target", "normalized_prediction": "1", "prediction": "answer is 1"},
        {"candidate_source": "seed_0", "normalized_prediction": "2", "prediction": "answer is 2"},
    ]

    prompt = verifier.build_verifier_prompt(problem="What is 1+1?", candidates=candidates, max_candidate_chars=20)

    assert "What is 1+1?" in prompt
    assert "A. source=target" in prompt
    assert "B. source=seed_0" in prompt
    assert "Reply with exactly one letter" in prompt


def test_order_verifier_candidates_is_seeded_and_can_move_target() -> None:
    candidates = [
        {"candidate_source": "target", "normalized_prediction": "1"},
        {"candidate_source": "seed_0", "normalized_prediction": "2"},
        {"candidate_source": "seed_1", "normalized_prediction": "3"},
        {"candidate_source": "seed_2", "normalized_prediction": "4"},
    ]

    original = verifier.order_verifier_candidates(
        candidates,
        example_index=0,
        shuffle_labels=False,
        label_seed=7,
    )
    shuffled = verifier.order_verifier_candidates(
        candidates,
        example_index=0,
        shuffle_labels=True,
        label_seed=7,
    )
    repeated = verifier.order_verifier_candidates(
        candidates,
        example_index=0,
        shuffle_labels=True,
        label_seed=7,
    )

    assert [row["candidate_source"] for row in original] == ["target", "seed_0", "seed_1", "seed_2"]
    assert shuffled == repeated
    assert sorted(row["candidate_source"] for row in shuffled) == ["seed_0", "seed_1", "seed_2", "target"]
    assert [row["candidate_source"] for row in shuffled] != [row["candidate_source"] for row in original]


def test_verifier_rerank_records_uses_response_and_fallback() -> None:
    baseline = [_record(0, "target_alone", "target says 7", "7", False)]
    seed0 = [_record(0, "bridge", "candidate A says 5", "5", False)]
    seed1 = [_record(0, "bridge", "candidate B says 8", "8", True)]
    examples = [_Example(prompt="What is 4+4?")]

    records = verifier.verifier_rerank_records(
        [baseline + seed0, baseline + seed1],
        examples=examples,
        method="bridge",
        response_fn=lambda _idx, _prompt, _candidates: "C",
    )
    selected = next(row for row in records if row["method"] == "rerank_target_model_verifier")

    assert selected["selected_candidate_source"] == "seed_1"
    assert selected["correct"] is True
    assert selected["verifier_choice_label"] == "C"
    assert selected["verifier_fallback_used"] is False

    fallback_records = verifier.verifier_rerank_records(
        [baseline + seed0],
        examples=examples,
        method="bridge",
        response_fn=lambda _idx, _prompt, _candidates: "not sure",
    )
    fallback_selected = next(row for row in fallback_records if row["method"] == "rerank_target_model_verifier")
    assert fallback_selected["selected_candidate_source"] == "target"
    assert fallback_selected["verifier_fallback_used"] is True


def test_verifier_rerank_records_maps_shuffled_labels_to_candidates() -> None:
    baseline = [_record(0, "target_alone", "target says 7", "7", False)]
    seed0 = [_record(0, "bridge", "candidate A says 5", "5", False)]
    seed1 = [_record(0, "bridge", "candidate B says 8", "8", True)]
    examples = [_Example(prompt="What is 4+4?")]
    observed_sources: list[str] = []

    def choose_seed_1(_idx: int, _prompt: str, candidates: list[dict[str, object]]) -> str:
        observed_sources.extend(str(row["candidate_source"]) for row in candidates)
        for label, row in zip(("A", "B", "C", "D"), candidates):
            if row["candidate_source"] == "seed_1":
                return label
        raise AssertionError("seed_1 missing from shuffled candidates")

    records = verifier.verifier_rerank_records(
        [baseline + seed0, baseline + seed1],
        examples=examples,
        method="bridge",
        response_fn=choose_seed_1,
        shuffle_labels=True,
        label_seed=11,
    )
    selected = next(row for row in records if row["method"] == "rerank_target_model_verifier")

    assert selected["selected_candidate_source"] == "seed_1"
    assert selected["verifier_choice_candidate_source"] == "seed_1"
    assert selected["verifier_labels_shuffled"] is True
    assert selected["verifier_label_seed"] == 11
    assert selected["verifier_target_label"] in {"A", "B", "C"}
    assert selected["reranking_policy"] == "target_model_listwise_verifier_randomized"
    assert observed_sources != ["target", "seed_0", "seed_1"]


def test_summary_and_markdown(tmp_path) -> None:
    baseline = [_record(0, "target_alone", "target says 7", "7", False)]
    seed0 = [_record(0, "bridge", "candidate says 8", "8", True)]
    examples = [_Example(prompt="What is 4+4?")]
    records = verifier.verifier_rerank_records(
        [baseline + seed0],
        examples=examples,
        method="bridge",
        response_fn=lambda _idx, _prompt, _candidates: "B",
    )
    results = verifier.summarize_results(records)
    output = tmp_path / "verifier.jsonl"
    markdown = tmp_path / "verifier.md"
    verifier.write_prediction_records(output, records)
    verifier.write_markdown_summary(results, markdown)

    loaded = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(loaded) == 2
    assert results["rerank_target_model_verifier"] == 1.0
    assert results["rerank_target_model_verifier_choice_a_rate"] == 0.0
    assert results["rerank_target_model_verifier_target_selection_rate"] == 0.0
    assert "Target-Model Verifier Reranker Summary" in markdown.read_text()
