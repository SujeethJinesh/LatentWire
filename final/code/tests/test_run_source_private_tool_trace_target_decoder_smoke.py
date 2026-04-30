from __future__ import annotations

import torch

from scripts import run_source_private_hidden_repair_packet_smoke as repair_gate
from scripts import run_source_private_tool_trace_target_decoder_smoke as target_gate


def _example() -> target_gate.LoadedExample:
    source = repair_gate.make_benchmark(examples=2, candidates=4, seed=29)[1]
    return target_gate.LoadedExample(
        example_id=source.example_id,
        answer_label=source.answer_label,
        diagnostic_code=source.diagnostic_code,
        candidates=tuple(
            {
                "label": candidate.label,
                "handles_diagnostic": candidate.handles_diagnostic,
                "prior_score": candidate.prior_score,
            }
            for candidate in source.candidates
        ),
    )


def test_prompt_uses_packet_metadata_and_prior_without_answer_label() -> None:
    example = _example()
    prompt = target_gate._prompt_for_target_decoder(example, payload=example.diagnostic_code)

    assert "handles_repair_diag" in prompt
    assert f"Source packet: {example.diagnostic_code}" in prompt
    assert f"Target-prior label: {target_gate._prior_prediction(example)}" in prompt
    assert f"answer_label: {example.answer_label}" not in prompt


def test_choice_alias_prompt_uses_option_letters_without_answer_label() -> None:
    example = _example()
    prompt = target_gate._prompt_for_target_decoder(
        example,
        payload=example.diagnostic_code,
        prompt_mode="choice_alias",
    )

    assert "Return only A, B, C, or D" in prompt
    assert f"Source packet: {example.diagnostic_code}" in prompt
    assert "Target-prior option:" in prompt
    assert f"answer_label: {example.answer_label}" not in prompt


def test_condition_payloads_keep_structured_relays_at_two_bytes() -> None:
    examples = [_example(), _example()]

    matched, _ = target_gate._condition_payload(
        condition="matched_packet", example=examples[0], examples=examples, index=0, rng=target_gate.random.Random(7)
    )
    json_payload, _ = target_gate._condition_payload(
        condition="structured_json_2byte", example=examples[0], examples=examples, index=0, rng=target_gate.random.Random(7)
    )
    text_payload, _ = target_gate._condition_payload(
        condition="structured_free_text_2byte", example=examples[0], examples=examples, index=0, rng=target_gate.random.Random(7)
    )
    random_noncandidate, metadata = target_gate._condition_payload(
        condition="random_noncandidate_same_byte",
        example=examples[0],
        examples=examples,
        index=0,
        rng=target_gate.random.Random(7),
    )

    assert matched == examples[0].diagnostic_code
    assert len(json_payload.encode("utf-8")) == 2
    assert len(text_payload.encode("utf-8")) == 2
    assert json_payload != examples[0].diagnostic_code
    assert text_payload != examples[0].diagnostic_code
    assert metadata["packet_kind"] == "random_noncandidate_diag"
    assert random_noncandidate not in {candidate["handles_diagnostic"] for candidate in examples[0].candidates}


def test_deranged_candidate_table_rotates_handles_without_relabeling() -> None:
    example = _example()
    deranged = target_gate._candidate_table_for_condition(example, condition="deranged_candidate_diag_table")
    original_handles = [candidate["handles_diagnostic"] for candidate in example.candidates]
    deranged_handles = [candidate["handles_diagnostic"] for candidate in deranged]

    assert [candidate["label"] for candidate in deranged] == [candidate["label"] for candidate in example.candidates]
    assert deranged_handles == original_handles[1:] + original_handles[:1]
    assert target_gate._candidate_table_for_condition(example, condition="matched_packet") == example.candidates


def test_parse_candidate_label_accepts_exact_or_embedded_label() -> None:
    example = _example()
    label = example.candidates[0]["label"]

    assert target_gate._parse_candidate_label(label, example) == label
    assert target_gate._parse_candidate_label(f"{label}\n", example) == label
    assert target_gate._parse_candidate_label(f"The answer is {label}.", example) == label
    assert target_gate._parse_candidate_label("candidate_9999_patch_0_repair_record", example) == ""


def test_parse_candidate_label_accepts_choice_alias() -> None:
    example = _example()

    assert target_gate._parse_candidate_label("A", example, prompt_mode="choice_alias") == example.candidates[0]["label"]
    assert target_gate._parse_candidate_label("Option D", example, prompt_mode="choice_alias") == example.candidates[3]["label"]


def test_choice_prediction_from_scores_ties_to_prior() -> None:
    example = _example()
    prior_choice = target_gate._prior_choice(example)
    scores = {choice: 0.0 for choice in "ABCD"}

    choice, prediction = target_gate._choice_prediction_from_scores(example, scores)

    assert choice == prior_choice
    assert prediction == target_gate._prior_prediction(example)


def test_choice_prediction_from_scores_selects_best_choice() -> None:
    example = _example()
    scores = {"A": -4.0, "B": -3.0, "C": -0.1, "D": -2.0}

    choice, prediction = target_gate._choice_prediction_from_scores(example, scores)

    assert choice == "C"
    assert prediction == example.candidates[2]["label"]


def test_binary_match_prompt_uses_packet_without_answer_label() -> None:
    example = _example()
    candidate = example.candidates[0]
    prompt = target_gate._prompt_for_binary_match(payload=example.diagnostic_code, candidate=candidate)

    assert f"Source packet: {example.diagnostic_code}" in prompt
    assert candidate["handles_diagnostic"] in prompt
    assert candidate["label"] in prompt
    assert f"answer_label: {example.answer_label}" not in prompt


def test_valid_diag_payload_rejects_non_packet_controls() -> None:
    assert target_gate._valid_diag_payload("A1") is True
    assert target_gate._valid_diag_payload(" Z9 ") is True
    assert target_gate._valid_diag_payload("<NO_SOURCE_PACKET>") is False
    assert target_gate._valid_diag_payload("repair diag is A1") is False
    assert target_gate._valid_diag_payload("A10") is False
    assert target_gate._valid_diag_payload("AA") is False


class _TinyTokenizer:
    def encode(self, surface: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        table = {" yes": [1], "yes": [2], " no": [3], "no": [4], "multi token": [5, 6]}
        return table.get(surface, [])


def test_token_surface_score_uses_best_single_token_surface() -> None:
    logits = torch.full((8,), -10.0)
    logits[1] = 1.0
    logits[2] = 3.0

    score, surface = target_gate._token_surface_score(_TinyTokenizer(), logits, (" yes", "yes", "multi token"))

    assert surface == "yes"
    assert score > -0.2


def test_binary_prediction_requires_positive_yes_margin() -> None:
    example = _example()
    scores = [
        {"candidate_label": candidate["label"], "yes_minus_no": -0.1 * (index + 1)}
        for index, candidate in enumerate(example.candidates)
    ]

    prediction, fallback = target_gate._binary_prediction_from_scores(example, scores)

    assert fallback is True
    assert prediction == target_gate._prior_prediction(example)


def test_binary_prediction_selects_positive_match() -> None:
    example = _example()
    scores = [
        {"candidate_label": candidate["label"], "yes_minus_no": -1.0}
        for candidate in example.candidates
    ]
    scores[2]["yes_minus_no"] = 2.0

    prediction, fallback = target_gate._binary_prediction_from_scores(example, scores)

    assert fallback is False
    assert prediction == example.candidates[2]["label"]


def test_summarize_passes_when_matched_beats_controls() -> None:
    example = _example()
    prior = target_gate._prior_prediction(example)
    rows = []
    for condition in target_gate._conditions():
        prediction = example.answer_label if condition == "matched_packet" else prior
        rows.append(
            {
                "example_id": example.example_id,
                "condition": condition,
                "prediction": prediction,
                "correct": prediction == example.answer_label,
                "valid_prediction": True,
                "payload_bytes": 2 if condition != "target_only" else 0,
                "payload_tokens": 1 if condition != "target_only" else 0,
                "generated_tokens": 1,
                "latency_ms": 1.0,
            }
        )

    summary = target_gate._summarize(rows)

    assert summary["pass_gate"] is True
    assert summary["matched_accuracy"] == 1.0
    assert summary["best_control_accuracy"] == summary["target_only_accuracy"]


def test_summarize_supports_condition_subset_for_resumable_receiver_runs() -> None:
    example = _example()
    rows = []
    for condition in ["target_only", "matched_packet", "shuffled_packet"]:
        prediction = example.answer_label if condition == "matched_packet" else target_gate._prior_prediction(example)
        rows.append(
            {
                "example_id": example.example_id,
                "condition": condition,
                "prediction": prediction,
                "correct": prediction == example.answer_label,
                "valid_prediction": True,
                "payload_bytes": 2 if condition != "target_only" else 0,
                "payload_tokens": 1 if condition != "target_only" else 0,
                "generated_tokens": 1,
                "latency_ms": 1.0,
            }
        )

    summary = target_gate._summarize(rows, conditions=["target_only", "matched_packet", "shuffled_packet"])

    assert summary["conditions"] == ["target_only", "matched_packet", "shuffled_packet"]
    assert summary["exact_id_parity"] is True
    assert summary["best_control_accuracy"] == summary["metrics"]["shuffled_packet"]["accuracy"]


def test_partial_prediction_reader_returns_rows_when_present(tmp_path) -> None:
    path = tmp_path / "target_predictions.partial.jsonl"
    path.write_text('{"example_id":"e0","condition":"target_only"}\n', encoding="utf-8")

    rows = target_gate._read_partial_jsonl(path)

    assert rows == [{"example_id": "e0", "condition": "target_only"}]
    assert target_gate._read_partial_jsonl(tmp_path / "missing.jsonl") == []


def test_validate_conditions_rejects_unknown_condition() -> None:
    try:
        target_gate._validate_conditions(["target_only", "bogus"])
    except ValueError as exc:
        assert "bogus" in str(exc)
    else:
        raise AssertionError("expected ValueError")
