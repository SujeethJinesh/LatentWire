from __future__ import annotations

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

    assert matched == examples[0].diagnostic_code
    assert len(json_payload.encode("utf-8")) == 2
    assert len(text_payload.encode("utf-8")) == 2
    assert json_payload != examples[0].diagnostic_code
    assert text_payload != examples[0].diagnostic_code


def test_parse_candidate_label_accepts_exact_or_embedded_label() -> None:
    example = _example()
    label = example.candidates[0]["label"]

    assert target_gate._parse_candidate_label(label, example) == label
    assert target_gate._parse_candidate_label(f"{label}\n", example) == label
    assert target_gate._parse_candidate_label(f"The answer is {label}.", example) == label
    assert target_gate._parse_candidate_label("candidate_9999_patch_0_repair_record", example) == ""


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


def test_validate_conditions_rejects_unknown_condition() -> None:
    try:
        target_gate._validate_conditions(["target_only", "bogus"])
    except ValueError as exc:
        assert "bogus" in str(exc)
    else:
        raise AssertionError("expected ValueError")
