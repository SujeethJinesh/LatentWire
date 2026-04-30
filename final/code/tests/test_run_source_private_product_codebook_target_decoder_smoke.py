from __future__ import annotations

from scripts import run_source_private_product_codebook_target_decoder_smoke as gate


def _state() -> gate.ProductCodebookReceiverState:
    return gate.build_receiver_state(
        train_examples=64,
        eval_examples=4,
        train_seed=5,
        eval_seed=6,
        train_family_set="all",
        eval_family_set="all",
        candidates=4,
        feature_dim=64,
        budget_bytes=2,
        ridge=1e-2,
        candidate_view="slot",
        fit_intercept=False,
        remap_slot_seed=101,
    )


def test_prompt_exposes_public_pq_signatures_without_answer_field() -> None:
    state = _state()
    example = state.eval_rows[0]
    payload, _ = gate._condition_payload(
        condition="matched_product_codebook",
        example=example,
        state=state,
        index=0,
        rng=gate.random.Random(7),
    )

    prompt = gate._prompt_for_product_codebook_decoder(
        example,
        payload=payload,
        state=state,
        candidate_metadata_mode="signature",
    )

    assert "pq_signature=[" in prompt
    assert f"hex={payload.hex()}" in prompt
    assert "highest prior_score" in prompt
    assert "Target-prior choice:" not in prompt
    assert "answer_label:" not in prompt
    assert "handles_repair_diag" not in prompt
    for candidate in example.candidates:
        assert candidate.label not in prompt


def test_condition_payloads_are_rate_capped_and_source_destroying() -> None:
    state = _state()
    example = state.eval_rows[0]
    rng = gate.random.Random(7)

    matched, _ = gate._condition_payload(
        condition="matched_product_codebook",
        example=example,
        state=state,
        index=0,
        rng=rng,
    )
    random_payload, _ = gate._condition_payload(
        condition="random_same_byte",
        example=example,
        state=state,
        index=0,
        rng=gate.random.Random(7),
    )
    text_payload, _ = gate._condition_payload(
        condition="structured_free_text_same_byte",
        example=example,
        state=state,
        index=0,
        rng=gate.random.Random(7),
    )
    target_payload, _ = gate._condition_payload(
        condition="target_derived_sidecar",
        example=example,
        state=state,
        index=0,
        rng=gate.random.Random(7),
    )
    wrong_codebook_payload, _ = gate._condition_payload(
        condition="wrong_codebook_packet",
        example=example,
        state=state,
        index=0,
        rng=gate.random.Random(7),
    )

    assert matched is not None
    assert len(matched) == state.budget_bytes
    assert len(random_payload or b"") == state.budget_bytes
    assert len(text_payload or b"") == state.budget_bytes
    assert len(target_payload or b"") == state.budget_bytes
    assert len(wrong_codebook_payload or b"") == state.budget_bytes
    assert text_payload != matched
    assert target_payload != matched


def test_distance_prompt_adds_packet_distances_only_when_requested() -> None:
    state = _state()
    example = state.eval_rows[0]
    payload, _ = gate._condition_payload(
        condition="matched_product_codebook",
        example=example,
        state=state,
        index=0,
        rng=gate.random.Random(7),
    )

    signature_prompt = gate._prompt_for_product_codebook_decoder(
        example,
        payload=payload,
        state=state,
        candidate_metadata_mode="signature",
    )
    distance_prompt = gate._prompt_for_product_codebook_decoder(
        example,
        payload=payload,
        state=state,
        candidate_metadata_mode="distance",
    )

    assert "distance_to_packet" not in signature_prompt
    assert "distance_to_packet=" in distance_prompt
    assert example.example_id not in distance_prompt
    assert example.family_name not in distance_prompt
    assert "candidate_" not in distance_prompt
    assert "answer_label" not in distance_prompt
    for candidate in example.candidates:
        assert candidate.label not in distance_prompt


def test_no_packet_distance_prompt_uses_no_packet_distances() -> None:
    state = _state()
    example = state.eval_rows[0]

    prompt = gate._prompt_for_product_codebook_decoder(
        example,
        payload=None,
        state=state,
        candidate_metadata_mode="distance",
    )

    assert prompt.count("distance_to_packet=<NO_PACKET>") == len(example.candidates)
    assert "distance_to_packet=0." not in prompt


def test_binary_match_prompt_uses_blinded_choices_only() -> None:
    state = _state()
    example = state.eval_rows[0]
    payload, _ = gate._condition_payload(
        condition="matched_product_codebook",
        example=example,
        state=state,
        index=0,
        rng=gate.random.Random(7),
    )

    prompt = gate._prompt_for_product_codebook_binary_match(
        example,
        payload=payload,
        state=state,
        candidate_metadata_mode="signature",
        focus_choice=gate._choice_labels(example)[0],
    )

    assert "Focus candidate:" in prompt
    assert "Return only yes or no" in prompt
    assert example.example_id not in prompt
    assert example.family_name not in prompt
    assert "candidate_" not in prompt
    for candidate in example.candidates:
        assert candidate.label not in prompt


def test_parse_candidate_choice_accepts_exact_or_embedded_choice() -> None:
    state = _state()
    example = state.eval_rows[0]
    choice = gate._choice_labels(example)[0]
    label = gate._candidate_label_for_choice(example, choice)

    assert label in {candidate.label for candidate in example.candidates}
    assert gate._parse_candidate_choice(choice, example) == choice
    assert gate._parse_candidate_choice(f"{choice}\n", example) == choice
    assert gate._parse_candidate_choice(f"The answer is {choice}.", example) == choice
    assert gate._parse_candidate_choice("candidate_9999_patch_0_repair_record", example) == ""


def test_choice_logprob_prediction_maps_display_choice_to_candidate_label() -> None:
    state = _state()
    example = state.eval_rows[0]
    answer_choice = gate._choice_for_candidate_label(example, example.answer_label)
    choice_scores = {choice: -10.0 for choice in gate._choice_labels(example)}
    choice_scores[answer_choice] = 0.0

    display_prediction, prediction = gate._choice_prediction_from_scores(example, choice_scores)

    assert display_prediction == answer_choice
    assert prediction == example.answer_label


def test_choice_logprob_prediction_tie_falls_back_to_target_prior() -> None:
    state = _state()
    example = state.eval_rows[0]
    prior_choice = gate._choice_for_candidate_label(example, gate._prior_prediction(example))
    choice_scores = {choice: 0.0 for choice in gate._choice_labels(example)}

    display_prediction, prediction = gate._choice_prediction_from_scores(example, choice_scores)

    assert display_prediction == prior_choice
    assert prediction == gate._prior_prediction(example)


def test_choice_token_scores_use_best_single_token_surface() -> None:
    import torch

    class FakeTokenizer:
        vocab = {"A": 0, " A": 1, "B": 2, " B": 3}

        def encode(self, surface: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return [self.vocab[surface]]

    logits = torch.zeros(4)
    logits[1] = 5.0
    logits[2] = 6.0

    scores, surfaces = gate._choice_token_scores(FakeTokenizer(), logits, ["A", "B"])

    assert surfaces["A"] == " A"
    assert surfaces["B"] == "B"
    assert scores["B"] > scores["A"]


def test_binary_prediction_falls_back_when_all_yes_scores_below_threshold() -> None:
    state = _state()
    example = state.eval_rows[0]
    scores = [
        {"display_choice": choice, "yes_minus_no": -0.1}
        for choice in gate._choice_labels(example)
    ]

    display_prediction, prediction, fallback = gate._binary_prediction_from_scores(
        example,
        scores,
        threshold=0.0,
    )

    assert display_prediction == gate._choice_for_candidate_label(example, gate._prior_prediction(example))
    assert prediction == gate._prior_prediction(example)
    assert fallback is True


def test_summarize_rejects_duplicate_condition_ids_as_parity_failure() -> None:
    state = _state()
    examples = state.eval_rows[:2]
    rows = []
    for condition in ["target_only", "matched_product_codebook"]:
        condition_examples = [examples[0], examples[0]] if condition == "matched_product_codebook" else examples
        for example in condition_examples:
            rows.append(
                {
                    "example_id": example.example_id,
                    "condition": condition,
                    "prediction": gate._prior_prediction(example),
                    "correct": False,
                    "valid_prediction": True,
                    "payload_bytes": 0,
                    "payload_tokens": 0,
                    "generated_tokens": 1,
                    "latency_ms": 1.0,
                }
            )

    summary = gate._summarize(rows, conditions=["target_only", "matched_product_codebook"])

    assert summary["exact_id_parity"] is False
    assert summary["condition_id_parity"]["matched_product_codebook"]["duplicate_ids"] == [examples[0].example_id]


def test_summarize_passes_when_matched_beats_clean_controls() -> None:
    state = _state()
    example = next(row for row in state.eval_rows if gate._prior_prediction(row) != row.answer_label)
    prior = gate._prior_prediction(example)
    rows = []
    for condition in gate._conditions():
        prediction = example.answer_label if condition == "matched_product_codebook" else prior
        rows.append(
            {
                "example_id": example.example_id,
                "condition": condition,
                "prediction": prediction,
                "correct": prediction == example.answer_label,
                "valid_prediction": True,
                "payload_bytes": state.budget_bytes if condition != "target_only" else 0,
                "payload_tokens": 1 if condition != "target_only" else 0,
                "generated_tokens": 1,
                "latency_ms": 1.0,
            }
        )

    summary = gate._summarize(rows)

    assert summary["pass_gate"] is True
    assert summary["matched_accuracy"] == 1.0
    assert summary["best_control_accuracy"] == summary["target_only_accuracy"]


def test_validate_conditions_rejects_unknown_condition() -> None:
    try:
        gate._validate_conditions(["target_only", "bogus"])
    except ValueError as exc:
        assert "bogus" in str(exc)
    else:
        raise AssertionError("expected ValueError")
