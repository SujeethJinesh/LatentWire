from __future__ import annotations

from scripts import run_source_private_testlog_packet_llm_packet as llm_gate
from scripts import run_source_private_testlog_packet_strict_small as strict_gate


def _loaded_examples(n: int = 8) -> list[llm_gate.LoadedExample]:
    examples = strict_gate.make_benchmark(examples=n, candidates=4, seed=28)
    return [
        llm_gate.LoadedExample(
            example_id=example.example_id,
            answer_label=example.answer_label,
            answer_signature=example.answer_signature,
            private_test_log=example.private_test_log,
            source_prompt=example.source_prompt,
            candidates=tuple(
                {
                    "label": candidate.label,
                    "failure_signature": candidate.failure_signature,
                    "prior_score": candidate.prior_score,
                }
                for candidate in example.candidates
            ),
        )
        for example in examples
    ]


def test_signature_parser_accepts_plain_or_labeled_packets() -> None:
    assert llm_gate._extract_signature("GE") == "GE"
    assert llm_gate._extract_signature("TRACE_SIG=J8") == "J8"
    assert llm_gate._extract_signature("packet: MF") == "MF"
    assert llm_gate._extract_signature("TRACE") == ""
    assert llm_gate._extract_signature("TRACE_SIG=") == ""
    assert llm_gate._extract_signature("candidate_0001_patch_2") == ""


def test_prompt_modes_include_or_omit_helper_line() -> None:
    example = _loaded_examples(1)[0]

    helper_prompt = llm_gate._prompt_for_signature(example, prompt_mode="helper_line")
    full_log_prompt = llm_gate._prompt_for_signature(example, prompt_mode="full_log")

    assert "Private TRACE_SIG line copied from the log:" in helper_prompt
    assert "Private TRACE_SIG line copied from the log:" not in full_log_prompt
    assert "TRACE_SIG=" in full_log_prompt


def test_exact_signature_packet_decodes_to_answer() -> None:
    examples = _loaded_examples(4)
    example = examples[2]

    assert llm_gate._decode_signature_packet(example, example.answer_signature) == example.answer_label
    assert llm_gate._decode_signature_packet(example, "AA") == llm_gate._prior_prediction(example)


def test_model_packet_summary_passes_for_oracle_extractions() -> None:
    examples = _loaded_examples(8)
    packets = [
        {
            "example_id": example.example_id,
            "generated_text": example.answer_signature,
            "packet": example.answer_signature,
            "packet_bytes": 2,
            "packet_tokens": 1,
            "latency_ms": 0.0,
            "valid_packet": True,
        }
        for example in examples
    ]

    _, summary = llm_gate._evaluate(examples, packets, seed=28)

    assert summary["pass_gate"] is True
    assert summary["metrics"]["matched_model_packet"]["accuracy"] == 1.0
    assert summary["metrics"]["target_only"]["accuracy"] < 1.0


def test_key_like_or_label_like_outputs_do_not_create_gain() -> None:
    examples = _loaded_examples(8)
    packets = [
        {
            "example_id": example.example_id,
            "generated_text": example.answer_label,
            "packet": llm_gate._extract_signature(example.answer_label),
            "packet_bytes": 0,
            "packet_tokens": 1,
            "latency_ms": 0.0,
            "valid_packet": False,
        }
        for example in examples
    ]

    _, summary = llm_gate._evaluate(examples, packets, seed=28)

    assert summary["metrics"]["matched_model_packet"]["accuracy"] == summary["metrics"]["target_only"]["accuracy"]


def test_shuffled_model_packet_records_nonself_source_id() -> None:
    examples = _loaded_examples(8)
    packets = [
        {
            "example_id": example.example_id,
            "generated_text": example.answer_signature,
            "packet": example.answer_signature,
            "packet_bytes": 2,
            "packet_tokens": 1,
            "latency_ms": 0.0,
            "valid_packet": True,
        }
        for example in examples
    ]

    rows, _ = llm_gate._evaluate(examples, packets, seed=28)

    for row in rows:
        assert row["conditions"]["shuffled_model_packet"]["source_example_id"] != row["example_id"]
