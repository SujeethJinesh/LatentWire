from __future__ import annotations

from scripts import run_source_private_hidden_repair_packet_llm as llm_gate
from scripts import run_source_private_hidden_repair_packet_smoke as repair_gate


def _loaded_examples(n: int = 8) -> list[llm_gate.LoadedExample]:
    examples = repair_gate.make_benchmark(examples=n, candidates=4, seed=28)
    return [
        llm_gate.LoadedExample(
            example_id=example.example_id,
            answer_label=example.answer_label,
            diagnostic_code=example.diagnostic_code,
            private_test_log=example.private_test_log,
            candidates=tuple(
                {
                    "label": candidate.label,
                    "handles_diagnostic": candidate.handles_diagnostic,
                    "prior_score": candidate.prior_score,
                }
                for candidate in example.candidates
            ),
        )
        for example in examples
    ]


def test_diag_parser_accepts_plain_or_labeled_packet() -> None:
    assert llm_gate._extract_diag("F0") == "F0"
    assert llm_gate._extract_diag("REPAIR_DIAG=G1") == "G1"
    assert llm_gate._extract_diag("Packet: H2") == "H2"
    assert llm_gate._extract_diag("REPAIR_DIAG=") == ""


def test_exact_diag_decodes_to_answer() -> None:
    example = _loaded_examples(4)[2]

    assert llm_gate._decode_packet(example, example.diagnostic_code) == example.answer_label
    assert llm_gate._decode_packet(example, "Z9") == llm_gate._prior_prediction(example)


def test_prompt_modes_remove_helper_scaffolding_incrementally() -> None:
    example = _loaded_examples(1)[0]

    copied = llm_gate._prompt_for_diag(example, prompt_mode="copied_helper")
    log_only = llm_gate._prompt_for_diag(example, prompt_mode="log_only")
    trace_no_hint = llm_gate._prompt_for_diag(example, prompt_mode="trace_no_hint")
    raw_log = llm_gate._prompt_for_diag(example, prompt_mode="raw_log_no_trace")

    assert "Private REPAIR_DIAG line copied from the log" in copied
    assert "Private REPAIR_DIAG line copied from the log" not in log_only
    assert "hint: emit only" in log_only
    assert "hint: emit only" not in trace_no_hint
    assert "private_tool_trace: REPAIR_DIAG=" in trace_no_hint
    assert "private_tool_trace: REPAIR_DIAG=" not in raw_log


def test_oracle_model_packets_pass_and_controls_fail() -> None:
    examples = _loaded_examples(8)
    packets = [
        {
            "example_id": example.example_id,
            "generated_text": example.diagnostic_code,
            "packet": example.diagnostic_code,
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


def test_shuffled_model_packet_records_nonself_source_id() -> None:
    examples = _loaded_examples(8)
    packets = [
        {
            "example_id": example.example_id,
            "generated_text": example.diagnostic_code,
            "packet": example.diagnostic_code,
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
