from __future__ import annotations

from scripts import run_source_private_evidence_packet_llm_packet as llm_gate
from scripts import run_source_private_evidence_packet_strict_small as strict_gate


def _loaded_examples(n: int = 8) -> list[llm_gate.LoadedExample]:
    examples = strict_gate.make_benchmark(examples=n, candidates=4, seed=28)
    rows = []
    for example in examples:
        rows.append(
            llm_gate.LoadedExample(
                example_id=example.example_id,
                answer_label=example.answer_label,
                private_evidence=example.private_evidence,
                source_prompt=example.source_prompt,
                candidates=tuple(
                    {
                        "label": candidate.label,
                        "display_name": candidate.display_name,
                        "prior_score": candidate.prior_score,
                    }
                    for candidate in example.candidates
                ),
            )
        )
    return rows


def test_digest_packet_decoder_recovers_only_exact_protocol_packet() -> None:
    examples = _loaded_examples(4)
    budget = 2
    salt = f"source-private-packet:28:budget:{budget}"
    key = llm_gate._key_from_evidence(examples[1].private_evidence)
    assert key is not None
    packet = llm_gate._packet_hex(key, salt=salt, budget_bytes=budget)

    assert (
        llm_gate._decode_digest_packet(examples[1], packet, packet_salt=salt, budget_bytes=budget)
        == examples[1].answer_label
    )
    assert (
        llm_gate._decode_digest_packet(examples[1], "0000", packet_salt=salt, budget_bytes=budget)
        == llm_gate._prior_prediction(examples[1])
    )


def test_llm_packet_summary_fails_when_model_outputs_key_prefixes() -> None:
    examples = _loaded_examples(8)
    packets = []
    for example in examples:
        key = llm_gate._key_from_evidence(example.private_evidence)
        assert key is not None
        packets.append(
            {
                "example_id": example.example_id,
                "generated_text": key,
                "packet": key[:4],
                "packet_bytes": 2,
                "packet_tokens": 1,
                "latency_ms": 0.0,
            }
        )

    _, summary = llm_gate._evaluate(examples, packets, budget_bytes=2, seed=28)

    assert summary["pass_gate"] is False
    assert summary["metrics"]["matched_model_packet"]["accuracy"] == summary["metrics"]["target_only"]["accuracy"]


def test_source_final_only_is_detected_as_leak_control() -> None:
    examples = _loaded_examples(8)
    packets = [
        {
            "example_id": example.example_id,
            "generated_text": "",
            "packet": "",
            "packet_bytes": 0,
            "packet_tokens": 0,
            "latency_ms": 0.0,
        }
        for example in examples
    ]
    _, summary = llm_gate._evaluate(examples, packets, budget_bytes=2, seed=28)

    assert summary["metrics"]["source_final_only"]["accuracy"] == 1.0
    assert summary["source_final_minus_best_no_source"] > 0.15
