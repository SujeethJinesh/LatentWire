from __future__ import annotations

import json
import re

from scripts import run_source_private_hidden_repair_packet_endpoint as endpoint_gate
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


def test_chat_completion_url_accepts_base_or_full_path() -> None:
    assert endpoint_gate._chat_completions_url("http://host:8000") == "http://host:8000/v1/chat/completions"
    assert endpoint_gate._chat_completions_url("http://host:8000/v1") == "http://host:8000/v1/chat/completions"
    assert (
        endpoint_gate._chat_completions_url("http://host:8000/v1/chat/completions")
        == "http://host:8000/v1/chat/completions"
    )


def test_completion_parsing_handles_openai_and_text_shapes() -> None:
    chat_payload = {"choices": [{"message": {"content": "G0"}}], "usage": {"completion_tokens": 3}}
    text_payload = {"choices": [{"text": "H1"}]}

    assert endpoint_gate._completion_text(chat_payload) == "G0"
    assert endpoint_gate._completion_tokens(chat_payload, "G0") == 3
    assert endpoint_gate._completion_text(text_payload) == "H1"
    assert endpoint_gate._completion_tokens(text_payload, "H1") == 2


def test_endpoint_packet_generation_reuses_standard_evaluator(monkeypatch) -> None:
    examples = _loaded_examples(8)

    def fake_completion(**kwargs: object) -> tuple[str, int, dict[str, object], dict[str, object]]:
        prompt = str(kwargs["prompt"])
        match = re.search(r"REPAIR_DIAG=([A-Z][0-9])", prompt)
        generated = match.group(1) if match else ""
        return (
            generated,
            1,
            {"choices": [{"message": {"content": generated}}], "usage": {"completion_tokens": 1}},
            {"model": kwargs["model"], "messages": [{"role": "user", "content": prompt}]},
        )

    monkeypatch.setattr(endpoint_gate, "_post_chat_completion", fake_completion)

    packets, trace = endpoint_gate._generate_endpoint_packets(
        examples,
        api_base="http://127.0.0.1:8000/v1",
        api_key=None,
        model="served-model",
        max_tokens=8,
        prompt_mode="trace_no_hint",
        timeout_s=1.0,
        seed=29,
        request_interval_s=0.0,
    )
    _, summary = llm_gate._evaluate(examples, packets, seed=29)

    assert len(trace) == len(examples)
    assert summary["pass_gate"] is True
    assert summary["metrics"]["matched_model_packet"]["accuracy"] == 1.0
    assert summary["metrics"]["target_only"]["accuracy"] == 0.25


def test_endpoint_trace_is_json_serializable(monkeypatch) -> None:
    examples = _loaded_examples(1)

    def fake_completion(**kwargs: object) -> tuple[str, int, dict[str, object], dict[str, object]]:
        return (
            examples[0].diagnostic_code,
            1,
            {"id": "cmpl-test", "choices": [{"message": {"content": examples[0].diagnostic_code}}]},
            {"model": kwargs["model"], "messages": [{"role": "user", "content": kwargs["prompt"]}]},
        )

    monkeypatch.setattr(endpoint_gate, "_post_chat_completion", fake_completion)
    _, trace = endpoint_gate._generate_endpoint_packets(
        examples,
        api_base="http://127.0.0.1:8000/v1",
        api_key=None,
        model="served-model",
        max_tokens=8,
        prompt_mode="trace_no_hint",
        timeout_s=1.0,
        seed=29,
        request_interval_s=0.0,
    )

    serialized = json.loads(json.dumps(trace))
    assert serialized[0]["raw_response"]["id"] == "cmpl-test"
    assert serialized[0]["request_body"]["model"] == "served-model"
