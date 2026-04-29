from __future__ import annotations

from scripts.run_source_private_mac_endpoint_proxy_frontier import summarize


def _row(example_id: str, condition: str, *, correct: bool, payload_bytes: int, prompt_tokens: int) -> dict:
    return {
        "example_id": example_id,
        "condition": condition,
        "correct": correct,
        "valid_prediction": True,
        "payload_bytes": payload_bytes,
        "payload_tokens_proxy": 1 if payload_bytes else 0,
        "prompt_bytes": prompt_tokens * 4,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": 4,
        "ttft_ms": float(prompt_tokens),
        "e2e_ms": float(prompt_tokens + 10),
    }


def test_summarize_endpoint_proxy_frontier_passes_packet_rate_case() -> None:
    conditions = [
        "target_only",
        "matched_packet",
        "matched_byte_text_2",
        "query_aware_diag_span",
        "structured_json_diag",
        "structured_free_text_diag",
        "full_hidden_log",
    ]
    rows = []
    for idx in range(4):
        example_id = f"ex{idx}"
        rows.extend(
            [
                _row(example_id, "target_only", correct=idx == 0, payload_bytes=0, prompt_tokens=100),
                _row(example_id, "matched_packet", correct=True, payload_bytes=2, prompt_tokens=101),
                _row(example_id, "matched_byte_text_2", correct=idx == 0, payload_bytes=2, prompt_tokens=101),
                _row(example_id, "query_aware_diag_span", correct=True, payload_bytes=14, prompt_tokens=105),
                _row(example_id, "structured_json_diag", correct=True, payload_bytes=21, prompt_tokens=107),
                _row(example_id, "structured_free_text_diag", correct=True, payload_bytes=17, prompt_tokens=106),
                _row(example_id, "full_hidden_log", correct=True, payload_bytes=360, prompt_tokens=190),
            ]
        )

    summary = summarize(rows, conditions=conditions)

    assert summary["pass_gate"] is True
    assert summary["packet_minus_target_accuracy"] == 0.75
    assert summary["packet_vs_query_payload_compression"] == 7.0
    assert summary["full_log_prompt_token_delta_vs_packet"] == 89
