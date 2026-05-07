from __future__ import annotations

from types import SimpleNamespace

import pytest

from experimental.hybridkernel.phase2 import profiler_driver


def _args(**overrides: object) -> SimpleNamespace:
    values = {
        "model": "ibm-granite/granite-4.0-h-tiny",
        "run_id": "granite_primary_r1",
        "endpoint": "http://127.0.0.1:8000",
        "batch_size": 1,
        "prefill_tokens": 8,
        "decode_tokens": 2,
        "requests": 2,
        "seed": 1,
        "tokenizer": None,
        "require_token_counts": False,
        "enforce_prefill_token_counts": False,
        "timeout_s": 5.0,
        "profile_bracket": False,
        "dry_run": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_dry_run_reports_profile_bracket_endpoints_without_network() -> None:
    result = profiler_driver.run(_args(profile_bracket=True, dry_run=True))

    assert result["run_id"] == "granite_primary_r1"
    assert result["profile_bracket"] is True
    assert result["profile_start_endpoint"] == "http://127.0.0.1:8000/start_profile"
    assert result["profile_stop_endpoint"] == "http://127.0.0.1:8000/stop_profile"
    assert [row["status"] for row in result["requests"]] == ["dry_run", "dry_run"]


def test_profile_bracket_wraps_fixed_request_replay(monkeypatch) -> None:
    calls: list[str] = []
    payloads: list[dict[str, object]] = []

    def fake_post_json(endpoint: str, payload: dict[str, object], timeout_s: float) -> None:
        del timeout_s
        calls.append(endpoint)
        payloads.append(payload)

    monkeypatch.setattr(profiler_driver, "_post_json", fake_post_json)

    result = profiler_driver.run(_args(profile_bracket=True, requests=2))

    assert calls == [
        "http://127.0.0.1:8000/start_profile",
        "http://127.0.0.1:8000/v1/completions",
        "http://127.0.0.1:8000/v1/completions",
        "http://127.0.0.1:8000/stop_profile",
    ]
    assert [row["status"] for row in result["requests"]] == ["ok", "ok"]
    completion_payloads = [payload for payload in payloads if payload.get("model")]
    assert all(payload["min_tokens"] == 2 for payload in completion_payloads)
    assert all(payload["ignore_eos"] is True for payload in completion_payloads)


def test_profile_bracket_stops_after_request_error(monkeypatch) -> None:
    calls: list[str] = []

    def fake_post_json(endpoint: str, payload: dict[str, object], timeout_s: float) -> None:
        del payload, timeout_s
        calls.append(endpoint)
        if endpoint.endswith("/v1/completions"):
            raise RuntimeError("synthetic request failure")

    monkeypatch.setattr(profiler_driver, "_post_json", fake_post_json)

    result = profiler_driver.run(_args(profile_bracket=True, requests=1))

    assert calls == [
        "http://127.0.0.1:8000/start_profile",
        "http://127.0.0.1:8000/v1/completions",
        "http://127.0.0.1:8000/stop_profile",
    ]
    assert result["requests"][0]["status"] == "error:synthetic request failure"


def test_dry_run_can_log_exact_prompt_token_counts(monkeypatch) -> None:
    class FakeTokenizer:
        def encode(self, prompt: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return prompt.split()

    monkeypatch.setattr(
        profiler_driver,
        "_load_tokenizer",
        lambda model, tokenizer_name, require: (FakeTokenizer(), "fake-tokenizer", "test"),
    )

    result = profiler_driver.run(
        _args(batch_size=2, prefill_tokens=8, tokenizer="fake-tokenizer", require_token_counts=True, dry_run=True)
    )

    assert result["token_counts_required"] is True
    assert result["tokenizer"] == "fake-tokenizer"
    assert result["requests"][0]["prompt_token_counts"] == [8, 8]
    assert result["requests"][0]["prompt_token_count_total"] == 16
    assert result["requests"][0]["requested_decode_tokens"] == 2
    assert result["requests"][0]["expected_completion_tokens_total"] == 4
    assert len(result["requests"][0]["prompt_sha256"]) == 2
    assert str(result["requests"][0]["payload_sha256"]).startswith("sha256:")


def test_required_token_counts_use_decode_roundtrip_prompt_synthesis(monkeypatch) -> None:
    class FakeTokenizer:
        vocab_size = 16
        all_special_ids: list[int] = []

        def decode(
            self,
            token_ids: list[int],
            *,
            skip_special_tokens: bool,
            clean_up_tokenization_spaces: bool,
        ) -> str:
            assert skip_special_tokens is True
            assert clean_up_tokenization_spaces is False
            return " ".join(f"safe{token_id}" for token_id in token_ids)

        def encode(self, prompt: str, *, add_special_tokens: bool) -> list[str]:
            assert add_special_tokens is False
            tokens = prompt.split()
            if all(token.startswith("safe") for token in tokens):
                return tokens
            return tokens + ["approximate_prompt_would_mismatch"]

    monkeypatch.setattr(
        profiler_driver,
        "_load_tokenizer",
        lambda model, tokenizer_name, require: (FakeTokenizer(), "fake-tokenizer", "test"),
    )

    planned = profiler_driver._planned_requests(
        _args(batch_size=2, prefill_tokens=8, tokenizer="fake-tokenizer", require_token_counts=True),
        FakeTokenizer(),
    )

    _request_id, prompt_payload, prompt_token_counts, _payload = planned[0]
    assert isinstance(prompt_payload, list)
    assert all(prompt.startswith("safe") for prompt in prompt_payload)
    assert prompt_token_counts == [8, 8]


def test_profile_run_fails_before_start_when_prefill_counts_mismatch(monkeypatch) -> None:
    calls: list[str] = []

    class FakeTokenizer:
        def encode(self, prompt: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return prompt.split() + ["extra"]

    def fake_post_json(endpoint: str, payload: dict[str, object], timeout_s: float) -> None:
        del payload, timeout_s
        calls.append(endpoint)

    monkeypatch.setattr(
        profiler_driver,
        "_load_tokenizer",
        lambda model, tokenizer_name, require: (FakeTokenizer(), "fake-tokenizer", "test"),
    )
    monkeypatch.setattr(profiler_driver, "_post_json", fake_post_json)

    with pytest.raises(ValueError, match="prompt token counts do not match"):
        profiler_driver.run(
            _args(
                profile_bracket=True,
                tokenizer="fake-tokenizer",
                require_token_counts=True,
                prefill_tokens=8,
                requests=1,
            )
        )

    assert calls == []
