from __future__ import annotations

from types import SimpleNamespace

from experimental.hybridkernel.phase2 import profiler_driver


def _args(**overrides: object) -> SimpleNamespace:
    values = {
        "model": "ibm-granite/granite-4.0-h-tiny",
        "endpoint": "http://127.0.0.1:8000",
        "batch_size": 1,
        "prefill_tokens": 8,
        "decode_tokens": 2,
        "requests": 2,
        "seed": 1,
        "timeout_s": 5.0,
        "profile_bracket": False,
        "dry_run": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_dry_run_reports_profile_bracket_endpoints_without_network() -> None:
    result = profiler_driver.run(_args(profile_bracket=True, dry_run=True))

    assert result["profile_bracket"] is True
    assert result["profile_start_endpoint"] == "http://127.0.0.1:8000/start_profile"
    assert result["profile_stop_endpoint"] == "http://127.0.0.1:8000/stop_profile"
    assert [row["status"] for row in result["requests"]] == ["dry_run", "dry_run"]


def test_profile_bracket_wraps_fixed_request_replay(monkeypatch) -> None:
    calls: list[str] = []

    def fake_post_json(endpoint: str, payload: dict[str, object], timeout_s: float) -> None:
        del payload, timeout_s
        calls.append(endpoint)

    monkeypatch.setattr(profiler_driver, "_post_json", fake_post_json)

    result = profiler_driver.run(_args(profile_bracket=True, requests=2))

    assert calls == [
        "http://127.0.0.1:8000/start_profile",
        "http://127.0.0.1:8000/v1/completions",
        "http://127.0.0.1:8000/v1/completions",
        "http://127.0.0.1:8000/stop_profile",
    ]
    assert [row["status"] for row in result["requests"]] == ["ok", "ok"]


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
