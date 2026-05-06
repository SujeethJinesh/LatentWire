from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]

from experimental.sinkaware.phase4.kernel.approx_sink_attention_triton import (
    approx_sink_attention_scalar_triton_interpret,
    exact_scalar_attention_reference,
    triton_interpreter_readiness,
)


def _enable_repo_local_triton_cpu_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    monkeypatch.setenv("TRITON_CPU_BACKEND", "1")
    monkeypatch.setenv("TRITON_HOME", str(_REPO_ROOT / ".debug/triton_home"))


def test_approx_sink_attention_triton_matches_exact_when_prediction_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("triton")
    _enable_repo_local_triton_cpu_interpreter(monkeypatch)
    generator = torch.Generator().manual_seed(20260505)
    query = torch.randn(8, generator=generator)
    keys = torch.randn(11, 8, generator=generator)
    values = torch.randn(11, generator=generator)
    sink_tokens = 3
    exact_logits = keys @ query / math.sqrt(query.numel())

    expected = exact_scalar_attention_reference(query, keys, values)
    actual = approx_sink_attention_scalar_triton_interpret(
        query,
        keys,
        values,
        exact_logits[:sink_tokens],
        sink_tokens=sink_tokens,
        block_seq=16,
        block_dim=16,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_approx_sink_attention_triton_requires_interpret(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("triton")
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    with pytest.raises(RuntimeError):
        approx_sink_attention_scalar_triton_interpret(
            torch.zeros(4),
            torch.zeros(6, 4),
            torch.zeros(6),
            torch.zeros(2),
            sink_tokens=2,
        )


def test_triton_interpreter_readiness_reports_dependency_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    readiness = triton_interpreter_readiness()

    assert readiness["ready"] is False
    assert readiness["triton_interpret_enabled"] is False
    assert "reason" in readiness

    _enable_repo_local_triton_cpu_interpreter(monkeypatch)
    readiness = triton_interpreter_readiness()
    if readiness["triton_importable"]:
        assert readiness["ready"] is True
        assert readiness["reason"] == "ready for interpreter correctness tests"
    else:
        assert readiness["ready"] is False
        assert readiness["reason"] == "triton is not importable"
