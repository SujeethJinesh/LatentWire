import pytest

from experimental.thoughtflow_fp8.phase2.perplexity_impact_proxy import (
    _select_context_ids,
    _status,
    _summary,
    thoughtflow_recent,
)
from experimental.thoughtflow_fp8.phase2.simulate_phase_retention import Token


def test_select_context_ids_preserves_original_order() -> None:
    assert _select_context_ids([10, 11, 12, 13], {3, 0, 2}) == [10, 12, 13]


def test_summary_averages_policy_rows() -> None:
    rows = [
        {"policy": "thoughtflow", "keep_rate": 0.2, "nll": 1.0, "delta_nll_vs_full": 0.1, "ppl": 2.0},
        {"policy": "thoughtflow", "keep_rate": 0.4, "nll": 2.0, "delta_nll_vs_full": 0.2, "ppl": 4.0},
    ]

    summary = _summary(rows)

    assert summary["thoughtflow"]["n_traces"] == 2
    assert summary["thoughtflow"]["keep_rate"] == pytest.approx(0.3)
    assert summary["thoughtflow"]["nll"] == pytest.approx(1.5)
    assert summary["thoughtflow"]["delta_nll_vs_full"] == pytest.approx(0.15)
    assert summary["thoughtflow"]["ppl"] == pytest.approx(3.0)


def test_status_alive_only_when_thoughtflow_beats_compressed_proxies() -> None:
    summary = {
        "full_context": {"nll": 1.0},
        "thoughtflow": {"nll": 1.10},
        "longflow_like": {"nll": 1.15},
        "thin_kv_like": {"nll": 1.20},
    }

    assert _status(summary).startswith("ALIVE")


def test_status_mixed_on_close_tie() -> None:
    summary = {
        "full_context": {"nll": 1.0},
        "thoughtflow": {"nll": 1.13},
        "longflow_like": {"nll": 1.11},
        "thin_kv_like": {"nll": 1.20},
    }

    assert _status(summary).startswith("MIXED")


def test_status_weakened_when_proxy_is_better() -> None:
    summary = {
        "full_context": {"nll": 1.0},
        "thoughtflow": {"nll": 1.20},
        "longflow_like": {"nll": 1.10},
        "thin_kv_like": {"nll": 1.30},
    }

    assert _status(summary).startswith("WEAKENED")


def test_thoughtflow_recent_reserves_recent_tokens() -> None:
    trace = [
        Token("a", "anchor", 1.0),
        Token("b", "anchor", 0.9),
        Token("phase", "phase", 0.8),
        Token("low", "reason", 0.1),
        Token("recent1", "reason", 0.1),
        Token("recent2", "reason", 0.1),
    ]

    kept = thoughtflow_recent(trace, budget=4)

    assert {0, 1}.issubset(kept)
    assert 5 in kept
    assert len(kept) == 4
