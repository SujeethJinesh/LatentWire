import pytest

from experimental.thoughtflow_fp8.phase2.hidden_saliency_retention_probe import _paired_delta, _status


def test_hidden_saliency_status_alive_when_phase_and_math_beat_all_proxies() -> None:
    summary = {
        "thoughtflow": {"phase_recall": 0.86, "math_state_recall": 0.78},
        "thoughtflow_saliency_recent": {"phase_recall": 0.88, "math_state_recall": 0.80},
        "attention_received_topk": {"phase_recall": 0.70, "math_state_recall": 0.62},
        "hidden_norm_topk": {"phase_recall": 0.74, "math_state_recall": 0.63},
        "key_norm_topk": {"phase_recall": 0.72, "math_state_recall": 0.65},
        "value_norm_topk": {"phase_recall": 0.73, "math_state_recall": 0.66},
        "kv_norm_topk": {"phase_recall": 0.75, "math_state_recall": 0.67},
        "rkv_like": {"phase_recall": 0.78, "math_state_recall": 0.70},
    }

    assert _status(summary).startswith("ALIVE")


def test_hidden_saliency_status_mixed_on_phase_only_real_saliency_win() -> None:
    summary = {
        "thoughtflow": {"phase_recall": 0.80, "math_state_recall": 0.72},
        "thoughtflow_saliency_recent": {"phase_recall": 0.82, "math_state_recall": 0.72},
        "attention_received_topk": {"phase_recall": 0.70, "math_state_recall": 0.73},
        "hidden_norm_topk": {"phase_recall": 0.71, "math_state_recall": 0.74},
        "key_norm_topk": {"phase_recall": 0.72, "math_state_recall": 0.73},
        "value_norm_topk": {"phase_recall": 0.71, "math_state_recall": 0.73},
        "kv_norm_topk": {"phase_recall": 0.72, "math_state_recall": 0.74},
        "rkv_like": {"phase_recall": 0.65, "math_state_recall": 0.90},
    }

    assert _status(summary).startswith("MIXED")


def test_hidden_saliency_status_weakened_on_hidden_kv_tie() -> None:
    summary = {
        "thoughtflow": {"phase_recall": 0.70, "math_state_recall": 0.70},
        "attention_received_topk": {"phase_recall": 0.70, "math_state_recall": 0.70},
        "hidden_norm_topk": {"phase_recall": 0.71, "math_state_recall": 0.71},
        "key_norm_topk": {"phase_recall": 0.69, "math_state_recall": 0.72},
        "value_norm_topk": {"phase_recall": 0.68, "math_state_recall": 0.71},
        "kv_norm_topk": {"phase_recall": 0.70, "math_state_recall": 0.72},
    }

    assert _status(summary).startswith("WEAKENED")


def test_paired_delta_reports_mean_and_uncertainty() -> None:
    rows = [
        {"trace_id": 0, "policy": "thoughtflow", "phase_recall": 0.8},
        {"trace_id": 0, "policy": "kv_norm_topk", "phase_recall": 0.6},
        {"trace_id": 1, "policy": "thoughtflow", "phase_recall": 0.4},
        {"trace_id": 1, "policy": "kv_norm_topk", "phase_recall": 0.5},
    ]

    result = _paired_delta(rows, "thoughtflow", "kv_norm_topk", "phase_recall")

    assert result["n"] == 2
    assert result["mean"] == pytest.approx(0.05)
    assert result["stderr"] > 0
