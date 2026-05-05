from experimental.thoughtflow_fp8.phase2.hidden_saliency_retention_probe import _status


def test_hidden_saliency_status_alive_when_phase_beats_saliency() -> None:
    summary = {
        "thoughtflow": {"phase_recall": 0.8},
        "attention_received_topk": {"phase_recall": 0.7},
    }

    assert _status(summary).startswith("ALIVE")


def test_hidden_saliency_status_weakened_on_tie() -> None:
    summary = {
        "thoughtflow": {"phase_recall": 0.7},
        "attention_received_topk": {"phase_recall": 0.7},
    }

    assert _status(summary).startswith("WEAKENED")
