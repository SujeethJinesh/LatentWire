from experimental.thoughtflow_fp8.phase2.run_real_trace_retention import _label_trace, thoughtflow


def test_real_trace_labeling_has_phase_and_math_state() -> None:
    trace = _label_trace("First identify 7 bags. Then subtract 12 - 7 = 5. Answer: 5")
    labels = {token.label for token in trace}

    assert "anchor" in labels
    assert "phase" in labels
    assert "math_state" in labels


def test_thoughtflow_keeps_phase_tokens_at_small_budget() -> None:
    trace = _label_trace("First identify 7 bags. Then subtract 12 - 7 = 5. Answer: 5")
    kept = thoughtflow(trace, budget=6)

    kept_labels = {trace[idx].label for idx in kept}
    assert "anchor" in kept_labels
    assert "phase" in kept_labels
