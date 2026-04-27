from __future__ import annotations

from scripts import analyze_answer_likelihood_controls as analyzer


def _row(example_id: str, score: float, correct: bool = False) -> dict:
    return {
        "example_id": example_id,
        "answer_mean_logprob": score,
        "correct": correct,
    }


def test_answer_likelihood_controls_pass_when_live_beats_controls() -> None:
    payload = analyzer.analyze(
        live_records=[_row("a", -1.0), _row("b", -1.0), _row("c", -1.0), _row("d", -1.0)],
        controls={
            "zero": [_row("a", -1.2), _row("b", -1.3), _row("c", -1.4), _row("d", -1.1)],
            "shuffle": [_row("a", -1.1), _row("b", -1.1), _row("c", -1.2), _row("d", -1.2)],
        },
        unavailable_controls=[],
        score_field="answer_mean_logprob",
        min_mean_delta=0.05,
        min_best_control_wins=3,
    )

    assert payload["gate"]["status"] == "answer_likelihood_controls_pass"
    assert payload["best_control"]["wins"] == 4


def test_answer_likelihood_controls_fail_when_control_matches_live() -> None:
    payload = analyzer.analyze(
        live_records=[_row("a", -1.0), _row("b", -1.0), _row("c", -1.0), _row("d", -1.0)],
        controls={
            "slots": [_row("a", -1.0), _row("b", -1.0), _row("c", -1.0), _row("d", -1.0)],
        },
        unavailable_controls=["target_only"],
        score_field="answer_mean_logprob",
        min_mean_delta=0.05,
        min_best_control_wins=3,
    )

    assert payload["gate"]["status"] == "answer_likelihood_controls_fail"
    checks = {check["name"]: check for check in payload["gate"]["checks"]}
    assert checks["mean_delta_vs_each_control"]["pass"] is False
    assert checks["per_example_best_control_wins"]["pass"] is False
    assert payload["gate"]["unavailable_controls"] == ["target_only"]
