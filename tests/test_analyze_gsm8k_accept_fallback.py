from __future__ import annotations

import json
import pathlib

from scripts import analyze_gsm8k_accept_fallback as replay


def _record(
    example_id: str,
    *,
    method: str,
    prediction: str,
    normalized_prediction: str | None,
    correct: bool,
    score_gap: float,
    index: int,
) -> dict:
    return {
        "answer": ["1"],
        "correct": correct,
        "example_id": example_id,
        "index": index,
        "method": method,
        "normalized_prediction": normalized_prediction,
        "prediction": prediction,
        "selector_trace": [{"score_gap": score_gap}],
    }


def _write_jsonl(path: pathlib.Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def test_selector_gap_score_reads_trace_stat() -> None:
    row = {
        "selector_trace": [
            {"score_gap": 0.3},
            {"score_gap": 0.1},
            {"score_gap": 0.2},
        ]
    }

    assert replay._score(row, "selector_gap_min") == 0.1
    assert replay._score(row, "selector_gap_max") == 0.3
    assert abs(replay._score(row, "selector_gap_avg") - 0.2) < 1e-12


def test_accept_fallback_replay_gate_passes_with_control_safe_gap_policy(
    tmp_path: pathlib.Path,
) -> None:
    baseline = tmp_path / "baseline.jsonl"
    seed0 = tmp_path / "seed0.jsonl"
    seed3 = tmp_path / "seed3.jsonl"
    control = tmp_path / "control.jsonl"
    output_json = tmp_path / "replay.json"
    output_md = tmp_path / "replay.md"

    target_records = [
        _record("a", method="target_alone", prediction="answer is 0", normalized_prediction="0", correct=False, score_gap=0.0, index=0),
        _record("b", method="target_alone", prediction="answer is 1", normalized_prediction="1", correct=True, score_gap=0.0, index=1),
        _record("c", method="target_alone", prediction="answer is 0", normalized_prediction="0", correct=False, score_gap=0.0, index=2),
    ]
    _write_jsonl(baseline, target_records)

    _write_jsonl(
        seed0,
        [
            _record("a", method="rotalign_kv_gate_0.10", prediction="answer is 1", normalized_prediction="1", correct=True, score_gap=0.8, index=0),
            _record("b", method="rotalign_kv_gate_0.10", prediction="answer is 2", normalized_prediction="2", correct=False, score_gap=0.1, index=1),
            _record("c", method="rotalign_kv_gate_0.10", prediction="answer is 0", normalized_prediction="0", correct=False, score_gap=0.9, index=2),
        ],
    )
    _write_jsonl(
        seed3,
        [
            _record("a", method="rotalign_kv_gate_0.10", prediction="answer is 1", normalized_prediction="1", correct=True, score_gap=0.8, index=0),
            _record("b", method="rotalign_kv_gate_0.10", prediction="answer is 2", normalized_prediction="2", correct=False, score_gap=0.1, index=1),
            _record("c", method="rotalign_kv_gate_0.10", prediction="answer is 0", normalized_prediction="0", correct=False, score_gap=0.9, index=2),
        ],
    )
    _write_jsonl(
        control,
        [
            _record("a", method="rotalign_kv", prediction="777777777777777777", normalized_prediction="777777777777777777", correct=False, score_gap=0.9, index=0),
            _record("b", method="rotalign_kv", prediction="no numeric output", normalized_prediction=None, correct=False, score_gap=0.9, index=1),
            _record("c", method="rotalign_kv", prediction="no numeric output", normalized_prediction=None, correct=False, score_gap=0.9, index=2),
        ],
    )

    payload = replay.main(
        [
            "--baseline-predictions",
            str(baseline),
            "--candidate",
            f"seed0={seed0}",
            "--candidate",
            f"seed3={seed3}",
            "--control",
            f"zero_source={control}",
            "--score-field",
            "selector_gap_min",
            "--score-quantile",
            "0.5",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )

    gate = next(
        item
        for item in payload["gates"]
        if item["policy"] == "selector_gap_min_ge_q0p5_numeric_changed"
    )
    assert gate["status"] == "accept_fallback_clears_offline_gate"
    seed0_summary = next(
        item
        for item in payload["candidates"]
        if item["label"] == "seed0"
        and item["policy"] == "selector_gap_min_ge_q0p5_numeric_changed"
    )
    control_summary = next(
        item
        for item in payload["controls"]
        if item["policy"] == "selector_gap_min_ge_q0p5_numeric_changed"
    )
    assert seed0_summary["correct"] == 2
    assert seed0_summary["paired_vs_target"] == {"win": 1, "loss": 0, "tie": 2}
    assert control_summary["accepted_count"] == 0
    assert output_json.exists()
    assert "# GSM8K Accept/Fallback Replay" in output_md.read_text()
