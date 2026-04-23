from __future__ import annotations

import json
import pathlib

from scripts import analyze_gsm8k_communication_headroom as headroom


def _record(
    example_id: str,
    *,
    method: str,
    correct: bool,
    index: int,
    score_gap: float | None = None,
) -> dict:
    prediction = "answer is 1" if correct else "answer is 0"
    row = {
        "answer": ["1"],
        "correct": correct,
        "example_id": example_id,
        "index": index,
        "method": method,
        "normalized_prediction": "1" if correct else "0",
        "prediction": prediction,
    }
    if score_gap is not None:
        row["selector_trace"] = [{"score_gap": score_gap}]
    return row


def _write_jsonl(path: pathlib.Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def test_headroom_analysis_tracks_source_and_control_overlap(tmp_path: pathlib.Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    source = tmp_path / "source.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    control = tmp_path / "control.jsonl"
    output_json = tmp_path / "headroom.json"
    output_md = tmp_path / "headroom.md"

    target = [
        _record("a", method="target_alone", correct=False, index=0),
        _record("b", method="target_alone", correct=True, index=1),
        _record("c", method="target_alone", correct=False, index=2),
        _record("d", method="target_alone", correct=False, index=3),
    ]
    text = [
        _record("a", method="text_to_text", correct=False, index=0),
        _record("b", method="text_to_text", correct=True, index=1),
        _record("c", method="text_to_text", correct=True, index=2),
        _record("d", method="text_to_text", correct=False, index=3),
    ]
    _write_jsonl(baseline, [*target, *text])
    _write_jsonl(
        source,
        [
            _record("a", method="source_alone", correct=True, index=0),
            _record("b", method="source_alone", correct=False, index=1),
            _record("c", method="source_alone", correct=False, index=2),
            _record("d", method="source_alone", correct=False, index=3),
        ],
    )
    _write_jsonl(
        candidate,
        [
            _record("a", method="rotalign_kv", correct=True, index=0),
            _record("b", method="rotalign_kv", correct=True, index=1),
            _record("c", method="rotalign_kv", correct=False, index=2),
            _record("d", method="rotalign_kv", correct=True, index=3),
        ],
    )
    _write_jsonl(
        control,
        [
            _record("a", method="rotalign_kv", correct=True, index=0),
            _record("b", method="rotalign_kv", correct=False, index=1),
            _record("c", method="rotalign_kv", correct=False, index=2),
            _record("d", method="rotalign_kv", correct=False, index=3),
        ],
    )

    payload = headroom.main(
        [
            "--baseline-predictions",
            str(baseline),
            "--source",
            f"seed0={source}",
            "--candidate",
            f"live={candidate}",
            "--control",
            f"zero={control}",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )

    live = payload["candidates"][0]
    assert live["candidate_win_count_vs_target"] == 2
    assert live["candidate_win_ids_vs_target"] == ["a", "d"]
    assert live["source_overlap"]["seed0"]["candidate_win_source_correct_count"] == 1
    assert live["control_overlap"]["zero"]["candidate_win_retention_count"] == 1
    assert live["candidate_win_provenance"][0]["example_id"] == "a"
    assert live["candidate_win_provenance"][0]["source_correct_labels"] == ["seed0"]
    assert live["candidate_win_provenance"][0]["control_correct_labels"] == ["zero"]
    assert live["oracle"]["target_or_candidate_or_any_source"] == 3
    assert payload["sources"][0]["oracle_target_or_source"] == 2
    assert payload["gate"]["status"] == "control_retention_blocks_positive_claim"
    assert output_json.exists()
    assert "# GSM8K Communication Headroom Diagnostic" in output_md.read_text()


def test_headroom_analysis_reports_source_headroom_without_control_retention(
    tmp_path: pathlib.Path,
) -> None:
    baseline = tmp_path / "baseline.jsonl"
    source = tmp_path / "source.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    control = tmp_path / "control.jsonl"

    target = [
        _record("a", method="target_alone", correct=False, index=0),
        _record("b", method="target_alone", correct=True, index=1),
    ]
    _write_jsonl(
        baseline,
        [
            *target,
            _record("a", method="text_to_text", correct=False, index=0),
            _record("b", method="text_to_text", correct=True, index=1),
        ],
    )
    _write_jsonl(
        source,
        [
            _record("a", method="source_alone", correct=True, index=0),
            _record("b", method="source_alone", correct=False, index=1),
        ],
    )
    _write_jsonl(
        candidate,
        [
            _record("a", method="rotalign_kv", correct=True, index=0),
            _record("b", method="rotalign_kv", correct=True, index=1),
        ],
    )
    _write_jsonl(
        control,
        [
            _record("a", method="rotalign_kv", correct=False, index=0),
            _record("b", method="rotalign_kv", correct=False, index=1),
        ],
    )

    payload = headroom.main(
        [
            "--baseline-predictions",
            str(baseline),
            "--source",
            f"seed0={source}",
            "--candidate",
            f"live={candidate}",
            "--control",
            f"zero={control}",
            "--output-json",
            str(tmp_path / "headroom.json"),
            "--output-md",
            str(tmp_path / "headroom.md"),
        ]
    )

    assert payload["candidates"][0]["gate_status"] == "source_headroom_available_for_method_probe"
    assert payload["gate"]["status"] == "source_headroom_available"


def test_headroom_score_contrast_requires_candidate_to_beat_controls(
    tmp_path: pathlib.Path,
) -> None:
    baseline = tmp_path / "baseline.jsonl"
    source = tmp_path / "source.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    control = tmp_path / "control.jsonl"

    _write_jsonl(
        baseline,
        [
            _record("a", method="target_alone", correct=False, index=0),
            _record("a", method="text_to_text", correct=False, index=0),
        ],
    )
    _write_jsonl(source, [_record("a", method="source_alone", correct=True, index=0)])
    _write_jsonl(
        candidate,
        [_record("a", method="rotalign_kv", correct=True, index=0, score_gap=0.3)],
    )
    _write_jsonl(
        control,
        [_record("a", method="rotalign_kv", correct=False, index=0, score_gap=0.2)],
    )

    payload = headroom.main(
        [
            "--baseline-predictions",
            str(baseline),
            "--source",
            f"seed0={source}",
            "--candidate",
            f"live={candidate}",
            "--control",
            f"zero={control}",
            "--score-field",
            "selector_gap_min",
            "--score-margin",
            "0.05",
            "--output-json",
            str(tmp_path / "headroom.json"),
            "--output-md",
            str(tmp_path / "headroom.md"),
        ]
    )

    contrast = payload["candidates"][0]["score_contrast"]
    assert contrast["passed_candidate_win_count"] == 1
    assert contrast["passed_candidate_win_ids"] == ["a"]
    assert payload["candidates"][0]["candidate_win_provenance"][0]["score_contrast"][
        "candidate_minus_max_control"
    ] == 0.09999999999999998
