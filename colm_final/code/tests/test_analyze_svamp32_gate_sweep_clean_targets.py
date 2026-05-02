from __future__ import annotations

import json
import pathlib

from scripts import analyze_svamp32_gate_sweep_clean_targets as sweep


def _row(example_id: str, *, method: str, correct: bool) -> dict:
    return {
        "example_id": example_id,
        "method": method,
        "correct": correct,
        "normalized_prediction": "1" if correct else "0",
        "prediction": "answer is 1" if correct else "answer is 0",
    }


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _target_rows() -> list[dict]:
    return [
        _row("a", method="target_alone", correct=False),
        _row("b", method="target_alone", correct=False),
        _row("c", method="target_alone", correct=True),
        _row("d", method="target_alone", correct=False),
    ]


def _teacher_rows() -> list[dict]:
    return [
        _row("a", method="c2c_generate", correct=True),
        _row("b", method="c2c_generate", correct=True),
        _row("c", method="c2c_generate", correct=True),
        _row("d", method="c2c_generate", correct=False),
    ]


def _target_set() -> dict:
    return {
        "ids": {
            "teacher_only": ["a", "b"],
            "clean_residual_targets": ["a", "b"],
        }
    }


def test_sweep_ranks_raw_gate_methods_by_clean_recovery() -> None:
    candidates = [
        _row("a", method="rotalign_kv_gate_0.10", correct=True),
        _row("b", method="rotalign_kv_gate_0.10", correct=False),
        _row("c", method="rotalign_kv_gate_0.10", correct=True),
        _row("d", method="rotalign_kv_gate_0.10", correct=False),
        _row("a", method="rotalign_kv_gate_0.15", correct=True),
        _row("b", method="rotalign_kv_gate_0.15", correct=True),
        _row("c", method="rotalign_kv_gate_0.15", correct=True),
        _row("d", method="rotalign_kv_gate_0.15", correct=False),
    ]

    payload = sweep.analyze_sweep(
        target_records=_target_rows(),
        teacher_records=_teacher_rows(),
        candidate_records=candidates,
        target_set_payload=_target_set(),
        config=sweep.SweepConfig(
            expected_n=4,
            min_numeric_coverage=4,
            min_clean_residual_recovered=2,
            target_self_correct=2,
        ),
    )

    assert payload["status"] == "matched_gate_candidate_for_controls"
    assert payload["passing_methods"] == ["rotalign_kv_gate_0.15"]
    assert payload["rows"][0]["method"] == "rotalign_kv_gate_0.15"
    assert payload["rows"][0]["clean_residual_recovered_ids"] == ["a", "b"]
    assert payload["rows"][1]["clean_residual_recovered_ids"] == ["a"]


def test_sweep_rejects_duplicate_ids_for_a_gate() -> None:
    candidates = [
        _row("a", method="rotalign_kv_gate_0.10", correct=True),
        _row("a", method="rotalign_kv_gate_0.10", correct=True),
        _row("c", method="rotalign_kv_gate_0.10", correct=True),
        _row("d", method="rotalign_kv_gate_0.10", correct=False),
    ]

    payload = sweep.analyze_sweep(
        target_records=_target_rows(),
        teacher_records=_teacher_rows(),
        candidate_records=candidates,
        target_set_payload=_target_set(),
        config=sweep.SweepConfig(
            expected_n=4,
            min_numeric_coverage=4,
            min_clean_residual_recovered=1,
            target_self_correct=2,
        ),
    )

    row = payload["rows"][0]
    assert row["status"] == "matched_candidate_below_clean_gate"
    assert row["duplicate_example_ids"] == ["a"]
    assert row["exact_ordered_id_parity"] is False


def test_sweep_uses_prediction_numeric_extraction_contract() -> None:
    candidates = [
        {
            **_row("a", method="rotalign_kv_gate_0.15", correct=True),
            "normalized_prediction": "1",
            "prediction": "no parseable answer",
        },
        _row("b", method="rotalign_kv_gate_0.15", correct=True),
        _row("c", method="rotalign_kv_gate_0.15", correct=True),
        _row("d", method="rotalign_kv_gate_0.15", correct=False),
    ]

    payload = sweep.analyze_sweep(
        target_records=_target_rows(),
        teacher_records=_teacher_rows(),
        candidate_records=candidates,
        target_set_payload=_target_set(),
        config=sweep.SweepConfig(
            expected_n=4,
            min_numeric_coverage=4,
            min_clean_residual_recovered=2,
            target_self_correct=2,
        ),
    )

    row = payload["rows"][0]
    assert row["numeric_extraction_coverage"] == 3
    assert row["status"] == "matched_candidate_below_clean_gate"


def test_cli_writes_json_and_markdown(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "target.jsonl"
    teacher = tmp_path / "teacher.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    target_set = tmp_path / "target_set.json"
    output_json = tmp_path / "sweep.json"
    output_md = tmp_path / "sweep.md"

    _write_jsonl(target, _target_rows())
    _write_jsonl(teacher, _teacher_rows())
    _write_jsonl(
        candidate,
        [
            _row("a", method="rotalign_kv_gate_0.15", correct=True),
            _row("b", method="rotalign_kv_gate_0.15", correct=True),
            _row("c", method="rotalign_kv_gate_0.15", correct=True),
            _row("d", method="rotalign_kv_gate_0.15", correct=False),
        ],
    )
    target_set.write_text(json.dumps(_target_set()), encoding="utf-8")

    payload = sweep.main(
        [
            "--target-jsonl",
            str(target),
            "--teacher-jsonl",
            str(teacher),
            "--candidate-jsonl",
            str(candidate),
            "--target-set-json",
            str(target_set),
            "--expected-n",
            "4",
            "--min-numeric-coverage",
            "4",
            "--target-self-correct",
            "2",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )

    assert payload["status"] == "matched_gate_candidate_for_controls"
    assert json.loads(output_json.read_text())["passing_methods"] == [
        "rotalign_kv_gate_0.15"
    ]
    assert "SVAMP32 Gate Sweep Clean-Target Readout" in output_md.read_text()
