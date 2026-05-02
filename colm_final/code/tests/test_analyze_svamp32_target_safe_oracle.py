from __future__ import annotations

import json

from scripts import analyze_svamp32_target_safe_oracle as oracle


def _write_jsonl(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(example_id: str, method: str, correct: bool, prediction: str) -> dict:
    return {
        "answer": [prediction],
        "correct": correct,
        "example_id": example_id,
        "method": method,
        "normalized_prediction": prediction,
        "prediction": f"Final answer: {prediction}",
    }


def test_candidate_oracle_counts_clean_source_necessary_ids(tmp_path) -> None:
    ids = ["a", "b", "c"]
    target = [_row(ids[0], "target", True, "1"), _row(ids[1], "target", False, "0"), _row(ids[2], "target", False, "0")]
    teacher = [_row(ids[0], "teacher", True, "1"), _row(ids[1], "teacher", True, "2"), _row(ids[2], "teacher", True, "3")]
    baseline = [_row(ids[0], "baseline", True, "1"), _row(ids[1], "baseline", False, "0"), _row(ids[2], "baseline", False, "0")]
    candidate = [_row(ids[0], "candidate", False, "9"), _row(ids[1], "candidate", True, "2"), _row(ids[2], "candidate", True, "3")]
    control = [_row(ids[0], "control", False, "9"), _row(ids[1], "control", False, "0"), _row(ids[2], "control", False, "0")]
    target_set = {"ids": {"clean_residual_targets": ["b", "c"]}}

    paths = {}
    for name, rows in {
        "target": target,
        "teacher": teacher,
        "baseline": baseline,
        "candidate": candidate,
        "control": control,
    }.items():
        path = tmp_path / f"{name}.jsonl"
        _write_jsonl(path, rows)
        paths[name] = path
    target_set_path = tmp_path / "target_set.json"
    target_set_path.write_text(json.dumps(target_set), encoding="utf-8")

    payload = oracle.main(
        [
            "--target",
            f"target=path={paths['target']},method=target",
            "--teacher",
            f"teacher=path={paths['teacher']},method=teacher",
            "--baseline",
            f"baseline=path={paths['baseline']},method=baseline",
            "--candidate",
            f"candidate=path={paths['candidate']},method=candidate",
            "--control",
            f"control=path={paths['control']},method=control",
            "--target-set-json",
            str(target_set_path),
            "--min-clean-source-necessary",
            "2",
            "--output-json",
            str(tmp_path / "out.json"),
            "--output-md",
            str(tmp_path / "out.md"),
        ]
    )

    assert payload["status"] == "oracle_has_enough_clean_source_signal"
    assert payload["candidate_oracle"]["correct"] == 3
    assert payload["clean_source_necessary"]["ids"] == ["b", "c"]


def test_control_oracle_removes_clean_source_necessary_id(tmp_path) -> None:
    ids = ["a", "b"]
    target = [_row(ids[0], "target", True, "1"), _row(ids[1], "target", False, "0")]
    teacher = [_row(ids[0], "teacher", True, "1"), _row(ids[1], "teacher", True, "2")]
    baseline = [_row(ids[0], "baseline", True, "1"), _row(ids[1], "baseline", False, "0")]
    candidate = [_row(ids[0], "candidate", False, "9"), _row(ids[1], "candidate", True, "2")]
    control = [_row(ids[0], "control", False, "9"), _row(ids[1], "control", True, "2")]
    target_set = {"ids": {"clean_residual_targets": ["b"]}}

    paths = {}
    for name, rows in {
        "target": target,
        "teacher": teacher,
        "baseline": baseline,
        "candidate": candidate,
        "control": control,
    }.items():
        path = tmp_path / f"{name}.jsonl"
        _write_jsonl(path, rows)
        paths[name] = path
    target_set_path = tmp_path / "target_set.json"
    target_set_path.write_text(json.dumps(target_set), encoding="utf-8")

    payload = oracle.main(
        [
            "--target",
            f"target=path={paths['target']},method=target",
            "--teacher",
            f"teacher=path={paths['teacher']},method=teacher",
            "--baseline",
            f"baseline=path={paths['baseline']},method=baseline",
            "--candidate",
            f"candidate=path={paths['candidate']},method=candidate",
            "--control",
            f"control=path={paths['control']},method=control",
            "--target-set-json",
            str(target_set_path),
            "--min-clean-source-necessary",
            "1",
            "--output-json",
            str(tmp_path / "out.json"),
            "--output-md",
            str(tmp_path / "out.md"),
        ]
    )

    assert payload["status"] == "oracle_lacks_clean_source_signal"
    assert payload["clean_source_necessary"]["candidate_clean_ids"] == ["b"]
    assert payload["clean_source_necessary"]["control_clean_ids"] == ["b"]
    assert payload["clean_source_necessary"]["ids"] == []
