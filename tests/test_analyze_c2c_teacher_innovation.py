from __future__ import annotations

import json
import pathlib

from scripts import analyze_c2c_teacher_innovation as probe


def _record(example_id: str, *, method: str, correct: bool, index: int) -> dict:
    return {
        "answer": ["1"],
        "correct": correct,
        "example_id": example_id,
        "index": index,
        "method": method,
        "normalized_prediction": "1" if correct else "0",
        "prediction": "answer is 1" if correct else "answer is 0",
    }


def _write_jsonl(path: pathlib.Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_probe_identifies_candidate_teacher_recovery_and_control_overlap(
    tmp_path: pathlib.Path,
) -> None:
    target = tmp_path / "target.jsonl"
    teacher = tmp_path / "teacher.jsonl"
    source = tmp_path / "source.jsonl"
    control = tmp_path / "control.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    ids = ["a", "b", "c", "d"]

    _write_jsonl(
        target,
        [
            _record("a", method="target_alone", correct=False, index=0),
            _record("b", method="target_alone", correct=True, index=1),
            _record("c", method="target_alone", correct=False, index=2),
            _record("d", method="target_alone", correct=False, index=3),
        ],
    )
    _write_jsonl(
        teacher,
        [
            _record("a", method="c2c", correct=True, index=0),
            _record("b", method="c2c", correct=True, index=1),
            _record("c", method="c2c", correct=True, index=2),
            _record("d", method="c2c", correct=False, index=3),
        ],
    )
    _write_jsonl(
        source,
        [
            _record(example_id, method="source_alone", correct=(example_id == "a"), index=idx)
            for idx, example_id in enumerate(ids)
        ],
    )
    _write_jsonl(
        control,
        [
            _record(example_id, method="target_self_repair", correct=(example_id == "a"), index=idx)
            for idx, example_id in enumerate(ids)
        ],
    )
    _write_jsonl(
        candidate,
        [
            _record(example_id, method="process_repair", correct=(example_id in {"a", "c"}), index=idx)
            for idx, example_id in enumerate(ids)
        ],
    )

    payload = probe.main(
        [
            "--target",
            f"target=path={target},method=target_alone",
            "--teacher",
            f"c2c=path={teacher},method=c2c_generate",
            "--source",
            f"source=path={source},method=source_alone",
            "--control",
            f"self_repair=path={control},method=target_self_repair",
            "--candidate",
            f"repair=path={candidate},method=process_repair",
            "--output-json",
            str(tmp_path / "probe.json"),
            "--output-md",
            str(tmp_path / "probe.md"),
        ]
    )

    assert payload["teacher_only_ids"] == ["a", "c"]
    repair = payload["candidates"][0]
    assert repair["teacher_only_recovered_count"] == 2
    assert repair["teacher_probe_status"] == "partial_control_overlap"
    overlap = payload["candidate_control_overlap"]["repair"]["self_repair"]
    assert overlap["candidate_teacher_only_retained_count"] == 1
    provenance = {row["example_id"]: row for row in payload["teacher_only_provenance"]}
    assert provenance["a"]["source_correct_labels"] == ["source"]
    assert provenance["a"]["control_correct_labels"] == ["self_repair"]
    assert provenance["c"]["candidate_correct_labels"] == ["repair"]
    assert "# C2C Teacher Innovation Probe" in (tmp_path / "probe.md").read_text()


def test_probe_rejects_rows_missing_reference_ids(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "target.jsonl"
    teacher = tmp_path / "teacher.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write_jsonl(
        target,
        [
            _record("a", method="target_alone", correct=False, index=0),
            _record("b", method="target_alone", correct=False, index=1),
        ],
    )
    _write_jsonl(
        teacher,
        [
            _record("a", method="c2c", correct=True, index=0),
            _record("b", method="c2c", correct=True, index=1),
        ],
    )
    _write_jsonl(candidate, [_record("a", method="candidate", correct=True, index=0)])

    try:
        probe.main(
            [
                "--target",
                f"target=path={target},method=target_alone",
                "--teacher",
                f"c2c=path={teacher},method=c2c_generate",
                "--candidate",
                f"candidate=path={candidate},method=candidate",
                "--output-json",
                str(tmp_path / "probe.json"),
                "--output-md",
                str(tmp_path / "probe.md"),
            ]
        )
    except ValueError as exc:
        assert "Missing reference IDs" in str(exc)
    else:
        raise AssertionError("expected missing reference ID validation failure")


def test_require_exact_artifacts_rejects_subsetted_candidate(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "target.jsonl"
    teacher = tmp_path / "teacher.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write_jsonl(
        target,
        [
            _record("a", method="target_alone", correct=False, index=0),
            _record("b", method="target_alone", correct=False, index=1),
        ],
    )
    _write_jsonl(
        teacher,
        [
            _record("a", method="c2c", correct=True, index=0),
            _record("b", method="c2c", correct=True, index=1),
        ],
    )
    _write_jsonl(
        candidate,
        [
            _record("a", method="candidate", correct=True, index=0),
            _record("b", method="candidate", correct=False, index=1),
            _record("c", method="candidate", correct=True, index=2),
        ],
    )

    try:
        probe.main(
            [
                "--target",
                f"target=path={target},method=target_alone",
                "--teacher",
                f"c2c=path={teacher},method=c2c_generate",
                "--candidate",
                f"candidate=path={candidate},method=candidate",
                "--require-exact-artifacts",
                "--output-json",
                str(tmp_path / "probe.json"),
                "--output-md",
                str(tmp_path / "probe.md"),
            ]
        )
    except ValueError as exc:
        assert "Non-exact artifacts" in str(exc)
        assert "candidate.candidate" in str(exc)
        assert "artifact_n=3 != reference_n=2" in str(exc)
    else:
        raise AssertionError("expected strict artifact validation failure")
