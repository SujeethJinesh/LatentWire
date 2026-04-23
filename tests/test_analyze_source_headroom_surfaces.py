from __future__ import annotations

import json
import pathlib

from scripts import analyze_source_headroom_surfaces as surfaces


def _record(example_id: str | None, *, method: str, correct: bool, index: int) -> dict:
    row = {
        "answer": ["1"],
        "correct": correct,
        "index": index,
        "method": method,
        "normalized_prediction": "1" if correct else "0",
        "prediction": "answer is 1" if correct else "answer is 0",
    }
    if example_id is not None:
        row["example_id"] = example_id
    return row


def _write_jsonl(path: pathlib.Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def test_same_file_surface_ranks_source_complementary_headroom(
    tmp_path: pathlib.Path,
) -> None:
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions,
        [
            _record("a", method="target_alone", correct=False, index=0),
            _record("b", method="target_alone", correct=True, index=1),
            _record("c", method="target_alone", correct=False, index=2),
            _record("a", method="text_to_text", correct=True, index=0),
            _record("b", method="text_to_text", correct=True, index=1),
            _record("c", method="text_to_text", correct=True, index=2),
        ],
    )

    payload = surfaces.main(
        [
            "--surface",
            f"toy=path={predictions},target_method=target_alone,source_method=text_to_text",
            "--min-source-only",
            "2",
            "--output-json",
            str(tmp_path / "scan.json"),
            "--output-md",
            str(tmp_path / "scan.md"),
        ]
    )

    row = payload["surfaces"][0]
    assert row["status"] == "strong_source_complementary_surface"
    assert row["overlap"]["source_only_count"] == 2
    assert row["overlap"]["target_or_source_oracle_count"] == 3
    assert payload["recommended_next_gate"] == "run learned connector on highest-ranked strong surface"


def test_cross_file_surface_uses_eval_file_to_attach_missing_ids(
    tmp_path: pathlib.Path,
) -> None:
    eval_file = tmp_path / "eval.jsonl"
    _write_jsonl(
        eval_file,
        [
            {"question": "q0", "answer": "1"},
            {"question": "q1", "answer": "1"},
        ],
    )
    target = tmp_path / "target.jsonl"
    source = tmp_path / "source.jsonl"
    _write_jsonl(
        target,
        [
            _record(None, method="target_alone", correct=False, index=0),
            _record(None, method="target_alone", correct=True, index=1),
        ],
    )
    _write_jsonl(
        source,
        [
            _record(None, method="source_alone", correct=True, index=0),
            _record(None, method="source_alone", correct=False, index=1),
        ],
    )

    payload = surfaces.main(
        [
            "--surface",
            (
                f"cross=target_path={target},source_path={source},"
                f"target_method=target_alone,source_method=source_alone,eval_file={eval_file}"
            ),
            "--min-source-only",
            "1",
            "--output-json",
            str(tmp_path / "scan.json"),
            "--output-md",
            str(tmp_path / "scan.md"),
        ]
    )

    row = payload["surfaces"][0]
    assert row["id_sources"] == {
        "exact_ordered_id_parity": True,
        "set_id_parity": True,
        "source": "eval_file",
        "strict_ids": True,
        "target": "eval_file",
    }
    assert row["overlap"]["source_only_count"] == 1
    assert row["status"] == "strong_source_complementary_surface"


def test_index_fallback_marks_surface_weak(tmp_path: pathlib.Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions,
        [
            _record(None, method="target_alone", correct=False, index=0),
            _record(None, method="source_alone", correct=True, index=0),
        ],
    )

    payload = surfaces.main(
        [
            "--surface",
            f"weak=path={predictions},target_method=target_alone,source_method=source_alone",
            "--min-source-only",
            "1",
            "--output-json",
            str(tmp_path / "scan.json"),
            "--output-md",
            str(tmp_path / "scan.md"),
        ]
    )

    assert payload["surfaces"][0]["status"] == "weak_index_only_surface"
