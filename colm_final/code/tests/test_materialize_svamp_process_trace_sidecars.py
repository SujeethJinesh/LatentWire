from __future__ import annotations

import json
import pathlib

from scripts import materialize_svamp_process_trace_sidecars as process_sidecars


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_process_trace_sidecars_score_masked_reasoning_text(tmp_path: pathlib.Path) -> None:
    source_path = tmp_path / "source.jsonl"
    target_path = tmp_path / "target.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    target_set_path = tmp_path / "target_set.json"
    output_dir = tmp_path / "sidecars"

    _write_jsonl(
        source_path,
        [
            {
                "example_id": "a",
                "method": "source",
                "prediction": "Use baskets: red plus green per basket, then divide total by each basket.",
                "normalized_prediction": "0",
                "correct": False,
                "answer": ["1"],
            }
        ],
    )
    _write_jsonl(
        target_path,
        [
            {
                "example_id": "a",
                "method": "target",
                "prediction": "Guess the answer.",
                "normalized_prediction": "3",
                "correct": False,
                "answer": ["1"],
            }
        ],
    )
    _write_jsonl(
        candidate_path,
        [
            {
                "example_id": "a",
                "method": "candidate",
                "prediction": "Find red plus green per basket and divide total by basket size.",
                "normalized_prediction": "1",
                "correct": True,
                "answer": ["1"],
            }
        ],
    )
    target_set_path.write_text(
        json.dumps(
            {
                "reference_ids": ["a"],
                "reference_n": 1,
                "artifacts": {
                    "target": {"label": "target", "path": str(target_path), "method": "target"},
                    "source": {"label": "source", "path": str(source_path), "method": "source"},
                    "baselines": [
                        {"label": "candidate", "path": str(candidate_path), "method": "candidate"}
                    ],
                    "controls": [],
                },
                "ids": {"clean_residual_targets": ["a"]},
                "rows": [{"example_id": "a", "labels": ["clean_residual_targets"]}],
            }
        ),
        encoding="utf-8",
    )

    manifest = process_sidecars.main(
        [
            "--target-set",
            str(target_set_path),
            "--output-dir",
            str(output_dir),
            "--sidecar-bits",
            "64",
            "--date",
            "2026-04-27",
        ]
    )

    rows = [json.loads(line) for line in (output_dir / "live_candidate_sidecars.jsonl").read_text().splitlines()]
    assert manifest["summary"]["n"] == 1
    assert manifest["summary"]["source_vector_telemetry"]["feature_count"] > 0
    assert rows[0]["candidate_scores"][0]["value"] == "1"
    assert rows[0]["candidate_scores"][0]["label"] == "candidate"
    assert rows[0]["profile_mode"] == "answer_masked_process_trace"
