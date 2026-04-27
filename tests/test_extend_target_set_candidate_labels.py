from __future__ import annotations

import json
import pathlib

from scripts import extend_target_set_candidate_labels as extend


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_extend_preserves_clean_ids_while_adding_candidates(tmp_path: pathlib.Path) -> None:
    candidate_path = tmp_path / "samples.jsonl"
    base_path = tmp_path / "target_set.json"
    output_json = tmp_path / "extended.json"
    output_md = tmp_path / "extended.md"

    _write_jsonl(
        candidate_path,
        [
            {"example_id": "a", "method": "sample", "prediction": "5", "normalized_prediction": "5", "correct": True},
            {"example_id": "b", "method": "sample", "prediction": "1", "normalized_prediction": "1", "correct": False},
        ],
    )
    base_path.write_text(
        json.dumps(
            {
                "reference_ids": ["a", "b", "c"],
                "reference_n": 3,
                "artifacts": {"baselines": []},
                "ids": {"clean_source_only": ["a", "b"], "target_only": ["c"]},
                "rows": [
                    {"example_id": "a", "labels": ["clean_source_only"]},
                    {"example_id": "b", "labels": ["clean_source_only"]},
                    {"example_id": "c", "labels": ["target_only"]},
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = extend.main(
        [
            "--base-target-set",
            str(base_path),
            "--id-fields",
            "clean_source_only",
            "--candidate",
            f"sample=path={candidate_path},method=sample",
            "--date",
            "2026-04-27",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )

    payload = json.loads(output_json.read_text())
    assert manifest["reference_n"] == 2
    assert payload["reference_ids"] == ["a", "b"]
    assert payload["ids"]["clean_source_only"] == ["a", "b"]
    assert payload["extension_preserves_clean_ids"] is True
    assert payload["artifacts"]["baselines"] == [
        {"label": "sample", "path": str(candidate_path), "method": "sample"}
    ]


def test_extend_accepts_explicit_ids(tmp_path: pathlib.Path) -> None:
    candidate_path = tmp_path / "samples.jsonl"
    base_path = tmp_path / "target_set.json"
    output_json = tmp_path / "extended.json"
    output_md = tmp_path / "extended.md"

    _write_jsonl(
        candidate_path,
        [
            {"example_id": "a", "method": "sample", "prediction": "5", "normalized_prediction": "5", "correct": True},
            {"example_id": "c", "method": "sample", "prediction": "7", "normalized_prediction": "7", "correct": True},
        ],
    )
    base_path.write_text(
        json.dumps(
            {
                "reference_ids": ["a", "b", "c"],
                "reference_n": 3,
                "artifacts": {"baselines": []},
                "ids": {"clean_source_only": ["a", "b"], "target_only": ["c"]},
                "rows": [
                    {"example_id": "a", "labels": ["clean_source_only"]},
                    {"example_id": "b", "labels": ["clean_source_only"]},
                    {"example_id": "c", "labels": ["target_only"]},
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = extend.main(
        [
            "--base-target-set",
            str(base_path),
            "--ids",
            "c",
            "a",
            "c",
            "--candidate",
            f"sample=path={candidate_path},method=sample",
            "--date",
            "2026-04-27",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )

    payload = json.loads(output_json.read_text())
    assert manifest["reference_n"] == 2
    assert manifest["id_source"] == {"type": "ids"}
    assert payload["reference_ids"] == ["a", "c"]
    assert payload["ids"]["clean_source_only"] == ["a"]
    assert payload["ids"]["target_only"] == ["c"]
    assert payload["candidate_extension"]["id_source"] == {"type": "ids"}


def test_extend_can_override_clean_residual_ids(tmp_path: pathlib.Path) -> None:
    candidate_path = tmp_path / "samples.jsonl"
    base_path = tmp_path / "target_set.json"
    output_json = tmp_path / "extended.json"
    output_md = tmp_path / "extended.md"

    _write_jsonl(
        candidate_path,
        [
            {"example_id": "a", "method": "sample", "prediction": "5", "normalized_prediction": "5", "correct": True},
            {"example_id": "b", "method": "sample", "prediction": "1", "normalized_prediction": "1", "correct": False},
        ],
    )
    base_path.write_text(
        json.dumps(
            {
                "reference_ids": ["a", "b", "c"],
                "reference_n": 3,
                "artifacts": {"baselines": []},
                "ids": {"clean_source_only": [], "clean_residual_targets": ["c"]},
                "rows": [
                    {"example_id": "a", "labels": []},
                    {"example_id": "b", "labels": []},
                    {"example_id": "c", "labels": ["clean_residual_targets"]},
                ],
            }
        ),
        encoding="utf-8",
    )

    extend.main(
        [
            "--base-target-set",
            str(base_path),
            "--ids",
            "a",
            "b",
            "--override-clean-residual-ids",
            "b",
            "--candidate",
            f"sample=path={candidate_path},method=sample",
            "--date",
            "2026-04-27",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )

    payload = json.loads(output_json.read_text())
    assert payload["ids"]["clean_residual_targets"] == ["b"]
    assert payload["candidate_extension"]["override_clean_residual_ids"] == ["b"]
    labels_by_id = {row["example_id"]: row["labels"] for row in payload["rows"]}
    assert "clean_residual_target_override" in labels_by_id["b"]
