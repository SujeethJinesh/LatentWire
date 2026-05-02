from __future__ import annotations

import json
import pathlib

from scripts import build_svamp32_innovation_target_set as target_set


def _probe_payload() -> dict:
    return {
        "target_summary": {"correct": 8},
        "teacher_summary": {"correct": 16},
        "teacher_only_ids": ["a", "b", "c", "d", "e", "f"],
        "sources": [
            {
                "label": "source",
                "teacher_only_recovered_ids": ["c"],
            },
            {
                "label": "t2t",
                "teacher_only_recovered_ids": [],
            },
        ],
        "controls": [
            {
                "label": "target_self_repair",
                "correct": 12,
                "teacher_only_recovered_ids": ["a", "b"],
            },
            {
                "label": "zero_source",
                "teacher_only_recovered_ids": ["c"],
            },
            {
                "label": "shuffled_source",
                "teacher_only_recovered_ids": [],
            },
        ],
        "candidates": [
            {
                "label": "candidate",
                "teacher_only_recovered_ids": ["d"],
            }
        ],
        "teacher_only_provenance": [
            {
                "example_id": "a",
                "source_correct_labels": [],
                "control_correct_labels": ["target_self_repair"],
                "candidate_correct_labels": [],
            },
            {
                "example_id": "b",
                "source_correct_labels": [],
                "control_correct_labels": ["target_self_repair"],
                "candidate_correct_labels": [],
            },
            {
                "example_id": "c",
                "source_correct_labels": ["source"],
                "control_correct_labels": ["zero_source"],
                "candidate_correct_labels": [],
            },
            {
                "example_id": "d",
                "source_correct_labels": [],
                "control_correct_labels": [],
                "candidate_correct_labels": ["candidate"],
            },
            {
                "example_id": "e",
                "source_correct_labels": [],
                "control_correct_labels": [],
                "candidate_correct_labels": [],
            },
            {
                "example_id": "f",
                "source_correct_labels": [],
                "control_correct_labels": [],
                "candidate_correct_labels": [],
            },
        ],
    }


def test_build_target_set_separates_clean_residual_ids() -> None:
    payload = target_set.build_target_set(
        _probe_payload(),
        config=target_set.TargetSetConfig(
            min_correct=14,
            min_teacher_only=4,
            min_unique_vs_target_self=2,
        ),
    )

    assert payload["status"] == "residual_headroom_available"
    assert payload["ids"]["target_self_repair"] == ["a", "b"]
    assert payload["ids"]["source_explained"] == ["c"]
    assert payload["ids"]["source_control_explained"] == ["c"]
    assert payload["ids"]["clean_residual_targets"] == ["d", "e", "f"]
    assert payload["summary"]["oracle_target_self_plus_teacher_correct"] == 16
    assert payload["summary"]["required_clean_residual_to_clear_gate_if_preserving_self"] == 2


def test_cli_writes_json_and_markdown(tmp_path: pathlib.Path) -> None:
    probe_json = tmp_path / "probe.json"
    output_json = tmp_path / "target_set.json"
    output_md = tmp_path / "target_set.md"
    probe_json.write_text(json.dumps(_probe_payload()), encoding="utf-8")

    payload = target_set.main(
        [
            "--probe-json",
            str(probe_json),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--min-correct",
            "14",
            "--min-teacher-only",
            "4",
        ]
    )

    assert payload["ids"]["clean_residual_targets"] == ["d", "e", "f"]
    assert json.loads(output_json.read_text())["status"] == "residual_headroom_available"
    assert "SVAMP32 Innovation Target Set" in output_md.read_text()
