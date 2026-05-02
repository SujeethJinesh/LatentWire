from __future__ import annotations

import json
import pathlib
from copy import deepcopy

from scripts import analyze_svamp32_source_sidecar_bound as sidecar


def _summary(label: str, *, correct: int, teacher_ids: list[str] | None = None) -> dict:
    return {
        "label": label,
        "correct": correct,
        "n": 32,
        "artifact_n": 32,
        "exact_ordered_id_parity": True,
        "numeric_extraction_coverage": 32,
        "teacher_only_recovered_count": len(teacher_ids or []),
        "teacher_only_recovered_ids": teacher_ids or [],
    }


def _probe_payload() -> dict:
    return {
        "reference_n": 32,
        "teacher_only_count": 6,
        "teacher_only_ids": ["clean1", "clean2", "retained", "self1", "self2", "self3"],
        "target_summary": _summary("target", correct=8),
        "teacher_summary": _summary("teacher", correct=16),
        "sources": [],
        "controls": [
            _summary(
                "target_self_repair",
                correct=14,
                teacher_ids=["self1", "self2", "self3"],
            ),
            _summary("translated_kv_zero", correct=8, teacher_ids=["retained"]),
        ],
        "candidates": [
            _summary("one_clean", correct=10, teacher_ids=["clean1", "retained"]),
            _summary("two_clean", correct=11, teacher_ids=["clean1", "clean2"]),
        ],
        "candidate_control_overlap": {
            "one_clean": {
                "translated_kv_zero": {
                    "candidate_teacher_only_retained_count": 1,
                    "candidate_teacher_only_retained_ids": ["retained"],
                    "control_teacher_only_recovered_count": 1,
                }
            },
            "two_clean": {
                "translated_kv_zero": {
                    "candidate_teacher_only_retained_count": 0,
                    "candidate_teacher_only_retained_ids": [],
                    "control_teacher_only_recovered_count": 1,
                }
            },
        },
    }


def _target_set_payload() -> dict:
    return {
        "ids": {
            "teacher_only": ["clean1", "clean2", "retained", "self1", "self2", "self3"],
            "clean_residual_targets": ["clean1", "clean2"],
        }
    }


def test_one_clean_id_fails_sidecar_bound() -> None:
    payload = sidecar.evaluate_sidecar_bound(
        _probe_payload(),
        _target_set_payload(),
        config=sidecar.SidecarGateConfig(candidate_label="one_clean"),
    )

    assert payload["status"] == "oracle_sidecar_bound_fails_gate"
    assert payload["oracle_sidecar_bound"]["correct"] == 15
    assert payload["candidate"]["clean_source_necessary_ids"] == ["clean1"]
    assert "min_correct" in payload["oracle_sidecar_bound"]["failing_criteria"]
    assert "min_clean_source_necessary" in payload["oracle_sidecar_bound"]["failing_criteria"]


def test_two_clean_ids_clear_oracle_bound_but_not_method() -> None:
    payload = sidecar.evaluate_sidecar_bound(
        _probe_payload(),
        _target_set_payload(),
        config=sidecar.SidecarGateConfig(candidate_label="two_clean"),
    )

    assert payload["status"] == "oracle_sidecar_bound_clears_gate_not_method"
    assert payload["oracle_sidecar_bound"]["correct"] == 16
    assert payload["oracle_sidecar_bound"]["delta_vs_target_self"] == 2
    assert payload["oracle_sidecar_bound"]["target_losses_vs_target_self"] == 0
    assert payload["oracle_sidecar_bound"]["clean_source_necessary_ids"] == [
        "clean1",
        "clean2",
    ]


def test_source_control_retention_removes_clean_source_necessary_id() -> None:
    probe = deepcopy(_probe_payload())
    probe["candidate_control_overlap"]["two_clean"]["translated_kv_zero"] = {
        "candidate_teacher_only_retained_count": 1,
        "candidate_teacher_only_retained_ids": ["clean2"],
        "control_teacher_only_recovered_count": 1,
    }

    payload = sidecar.evaluate_sidecar_bound(
        probe,
        _target_set_payload(),
        config=sidecar.SidecarGateConfig(candidate_label="two_clean"),
    )

    assert payload["status"] == "oracle_sidecar_bound_fails_gate"
    assert payload["candidate"]["retained_by_source_controls_ids"] == ["clean2"]
    assert payload["oracle_sidecar_bound"]["clean_source_necessary_ids"] == ["clean1"]


def test_cli_writes_json_and_markdown(tmp_path: pathlib.Path) -> None:
    probe_json = tmp_path / "probe.json"
    target_set_json = tmp_path / "target_set.json"
    output_json = tmp_path / "sidecar.json"
    output_md = tmp_path / "sidecar.md"
    probe_json.write_text(json.dumps(_probe_payload()), encoding="utf-8")
    target_set_json.write_text(json.dumps(_target_set_payload()), encoding="utf-8")

    payload = sidecar.main(
        [
            "--probe-json",
            str(probe_json),
            "--target-set-json",
            str(target_set_json),
            "--candidate-label",
            "two_clean",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )

    assert payload["status"] == "oracle_sidecar_bound_clears_gate_not_method"
    assert json.loads(output_json.read_text())["status"] == payload["status"]
    assert "SVAMP32 Source-Innovation Sidecar Bound" in output_md.read_text()
