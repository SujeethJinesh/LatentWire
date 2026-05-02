from __future__ import annotations

import json
import pathlib
from copy import deepcopy

from scripts import analyze_svamp32_paper_gate as gate


def _probe_payload() -> dict:
    return {
        "target_summary": {"correct": 8},
        "teacher_summary": {"correct": 16},
        "teacher_only_count": 10,
        "controls": [
            {
                "label": "target_self_repair",
                "correct": 14,
                "teacher_only_recovered_count": 3,
                "teacher_only_recovered_ids": ["a", "b", "c"],
            },
            {
                "label": "zero_source",
                "correct": 8,
                "teacher_only_recovered_count": 1,
                "teacher_only_recovered_ids": ["d"],
            },
            {
                "label": "shuffled_source",
                "correct": 9,
                "teacher_only_recovered_count": 1,
                "teacher_only_recovered_ids": ["d"],
            },
        ],
        "candidates": [
            {
                "label": "passes",
                "correct": 16,
                "losses_vs_target_count": 1,
                "teacher_only_recovered_count": 6,
                "teacher_only_recovered_ids": ["a", "b", "c", "d", "e", "f"],
            },
            {
                "label": "fails_like_query_pool",
                "correct": 9,
                "losses_vs_target_count": 1,
                "teacher_only_recovered_count": 1,
                "teacher_only_recovered_ids": ["d"],
            },
        ],
        "candidate_control_overlap": {
            "passes": {
                "zero_source": {
                    "candidate_teacher_only_retained_count": 1,
                    "candidate_teacher_only_retained_ids": ["d"],
                },
                "shuffled_source": {
                    "candidate_teacher_only_retained_count": 1,
                    "candidate_teacher_only_retained_ids": ["d"],
                },
                "target_self_repair": {
                    "candidate_teacher_only_retained_count": 3,
                    "candidate_teacher_only_retained_ids": ["a", "b", "c"],
                },
            },
            "fails_like_query_pool": {
                "zero_source": {
                    "candidate_teacher_only_retained_count": 1,
                    "candidate_teacher_only_retained_ids": ["d"],
                },
                "shuffled_source": {
                    "candidate_teacher_only_retained_count": 1,
                    "candidate_teacher_only_retained_ids": ["d"],
                },
                "target_self_repair": {
                    "candidate_teacher_only_retained_count": 0,
                    "candidate_teacher_only_retained_ids": [],
                },
            },
        },
    }


def _paper_probe_payload() -> dict:
    payload = deepcopy(_probe_payload())
    payload["reference_n"] = 32
    payload["teacher_only_ids"] = ["a", "b", "c", "d", "e", "f"]
    for role in ("target_summary", "teacher_summary"):
        payload[role].update(
            {
                "label": role.removesuffix("_summary"),
                "n": 32,
                "artifact_n": 32,
                "exact_ordered_id_parity": True,
                "numeric_extraction_coverage": 32,
            }
        )
    for group in ("controls", "candidates"):
        for row in payload[group]:
            row.update(
                {
                    "n": 32,
                    "artifact_n": 32,
                    "exact_ordered_id_parity": True,
                    "numeric_extraction_coverage": 32,
                }
            )
    return payload


def _target_set_payload() -> dict:
    return {
        "summary": {
            "required_clean_residual_to_clear_gate_if_preserving_self": 2,
        },
        "ids": {
            "teacher_only": ["a", "b", "c", "d", "e", "f"],
            "clean_residual_targets": ["e", "f"],
        },
    }


def test_gate_requires_target_self_repair_margin_and_unique_teacher_wins() -> None:
    payload = gate.evaluate_gate(_probe_payload(), config=gate.GateConfig())

    assert payload["status"] == "candidate_passes_target_self_repair_gate"
    by_label = {row["candidate_label"]: row for row in payload["candidates"]}
    assert by_label["passes"]["status"] == "passes_paper_gate"
    assert by_label["passes"]["unique_vs_target_self_repair_ids"] == ["d", "e", "f"]
    assert by_label["fails_like_query_pool"]["status"] == "fails_paper_gate"
    assert "min_correct" in by_label["fails_like_query_pool"]["failing_criteria"]
    assert "min_teacher_only" in by_label["fails_like_query_pool"]["failing_criteria"]


def test_missing_required_controls_cannot_pass() -> None:
    probe = _probe_payload()
    probe["controls"] = []
    probe["candidate_control_overlap"] = {}

    payload = gate.evaluate_gate(probe, config=gate.GateConfig())

    by_label = {row["candidate_label"]: row for row in payload["candidates"]}
    assert payload["status"] == "no_candidate_passes_target_self_repair_gate"
    assert by_label["passes"]["status"] == "fails_paper_gate"
    assert "target_self_repair_present" in by_label["passes"]["failing_criteria"]
    assert "source_controls_present" in by_label["passes"]["failing_criteria"]
    assert by_label["passes"]["missing_source_control_labels"] == [
        "zero_source",
        "shuffled_source",
    ]


def test_target_set_requires_clean_residual_recovery() -> None:
    payload = gate.evaluate_gate(
        _probe_payload(),
        config=gate.GateConfig(min_clean_residual_recovered=2),
        clean_residual_ids={"e", "f"},
    )

    by_label = {row["candidate_label"]: row for row in payload["candidates"]}
    assert by_label["passes"]["status"] == "passes_paper_gate"
    assert by_label["passes"]["clean_residual_recovered_ids"] == ["e", "f"]
    assert by_label["passes"]["clean_source_necessary_ids"] == ["e", "f"]
    assert by_label["fails_like_query_pool"]["status"] == "fails_paper_gate"
    assert "min_clean_residual_recovered" in by_label["fails_like_query_pool"]["failing_criteria"]


def test_source_controls_can_remove_clean_source_necessary_wins() -> None:
    payload = gate.evaluate_gate(
        _probe_payload(),
        config=gate.GateConfig(
            min_clean_residual_recovered=2,
            min_clean_source_necessary=2,
        ),
        clean_residual_ids={"d", "e", "f"},
    )

    by_label = {row["candidate_label"]: row for row in payload["candidates"]}
    assert by_label["passes"]["clean_residual_recovered_ids"] == ["d", "e", "f"]
    assert by_label["passes"]["clean_residual_retained_by_source_controls_ids"] == ["d"]
    assert by_label["passes"]["clean_source_necessary_ids"] == ["e", "f"]
    assert by_label["passes"]["status"] == "passes_paper_gate"
    assert by_label["fails_like_query_pool"]["clean_residual_recovered_ids"] == ["d"]
    assert by_label["fails_like_query_pool"]["clean_source_necessary_ids"] == []
    assert "min_clean_source_necessary" in by_label["fails_like_query_pool"]["failing_criteria"]


def test_clean_residual_requirement_fails_without_target_set() -> None:
    payload = gate.evaluate_gate(
        _probe_payload(),
        config=gate.GateConfig(min_clean_residual_recovered=2),
    )

    by_label = {row["candidate_label"]: row for row in payload["candidates"]}
    assert by_label["passes"]["status"] == "fails_paper_gate"
    assert "clean_residual_target_set_present" in by_label["passes"]["failing_criteria"]


def test_cli_writes_json_and_markdown(tmp_path: pathlib.Path) -> None:
    probe_json = tmp_path / "probe.json"
    output_json = tmp_path / "gate.json"
    output_md = tmp_path / "gate.md"
    probe_json.write_text(json.dumps(_probe_payload()), encoding="utf-8")

    payload = gate.main(
        [
            "--probe-json",
            str(probe_json),
            "--allow-legacy-gate",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )

    assert payload["passing_candidates"] == ["passes"]
    assert json.loads(output_json.read_text())["passing_candidates"] == ["passes"]
    assert "SVAMP32 Target-Self-Repair Paper Gate" in output_md.read_text()


def test_cli_uses_target_set_default_clean_residual_requirement(tmp_path: pathlib.Path) -> None:
    probe_json = tmp_path / "probe.json"
    target_set_json = tmp_path / "target_set.json"
    output_json = tmp_path / "gate.json"
    output_md = tmp_path / "gate.md"
    probe_json.write_text(json.dumps(_probe_payload()), encoding="utf-8")
    target_set_json.write_text(
        json.dumps(
            {
                "summary": {
                    "required_clean_residual_to_clear_gate_if_preserving_self": 2,
                },
                "ids": {
                    "teacher_only": ["a", "b", "c", "d", "e", "f"],
                    "clean_residual_targets": ["e", "f"],
                },
            }
        ),
        encoding="utf-8",
    )

    payload = gate.main(
        [
            "--probe-json",
            str(probe_json),
            "--target-set-json",
            str(target_set_json),
            "--allow-legacy-gate",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )

    assert payload["config"]["min_clean_residual_recovered"] == 2
    assert payload["config"]["min_clean_source_necessary"] == 2
    assert payload["clean_residual_target_set"]["ids"] == ["e", "f"]
    assert payload["passing_candidates"] == ["passes"]
    assert "minimum clean residual C2C-only recovered" in output_md.read_text()
    assert "minimum clean source-necessary recovered" in output_md.read_text()


def test_cli_requires_target_set_unless_legacy(tmp_path: pathlib.Path) -> None:
    probe_json = tmp_path / "probe.json"
    probe_json.write_text(json.dumps(_probe_payload()), encoding="utf-8")

    try:
        gate.main(
            [
                "--probe-json",
                str(probe_json),
                "--output-json",
                str(tmp_path / "gate.json"),
                "--output-md",
                str(tmp_path / "gate.md"),
            ]
        )
    except ValueError as exc:
        assert "requires --target-set-json" in str(exc)
    else:
        raise AssertionError("expected missing target set failure")


def test_strict_paper_provenance_rejects_subset_artifact(tmp_path: pathlib.Path) -> None:
    probe = _paper_probe_payload()
    probe["controls"][0]["artifact_n"] = 70
    probe_json = tmp_path / "probe.json"
    target_set_json = tmp_path / "target_set.json"
    probe_json.write_text(json.dumps(probe), encoding="utf-8")
    target_set_json.write_text(json.dumps(_target_set_payload()), encoding="utf-8")

    try:
        gate.main(
            [
                "--probe-json",
                str(probe_json),
                "--target-set-json",
                str(target_set_json),
                "--output-json",
                str(tmp_path / "gate.json"),
                "--output-md",
                str(tmp_path / "gate.md"),
            ]
        )
    except ValueError as exc:
        assert "SVAMP32 paper provenance validation failed" in str(exc)
        assert "control.target_self_repair" in str(exc)
        assert "artifact_n=70 != reference_n=32" in str(exc)
    else:
        raise AssertionError("expected strict provenance failure")


def test_strict_paper_provenance_rejects_target_set_mismatch() -> None:
    target_set_payload = _target_set_payload()
    target_set_payload["ids"]["teacher_only"] = ["a", "b"]

    try:
        gate.validate_paper_provenance(
            _paper_probe_payload(),
            target_set_payload,
            expected_n=32,
            min_numeric_coverage=31,
        )
    except ValueError as exc:
        assert "teacher_only does not match" in str(exc)
    else:
        raise AssertionError("expected target-set mismatch failure")


def test_strict_paper_provenance_rejects_low_numeric_coverage() -> None:
    probe = _paper_probe_payload()
    probe["candidates"][0]["numeric_extraction_coverage"] = 30

    try:
        gate.validate_paper_provenance(
            probe,
            _target_set_payload(),
            expected_n=32,
            min_numeric_coverage=31,
        )
    except ValueError as exc:
        assert "numeric_extraction_coverage=30 < 31" in str(exc)
    else:
        raise AssertionError("expected numeric coverage failure")
