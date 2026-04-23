from __future__ import annotations

import json
import pathlib

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


def test_cli_writes_json_and_markdown(tmp_path: pathlib.Path) -> None:
    probe_json = tmp_path / "probe.json"
    output_json = tmp_path / "gate.json"
    output_md = tmp_path / "gate.md"
    probe_json.write_text(json.dumps(_probe_payload()), encoding="utf-8")

    payload = gate.main(
        [
            "--probe-json",
            str(probe_json),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )

    assert payload["passing_candidates"] == ["passes"]
    assert json.loads(output_json.read_text())["passing_candidates"] == ["passes"]
    assert "SVAMP32 Target-Self-Repair Paper Gate" in output_md.read_text()
