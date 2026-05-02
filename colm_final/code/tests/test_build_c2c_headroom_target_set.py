from __future__ import annotations

from scripts import build_c2c_headroom_target_set as target_set


def _row(label: str, ids: list[str]) -> dict:
    return {
        "label": label,
        "teacher_only_recovered_ids": ids,
    }


def test_build_target_set_separates_source_explained_and_clean_ids() -> None:
    payload = {
        "reference_n": 4,
        "target_summary": {"correct": 1},
        "teacher_summary": {"correct": 3, "losses_vs_target_count": 1},
        "teacher_only_ids": ["a", "b", "c"],
        "sources": [_row("source", ["a"]), _row("t2t", [])],
        "controls": [_row("zero_source", ["b"])],
        "candidates": [_row("candidate", ["c"])],
        "teacher_only_provenance": [
            {"example_id": "a", "source_correct_labels": ["source"]},
            {"example_id": "b", "control_correct_labels": ["zero_source"]},
            {"example_id": "c", "candidate_correct_labels": ["candidate"]},
        ],
    }

    built = target_set.build_target_set(
        payload,
        config=target_set.HeadroomConfig(
            source_labels=("source", "t2t"),
            control_labels=("zero_source",),
            candidate_labels=("candidate",),
            min_teacher_only=3,
            min_clean_teacher_only=1,
        ),
    )

    assert built["status"] == "clean_headroom_available"
    assert built["summary"]["source_explained_teacher_only_count"] == 1
    assert built["summary"]["control_explained_teacher_only_count"] == 1
    assert built["summary"]["clean_teacher_only_count"] == 1
    assert built["ids"]["clean_teacher_only"] == ["c"]
    assert built["teacher_only_rows"][0]["source_correct_labels"] == ["source"]


def test_status_fails_when_clean_headroom_below_threshold() -> None:
    payload = {
        "reference_n": 2,
        "target_summary": {"correct": 1},
        "teacher_summary": {"correct": 2},
        "teacher_only_ids": ["a"],
        "sources": [_row("source", ["a"])],
    }

    built = target_set.build_target_set(
        payload,
        config=target_set.HeadroomConfig(min_teacher_only=1, min_clean_teacher_only=1),
    )

    assert built["status"] == "insufficient_clean_headroom"
    assert built["ids"]["clean_teacher_only"] == []
