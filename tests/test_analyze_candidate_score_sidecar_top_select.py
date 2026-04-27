from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

from scripts import analyze_candidate_score_sidecar_top_select as top_select


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_top_select_requires_confident_candidate() -> None:
    rows = [
        {
            "example_id": "a",
            "conditions": {
                "matched": {"correct": True, "accepted": True},
                "shuffled_source": {"correct": False, "accepted": False},
                "random_sidecar": {"correct": False, "accepted": False},
                "target_only": {"correct": False, "accepted": False},
                "slots_only": {"correct": False, "accepted": False},
            },
        }
    ]

    summary = top_select._summarize(rows, clean_ids={"a"}, target_correct_ids=set())

    assert summary["matched"]["clean_correct_ids"] == ["a"]
    assert summary["source_necessary_clean_ids"] == ["a"]
    assert summary["control_clean_union_ids"] == []


def test_load_sidecars_rejects_duplicate_ids(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "sidecars.jsonl"
    _write_jsonl(
        path,
        [
            {"example_id": "a", "candidate_scores": []},
            {"example_id": "a", "candidate_scores": []},
        ],
    )

    try:
        top_select._load_sidecars(path)
    except ValueError as exc:
        assert "Duplicate sidecar IDs" in str(exc)
    else:
        raise AssertionError("expected duplicate sidecar rejection")


def test_random_sidecar_breaks_score_value_mapping() -> None:
    sidecars = {
        "b": {
            "example_id": "b",
            "candidate_scores": [
                {"label": "correct", "value": "7", "score": 10.0},
                {"label": "wrong_a", "value": "2", "score": 4.0},
                {"label": "wrong_b", "value": "3", "score": 1.0},
            ],
        }
    }
    surface = SimpleNamespace(reference_ids=["b"])

    randomized = top_select._condition_sidecar(
        condition="random_sidecar",
        sidecars=sidecars,
        surface=surface,
        index=0,
    )

    assert randomized is not None
    assert top_select._top_from_sidecar(sidecars["b"])[0] == "7"
    assert top_select._top_from_sidecar(randomized)[0] != "7"
    assert sorted(item["score"] for item in randomized["candidate_scores"]) == [1.0, 4.0, 10.0]
