from __future__ import annotations

import json
import pathlib

import pytest

from scripts import rank_source_contrastive_target_sets as ranker


def _target_set(
    *,
    clean: list[str],
    source_only: list[str],
    target_correct: int = 1,
    source_correct: int = 3,
    oracle: int = 4,
    reference_ids: list[str] | None = None,
    exact_ordered_id_parity: bool = True,
    control_union: list[str] | None = None,
    baseline_union: list[str] | None = None,
    target_numeric: int = 4,
    source_numeric: int = 4,
) -> dict:
    reference_ids = reference_ids or ["a", "b", "c", "d"]
    return {
        "status": "source_contrastive_target_set_ready",
        "reference_n": len(reference_ids),
        "reference_ids": reference_ids,
        "provenance": {
            "exact_ordered_id_parity": exact_ordered_id_parity,
        },
        "summaries": {
            "target": {"label": "target", "n": len(reference_ids), "correct": target_correct, "numeric_coverage": target_numeric},
            "source": {"label": "source", "n": len(reference_ids), "correct": source_correct, "numeric_coverage": source_numeric},
            "controls": {"zero": {"label": "zero", "n": len(reference_ids), "correct": 0, "numeric_coverage": len(reference_ids)}},
            "baselines": {},
        },
        "ids": {
            "source_only": source_only,
            "clean_source_only": clean,
            "target_only": ["t"],
            "control_union": control_union or [],
            "baseline_union": baseline_union or [],
        },
        "counts": {
            "target_correct": target_correct,
            "source_correct": source_correct,
            "source_only": len(source_only),
            "clean_source_only": len(clean),
            "target_only": 1,
            "target_or_source_oracle": oracle,
        },
    }


def _write(path: pathlib.Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_ranks_by_clean_source_only_not_raw_source_only(tmp_path: pathlib.Path) -> None:
    raw_big = tmp_path / "raw_big.json"
    clean_big = tmp_path / "clean_big.json"
    _write(raw_big, _target_set(clean=["a"], source_only=["a", "b", "c", "d"], oracle=2))
    _write(clean_big, _target_set(clean=["a", "b", "c"], source_only=["a", "b", "c"], oracle=4))

    payload = ranker.main(
        [
            "--target-set",
            f"raw=path={raw_big},role=live",
            "--target-set",
            f"clean=path={clean_big},role=live",
            "--min-clean-source-only",
            "3",
            "--output-json",
            str(tmp_path / "rank.json"),
            "--output-md",
            str(tmp_path / "rank.md"),
        ]
    )

    assert payload["status"] == "primary_surface_selected"
    assert payload["ranked_surfaces"][0]["label"] == "clean"
    assert payload["ranked_surfaces"][0]["decision"] == "primary_ready"


def test_duplicate_reference_ids_are_rejected(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "dupe.json"
    _write(path, _target_set(clean=["a"], source_only=["a"], reference_ids=["a", "a"]))

    with pytest.raises(ValueError, match="duplicate reference_ids"):
        ranker.main(
            [
                "--target-set",
                f"dupe=path={path}",
                "--output-json",
                str(tmp_path / "rank.json"),
                "--output-md",
                str(tmp_path / "rank.md"),
            ]
        )


def test_controls_and_baselines_exclude_clean_ids(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "leak.json"
    _write(
        path,
        _target_set(
            clean=["a"],
            source_only=["a"],
            control_union=["a"],
        ),
    )

    with pytest.raises(ValueError, match="clean_source_only leaks"):
        ranker.main(
            [
                "--target-set",
                f"leak=path={path}",
                "--output-json",
                str(tmp_path / "rank.json"),
                "--output-md",
                str(tmp_path / "rank.md"),
            ]
        )


def test_missing_exact_ordered_parity_blocks_promotion(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "unordered.json"
    _write(
        path,
        _target_set(
            clean=["a", "b", "c"],
            source_only=["a", "b", "c"],
            exact_ordered_id_parity=False,
        ),
    )

    payload = ranker.main(
        [
            "--target-set",
            f"unordered=path={path}",
            "--min-clean-source-only",
            "3",
            "--output-json",
            str(tmp_path / "rank.json"),
            "--output-md",
            str(tmp_path / "rank.md"),
        ]
    )

    assert payload["status"] == "no_primary_surface_selected"
    assert payload["ranked_surfaces"][0]["decision"] == "invalid"
    assert "missing_exact_ordered_id_parity" in payload["ranked_surfaces"][0]["invalid_reasons"]


def test_output_has_schema_hashes_git_and_overlap_matrix(tmp_path: pathlib.Path) -> None:
    live = tmp_path / "live.json"
    holdout = tmp_path / "holdout.json"
    _write(live, _target_set(clean=["a", "b", "c"], source_only=["a", "b", "c"], oracle=4))
    _write(holdout, _target_set(clean=["b"], source_only=["b"], oracle=2))

    payload = ranker.main(
        [
            "--target-set",
            f"live=path={live},role=live",
            "--target-set",
            f"holdout=path={holdout},role=holdout",
            "--min-clean-source-only",
            "3",
            "--output-json",
            str(tmp_path / "rank.json"),
            "--output-md",
            str(tmp_path / "rank.md"),
        ]
    )

    assert payload["schema_version"] == "source_surface_ranking.v1"
    assert payload["git_commit"]
    assert len(payload["ranked_surfaces"][0]["target_set"]["sha256"]) == 64
    overlap = {
        (row["left"], row["right"]): row["clean_source_only_overlap_ids"]
        for row in payload["overlap_matrix"]
    }
    assert overlap[("live", "holdout")] == ["b"]
