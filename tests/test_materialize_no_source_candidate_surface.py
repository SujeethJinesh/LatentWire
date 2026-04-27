from __future__ import annotations

import json
import pathlib

from scripts import materialize_no_source_candidate_surface as materialize


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _row(example_id: str, method: str, pred: str, correct: bool) -> dict:
    return {
        "example_id": example_id,
        "method": method,
        "prediction": f"answer: {pred}",
        "normalized_prediction": pred,
        "correct": correct,
    }


def test_materializes_expanded_no_source_candidate_surface(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "target.jsonl"
    source = tmp_path / "source.jsonl"
    text = tmp_path / "text.jsonl"
    repair = tmp_path / "repair.jsonl"
    base = tmp_path / "base.json"
    out_dir = tmp_path / "out"

    _write_jsonl(
        target,
        [
            _row("a", "target", "1", False),
            _row("b", "target", "2", True),
        ],
    )
    _write_jsonl(
        source,
        [
            _row("a", "source", "3", True),
            _row("b", "source", "2", True),
        ],
    )
    _write_jsonl(
        text,
        [
            _row("a", "t2t", "9", False),
            _row("b", "t2t", "2", True),
        ],
    )
    _write_jsonl(
        repair,
        [
            {
                **_row("a", "selected_route_no_repair", "7", False),
                "candidate_scores": [
                    {"source": "target", "normalized_prediction": "1", "correct": False},
                    {"source": "seed_0", "normalized_prediction": "3", "correct": True},
                    {"source": "seed_1", "normalized_prediction": "8", "correct": False},
                ],
            },
            {
                **_row("b", "selected_route_no_repair", "2", True),
                "candidate_scores": [
                    {"source": "target", "normalized_prediction": "2", "correct": True},
                    {"source": "seed_0", "normalized_prediction": "4", "correct": False},
                    {"source": "seed_1", "normalized_prediction": "2", "correct": True},
                ],
            },
        ],
    )
    base.write_text(
        json.dumps(
            {
                "reference_ids": ["a", "b"],
                "artifacts": {
                    "target": {"label": "target", "path": str(target), "method": "target"},
                    "source": {"label": "source", "path": str(source), "method": "source"},
                    "baselines": [{"label": "t2t", "path": str(text), "method": "t2t"}],
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = materialize.main(
        [
            "--base-target-set",
            str(base),
            "--candidate",
            f"selected=path={repair},method=selected_route_no_repair",
            "--expand-candidate-scores",
            f"zero_pool=path={repair},method=selected_route_no_repair",
            "--output-dir",
            str(out_dir),
            "--date",
            "2026-04-27",
        ]
    )

    target_set = json.loads((out_dir / "source_contrastive_target_set.json").read_text())
    assert manifest["status"] == "no_source_candidate_surface_materialized"
    assert target_set["surface_kind"] == "no_source_candidate_surface"
    assert target_set["counts"]["source_only"] == 1
    assert target_set["counts"]["clean_source_only"] == 0
    assert (out_dir / "zero_pool_seed_0.jsonl").exists()
    assert (out_dir / "zero_pool_seed_1.jsonl").exists()
    assert not (out_dir / "zero_pool_target.jsonl").exists()
    assert {item["label"] for item in manifest["candidate_summaries"]} == {
        "selected",
        "zero_pool_seed_0",
        "zero_pool_seed_1",
    }
