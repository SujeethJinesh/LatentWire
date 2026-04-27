import json

from scripts import build_candidate_pool_decision_surface as builder


def test_build_surface_appends_extra_candidates_and_overrides_clean_ids(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(
        json.dumps(
            {
                "artifacts": {
                    "target": {"label": "target", "path": "target.jsonl", "method": "target"},
                    "source": {"label": "source", "path": "source.jsonl", "method": "source"},
                    "baselines": [],
                    "controls": [],
                },
                "counts": {"clean_source_only": 0},
                "ids": {"clean_source_only": [], "clean_residual_targets": [], "source_only": []},
                "reference_ids": ["a", "b"],
                "reference_n": 2,
                "rows": [{"example_id": "a", "labels": []}, {"example_id": "b", "labels": []}],
            }
        ),
        encoding="utf-8",
    )

    payload = builder.main(
        [
            "--base-target-set",
            str(base),
            "--extra-candidate",
            f"label=source_sample_s0,path={tmp_path / 'samples.jsonl'},method=target_sample_s0",
            "--clean-id",
            "b",
            "--date",
            "2026-04-27",
            "--output-json",
            str(tmp_path / "surface.json"),
            "--output-md",
            str(tmp_path / "surface.md"),
        ]
    )

    assert payload["status"] == "candidate_pool_decision_surface_ready"
    assert payload["ids"]["clean_source_only"] == ["b"]
    assert payload["ids"]["clean_residual_targets"] == ["b"]
    assert payload["artifacts"]["baselines"][0]["label"] == "source_sample_s0"
    assert payload["artifacts"]["baselines"][0]["method"] == "target_sample_s0"
    assert payload["rows"][1]["labels"] == ["clean_source_only", "source_only"]
