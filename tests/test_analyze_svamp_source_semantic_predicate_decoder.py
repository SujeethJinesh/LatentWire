import json

from scripts import analyze_svamp_source_semantic_predicate_decoder as decoder


def _row(example_id: str, method: str, prediction: str, normalized: str, answer: str) -> dict:
    return {
        "answer": [answer],
        "correct": normalized == answer,
        "example_id": example_id,
        "method": method,
        "normalized_prediction": normalized,
        "prediction": prediction,
    }


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_semantic_predicate_features_capture_verified_equation():
    profile = decoder._source_profile(
        _row("a", "source", "We compute 4 x 3 = 12. Answer: 12", "12", "12")
    )
    candidate = decoder.Candidate(value="12", labels=("source",))
    features = decoder._features_for_candidate(candidate, profile, fallback="7")
    assert "candidate_eq_source_final" in features
    assert "candidate_in_verified_equation" in features
    assert "op_mul" in features


def test_decode_abstains_without_source_quality(tmp_path):
    target_set = {
        "artifacts": {
            "target": {"path": str(tmp_path / "target.jsonl"), "method": "target_alone"},
            "source": {"path": str(tmp_path / "source.jsonl"), "method": "source_alone"},
            "baselines": [],
            "controls": [],
        },
        "ids": {"clean_source_only": ["a"], "target_self_repair": []},
        "reference_ids": ["a"],
    }
    _write_jsonl(tmp_path / "target.jsonl", [_row("a", "target_alone", "Target says 7", "7", "12")])
    _write_jsonl(tmp_path / "source.jsonl", [_row("a", "source_alone", "maybe twelve", "12", "12")])
    target_set_path = tmp_path / "target_set.json"
    target_set_path.write_text(json.dumps(target_set), encoding="utf-8")
    surface = decoder._load_surface("toy", target_set_path)
    weights = {"__prior__": 0.0, "candidate_eq_source_final": 10.0}
    row = decoder._decode_example(
        surface=surface,
        index=0,
        condition="matched",
        weights=weights,
        rule={"min_score": -1.0, "min_margin": -1.0},
    )
    assert row["prediction"] == "7"
    assert row["accepted_source_sidecar"] is False


def test_synthetic_surface_can_recover_source_with_controls(tmp_path):
    target_rows = [
        _row("a", "target_alone", "Target says 7", "7", "12"),
        _row("b", "target_alone", "Target says 5", "5", "5"),
        _row("c", "target_alone", "Target says 8", "8", "8"),
        _row("d", "target_alone", "Target says 9", "9", "9"),
    ]
    source_rows = [
        _row("a", "source_alone", "We compute 4 x 3 = 12. Answer: 12", "12", "12"),
        _row("b", "source_alone", "We compute 2 + 3 = 5. Answer: 5", "5", "5"),
        _row("c", "source_alone", "We compute 4 + 4 = 8. Answer: 8", "8", "8"),
        _row("d", "source_alone", "We compute 10 - 1 = 9. Answer: 9", "9", "9"),
    ]
    _write_jsonl(tmp_path / "target.jsonl", target_rows)
    _write_jsonl(tmp_path / "source.jsonl", source_rows)
    target_set = {
        "artifacts": {
            "target": {"path": str(tmp_path / "target.jsonl"), "method": "target_alone"},
            "source": {"path": str(tmp_path / "source.jsonl"), "method": "source_alone"},
            "baselines": [],
            "controls": [],
        },
        "ids": {"clean_source_only": ["a"], "target_self_repair": ["b", "c", "d"]},
        "reference_ids": ["a", "b", "c", "d"],
    }
    target_set_path = tmp_path / "target_set.json"
    target_set_path.write_text(json.dumps(target_set), encoding="utf-8")
    payload = decoder.main(
        [
            "--live-target-set",
            str(target_set_path),
            "--holdout-target-set",
            str(target_set_path),
            "--outer-folds",
            "1",
            "--min-live-correct",
            "4",
            "--min-live-clean-source-necessary",
            "1",
            "--min-holdout-correct",
            "4",
            "--min-holdout-clean-source-necessary",
            "1",
            "--max-accepted-harm",
            "0",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert payload["surfaces"][0]["summaries"]["matched"]["clean_source_necessary_count"] >= 1
    assert payload["surfaces"][0]["summaries"]["matched"]["accepted_harm_count"] == 0
