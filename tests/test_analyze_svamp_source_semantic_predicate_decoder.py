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
        _row("a", "target_alone", "Target considers 12 but says 7", "7", "12"),
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
    assert "random_sidecar" in payload["surfaces"][0]["summaries"]


def test_hash_shuffle_and_random_sidecar_are_nonself_controls(tmp_path):
    target_rows = [
        _row("a", "target_alone", "Target considers 12 but says 7", "7", "12"),
        _row("b", "target_alone", "Target says 5", "5", "5"),
        _row("c", "target_alone", "Target says 8", "8", "8"),
    ]
    source_rows = [
        _row("a", "source_alone", "We compute 4 x 3 = 12. Answer: 12", "12", "12"),
        _row("b", "source_alone", "We compute 2 + 3 = 5. Answer: 5", "5", "5"),
        _row("c", "source_alone", "We compute 4 + 4 = 8. Answer: 8", "8", "8"),
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
        "ids": {"clean_source_only": ["a"], "target_self_repair": ["b", "c"]},
        "reference_ids": ["a", "b", "c"],
    }
    target_set_path = tmp_path / "target_set.json"
    target_set_path.write_text(json.dumps(target_set), encoding="utf-8")
    surface = decoder._load_surface("toy", target_set_path)
    weights = {"__prior__": 0.0, "candidate_eq_source_final": 10.0}
    rule = {"min_score": -1.0, "min_margin": -1.0}

    shuffled = decoder._decode_example(
        surface=surface,
        index=0,
        condition="shuffled_source",
        weights=weights,
        rule=rule,
    )
    random_sidecar = decoder._decode_example(
        surface=surface,
        index=0,
        condition="random_sidecar",
        weights=weights,
        rule=rule,
    )

    assert shuffled["condition_source_example_id"] != "a"
    assert shuffled["condition_source_final"] in {"5", "8"}
    assert shuffled["source_control_source_answers_overlap_target"] is False
    assert random_sidecar["condition_source_example_id"] is None
    assert random_sidecar["source_quality"] is True
    assert random_sidecar["sidecar_bytes"] >= 1


def test_target_only_candidate_pool_excludes_source_only_values(tmp_path):
    target_rows = [
        _row("a", "target_alone", "Target says 7", "7", "12"),
        _row("b", "target_alone", "Target says 5", "5", "5"),
    ]
    source_rows = [
        _row("a", "source_alone", "source final 12", "12", "12"),
        _row("b", "source_alone", "source final 5", "5", "5"),
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
        "ids": {"clean_source_only": ["a"], "target_self_repair": ["b"]},
        "reference_ids": ["a", "b"],
    }
    target_set_path = tmp_path / "target_set.json"
    target_set_path.write_text(json.dumps(target_set), encoding="utf-8")
    surface = decoder._load_surface("toy", target_set_path)

    pool = decoder._candidate_pool(surface, "a")
    source_exposed_pool = decoder._candidate_pool(surface, "a", include_source=True)

    assert "12" not in {candidate.value for candidate in pool}
    assert "12" in {candidate.value for candidate in source_exposed_pool}


def test_sidecar_loader_rejects_duplicate_and_missing_ids(tmp_path):
    target_rows = [
        _row("a", "target_alone", "Target says 7", "7", "7"),
        _row("b", "target_alone", "Target says 5", "5", "5"),
    ]
    source_rows = [
        _row("a", "source_alone", "source final 7", "7", "7"),
        _row("b", "source_alone", "source final 5", "5", "5"),
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
        "ids": {"clean_source_only": [], "target_self_repair": ["a", "b"]},
        "reference_ids": ["a", "b"],
    }
    target_set_path = tmp_path / "target_set.json"
    target_set_path.write_text(json.dumps(target_set), encoding="utf-8")

    duplicate_sidecar = tmp_path / "duplicate_sidecar.jsonl"
    _write_jsonl(
        duplicate_sidecar,
        [
            {"example_id": "a", "candidate_scores": [{"label": "target", "score": 1.0}]},
            {"example_id": "a", "candidate_scores": [{"label": "target", "score": 0.0}]},
        ],
    )
    try:
        decoder._load_surface("toy", target_set_path, sidecar_path=duplicate_sidecar)
    except ValueError as exc:
        assert "Duplicate sidecar row" in str(exc)
    else:
        raise AssertionError("duplicate sidecar IDs should fail")

    missing_sidecar = tmp_path / "missing_sidecar.jsonl"
    _write_jsonl(
        missing_sidecar,
        [{"example_id": "a", "candidate_scores": [{"label": "target", "score": 1.0}]}],
    )
    try:
        decoder._load_surface("toy", target_set_path, sidecar_path=missing_sidecar)
    except ValueError as exc:
        assert "Sidecar IDs do not match" in str(exc)
    else:
        raise AssertionError("missing sidecar IDs should fail")


def test_candidate_score_sidecar_can_drive_target_safe_recovery(tmp_path):
    target_rows = [
        _row("a", "target_alone", "Target considers 12 but says 7", "7", "12"),
        _row("b", "target_alone", "Target says 5", "5", "5"),
        _row("c", "target_alone", "Target says 8", "8", "8"),
        _row("d", "target_alone", "Target says 9", "9", "9"),
    ]
    source_rows = [
        _row("a", "source_alone", "source final 12", "12", "12"),
        _row("b", "source_alone", "source final 5", "5", "5"),
        _row("c", "source_alone", "source final 8", "8", "8"),
        _row("d", "source_alone", "source final 9", "9", "9"),
    ]
    sidecar_rows = [
        {
            "example_id": "a",
            "candidate_scores": [
                {"label": "target", "value": "7", "score": 0.0},
                {"label": "target", "value": "12", "score": 3.0},
            ],
            "confidence": 3.0,
            "sidecar_bits": 32,
        }
    ]
    sidecar_rows.extend(
        {
            "example_id": example_id,
            "candidate_scores": [
                {"label": "target", "value": target_rows[ord(example_id) - ord("a")]["answer"][0], "score": 3.0},
                {"label": "source", "score": 0.0},
            ],
            "confidence": 3.0,
            "sidecar_bits": 32,
        }
        for example_id in ("b", "c", "d")
    )
    _write_jsonl(tmp_path / "target.jsonl", target_rows)
    _write_jsonl(tmp_path / "source.jsonl", source_rows)
    _write_jsonl(tmp_path / "sidecar.jsonl", sidecar_rows)
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
            "--live-sidecar-jsonl",
            str(tmp_path / "sidecar.jsonl"),
            "--holdout-sidecar-jsonl",
            str(tmp_path / "sidecar.jsonl"),
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
    matched = payload["surfaces"][0]["summaries"]["matched"]
    random_sidecar = payload["surfaces"][0]["summaries"]["random_sidecar"]
    zero = payload["surfaces"][0]["summaries"]["zero_source"]
    assert payload["status"] == "semantic_predicate_decoder_passes_smoke"
    assert matched["clean_source_necessary_count"] == 1
    assert matched["accepted_harm_count"] == 0
    assert matched["mean_sidecar_bytes"] == 4
    assert random_sidecar["mean_sidecar_bytes"] == 4
    assert zero["clean_source_necessary_count"] == 0
