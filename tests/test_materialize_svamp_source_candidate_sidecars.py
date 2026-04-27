import json

from scripts import materialize_svamp_source_candidate_sidecars as materializer


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


def test_materializer_scores_only_target_side_candidates(tmp_path):
    target_rows = [
        _row("a", "target_alone", "Target considers 12 but says 7", "7", "12"),
        _row("b", "target_alone", "Target says 5", "5", "5"),
    ]
    source_rows = [
        _row("a", "source_alone", "We compute 4 x 3 = 12. Answer: 12", "12", "12"),
        _row("b", "source_alone", "We compute 2 + 3 = 5. Answer: 5", "5", "5"),
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

    payload = materializer.main(
        [
            "--live-target-set",
            str(target_set_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--sidecar-bits",
            "16",
            "--date",
            "2026-04-27",
        ]
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "live_candidate_sidecars.jsonl").read_text().splitlines()
    ]
    by_id = {row["example_id"]: row for row in rows}

    assert payload["summaries"]["live"]["source_final_in_pool"] == 2
    assert by_id["a"]["sidecar_bits"] == 16
    assert {item["value"] for item in by_id["a"]["candidate_scores"]} == {"7", "12"}
    assert "source" not in {item["label"] for item in by_id["a"]["candidate_scores"]}
    assert by_id["a"]["candidate_scores"][0]["value"] == "12"


def test_materializer_does_not_add_source_only_answer(tmp_path):
    target_rows = [_row("a", "target_alone", "Target says 7", "7", "12")]
    source_rows = [_row("a", "source_alone", "Answer: 12", "12", "12")]
    _write_jsonl(tmp_path / "target.jsonl", target_rows)
    _write_jsonl(tmp_path / "source.jsonl", source_rows)
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
    target_set_path = tmp_path / "target_set.json"
    target_set_path.write_text(json.dumps(target_set), encoding="utf-8")

    materializer.main(
        [
            "--live-target-set",
            str(target_set_path),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    row = json.loads((tmp_path / "out" / "live_candidate_sidecars.jsonl").read_text())

    assert {item["value"] for item in row["candidate_scores"]} == {"7"}
    assert row["source_final_in_target_pool"] is False
