import json

from scripts import analyze_target_side_candidate_headroom as audit


def _row(example_id: str, method: str, prediction: str, answer: str) -> dict:
    return {
        "answer": [answer],
        "correct": prediction == answer,
        "example_id": example_id,
        "method": method,
        "normalized_prediction": prediction,
        "prediction": prediction,
    }


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_audit_counts_clean_gold_only_from_target_side_pool(tmp_path):
    target_path = tmp_path / "target.jsonl"
    source_path = tmp_path / "source.jsonl"
    text_path = tmp_path / "text.jsonl"
    target_set_path = tmp_path / "target_set.json"

    _write_jsonl(
        target_path,
        [
            _row("clean_in_pool", "target_alone", "0", "5"),
            _row("clean_source_only", "target_alone", "0", "7"),
            _row("target_ok", "target_alone", "3", "3"),
        ],
    )
    _write_jsonl(
        source_path,
        [
            _row("clean_in_pool", "source_alone", "5", "5"),
            _row("clean_source_only", "source_alone", "7", "7"),
            _row("target_ok", "source_alone", "0", "3"),
        ],
    )
    _write_jsonl(
        text_path,
        [
            _row("clean_in_pool", "text_to_text", "5", "5"),
            _row("clean_source_only", "text_to_text", "2", "7"),
            _row("target_ok", "text_to_text", "4", "3"),
        ],
    )
    target_set_path.write_text(
        json.dumps(
            {
                "artifacts": {
                    "target": {"path": str(target_path), "method": "target_alone"},
                    "source": {"path": str(source_path), "method": "source_alone"},
                    "baselines": [{"label": "t2t", "path": str(text_path), "method": "text_to_text"}],
                    "controls": [],
                },
                "ids": {
                    "clean_source_only": ["clean_in_pool", "clean_source_only"],
                    "target_self_repair": ["target_ok"],
                },
                "reference_ids": ["clean_in_pool", "clean_source_only", "target_ok"],
            }
        ),
        encoding="utf-8",
    )

    payload = audit.main(
        [
            "--target-set",
            f"toy=path={target_set_path},role=test,note=unit",
            "--date",
            "2026-04-27",
            "--output-json",
            str(tmp_path / "out.json"),
            "--output-md",
            str(tmp_path / "out.md"),
        ]
    )

    surface = payload["surfaces"][0]
    assert surface["target_correct"] == 1
    assert surface["source_correct"] == 2
    assert surface["target_side_oracle_correct"] == 2
    assert surface["target_side_oracle_gain"] == 1
    assert surface["clean_gold_in_target_side_pool_ids"] == ["clean_in_pool"]
    assert "clean_source_only" not in surface["clean_gold_in_target_side_pool_ids"]
