import json

from scripts import analyze_target_sampling_reachability as audit


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


def test_reachability_counts_c2c_clean_and_diversity(tmp_path):
    target_path = tmp_path / "target.jsonl"
    source_path = tmp_path / "source.jsonl"
    samples_path = tmp_path / "samples.jsonl"
    target_set_path = tmp_path / "target_set.json"
    c2c_path = tmp_path / "c2c.json"

    _write_jsonl(
        target_path,
        [
            _row("clean_hit", "target_alone", "0", "5"),
            _row("clean_miss", "target_alone", "0", "7"),
            _row("target_ok", "target_alone", "3", "3"),
        ],
    )
    _write_jsonl(
        source_path,
        [
            _row("clean_hit", "source_alone", "5", "5"),
            _row("clean_miss", "source_alone", "7", "7"),
            _row("target_ok", "source_alone", "0", "3"),
        ],
    )
    _write_jsonl(
        samples_path,
        [
            _row("clean_hit", "target_sample_s0", "5", "5"),
            _row("clean_hit", "target_sample_s1", "5", "5"),
            _row("clean_miss", "target_sample_s0", "8", "7"),
            _row("clean_miss", "target_sample_s1", "9", "7"),
            _row("target_ok", "target_sample_s0", "3", "3"),
        ],
    )
    target_set_path.write_text(
        json.dumps(
            {
                "artifacts": {
                    "target": {"label": "target", "path": str(target_path), "method": "target_alone"},
                    "source": {"label": "source", "path": str(source_path), "method": "source_alone"},
                    "baselines": [],
                    "controls": [],
                },
                "ids": {"clean_source_only": ["clean_hit", "clean_miss"]},
                "reference_ids": ["clean_hit", "clean_miss", "target_ok"],
            }
        ),
        encoding="utf-8",
    )
    c2c_path.write_text(
        json.dumps(
            {
                "ids": {
                    "clean_residual_targets": ["clean_hit", "clean_miss"],
                    "teacher_only": ["clean_hit", "target_ok"],
                }
            }
        ),
        encoding="utf-8",
    )

    payload = audit.main(
        [
            "--samples-jsonl",
            str(samples_path),
            "--base-target-set",
            str(target_set_path),
            "--c2c-headroom-json",
            str(c2c_path),
            "--date",
            "2026-04-27",
            "--output-json",
            str(tmp_path / "out.json"),
            "--output-md",
            str(tmp_path / "out.md"),
        ]
    )

    assert payload["sample_oracle_correct"] == 2
    assert payload["sample_oracle_gain_vs_target"] == 1
    assert payload["source_contrastive_clean_in_pool_ids"] == ["clean_hit"]
    assert payload["c2c_clean_residual_in_pool_ids"] == ["clean_hit"]
    assert payload["c2c_teacher_only_in_pool_ids"] == ["clean_hit", "target_ok"]
    assert payload["diversity"]["unique_answer_counts_by_id"]["clean_hit"] == 1
    assert payload["diversity"]["duplicate_nonempty_rows"] == 1
