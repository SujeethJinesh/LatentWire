from __future__ import annotations

import json

from scripts import calibrated_candidate_selector as selector


def _record(index: int, method: str, prediction: str, normalized: str, correct: bool) -> dict[str, object]:
    return {
        "index": index,
        "method": method,
        "prediction": prediction,
        "normalized_prediction": normalized,
        "correct": correct,
        "answer": [normalized if correct else "gold"],
        "generated_tokens": 20,
    }


def test_candidate_features_and_scoring_are_interpretable() -> None:
    row = {
        "candidate_source": "seed_1",
        "source_input_index": 1,
        "candidate_format_score": 2.0,
        "candidate_numeric_consistency_score": 3.0,
        "candidate_completion_score": 1.0,
        "candidate_answer_agreement": 2,
    }
    weights = {
        "format_score": 1.0,
        "numeric_consistency": 0.5,
        "completion": 1.0,
        "answer_agreement": 0.25,
        "is_target": 10.0,
        "is_seed": 1.0,
        "seed_index": -0.5,
    }

    features = selector.candidate_features(row)

    assert features["is_target"] == 0.0
    assert features["is_seed"] == 1.0
    assert features["seed_index"] == 1.0
    assert selector.candidate_score(row, weights) == 5.5


def test_calibrated_candidate_selector_records_are_split_and_logged() -> None:
    baseline = [
        _record(0, "target_alone", "target says 0", "0", False),
        _record(1, "target_alone", "target says 1", "1", False),
        _record(2, "target_alone", "target says 2", "2", False),
        _record(3, "target_alone", "target says 3", "3", False),
    ]
    seed0 = [
        _record(0, "bridge", "final answer is 0.", "0", True),
        _record(1, "bridge", "final answer is 1.", "1", True),
        _record(2, "bridge", "final answer is 2.", "2", True),
        _record(3, "bridge", "final answer is 3.", "3", True),
    ]
    seed1 = [
        _record(0, "bridge", "seed says 10", "10", False),
        _record(1, "bridge", "seed says 11", "11", False),
        _record(2, "bridge", "seed says 12", "12", False),
        _record(3, "bridge", "seed says 13", "13", False),
    ]
    records, metadata = selector.calibrated_candidate_selector_records(
        [baseline + seed0, baseline + seed1],
        method="bridge",
        calibration_fraction=0.5,
        target_selection_penalty=0.1,
    )

    assert metadata["calibration_indices"] == [0, 1]
    assert metadata["eval_indices"] == [2, 3]
    assert metadata["calibration"]["train_accuracy"] == 1.0
    assert "weights" in metadata["calibration"]

    selected = [row for row in records if row["method"] == "calibrated_feature_selector"]
    assert len(selected) == 2
    assert all(row["selector_split"] == "eval" for row in selected)
    assert all(row["selected_candidate_source"] == "seed_0" for row in selected)
    assert all("selector_candidate_feature_scores" in row for row in selected)

    results = selector.summarize_results(records)
    assert results["calibrated_feature_selector"] == 1.0
    assert results["calibrated_feature_selector_seed_selection_rate"] == 1.0


def test_calibrated_candidate_selector_cli_writes_outputs(tmp_path) -> None:
    path0 = tmp_path / "salt0.jsonl"
    path1 = tmp_path / "salt1.jsonl"
    output = tmp_path / "out.jsonl"
    markdown = tmp_path / "out.md"
    rows0 = [
        _record(0, "target_alone", "target says 0", "0", False),
        _record(0, "bridge", "final answer is 0.", "0", True),
        _record(1, "target_alone", "target says 1", "1", False),
        _record(1, "bridge", "final answer is 1.", "1", True),
    ]
    rows1 = [
        _record(0, "target_alone", "target says 0", "0", False),
        _record(0, "bridge", "seed says 10", "10", False),
        _record(1, "target_alone", "target says 1", "1", False),
        _record(1, "bridge", "seed says 11", "11", False),
    ]
    path0.write_text("\n".join(json.dumps(row) for row in rows0) + "\n")
    path1.write_text("\n".join(json.dumps(row) for row in rows1) + "\n")

    selector.main(
        [
            "--inputs",
            str(path0),
            str(path1),
            "--method",
            "bridge",
            "--calibration-fraction",
            "0.0",
            "--output-jsonl",
            str(output),
            "--output-md",
            str(markdown),
        ]
    )

    assert output.exists()
    assert output.with_suffix(".jsonl.meta.json").exists()
    assert "# Calibrated Feature Selector Summary" in markdown.read_text(encoding="utf-8")
