from __future__ import annotations

import json

from scripts import confidence_gated_route_expansion as gated


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


def _target(index: int, correct: bool, *, strong: bool) -> dict[str, object]:
    if strong:
        return _record(index, "target_alone", f"The final answer is {index}.", str(index), correct)
    return _record(index, "target_alone", f"Working trails off near {index}", str(index), correct)


def test_choose_seed_budget_uses_target_proxy_thresholds() -> None:
    assert gated.choose_seed_budget(
        target_proxy=9.0,
        low_threshold=2.0,
        high_threshold=5.0,
        medium_budget=1,
        max_budget=3,
    ) == 0
    assert gated.choose_seed_budget(
        target_proxy=3.0,
        low_threshold=2.0,
        high_threshold=5.0,
        medium_budget=1,
        max_budget=3,
    ) == 1
    assert gated.choose_seed_budget(
        target_proxy=1.0,
        low_threshold=2.0,
        high_threshold=5.0,
        medium_budget=1,
        max_budget=3,
    ) == 3


def test_confidence_gated_route_records_are_deterministic_and_interpretable() -> None:
    baseline = [
        _target(0, True, strong=True),
        _target(1, False, strong=False),
        _target(2, False, strong=True),
        _target(3, False, strong=False),
    ]
    seed0 = [
        _record(0, "bridge", "seed0 says 10", "10", False),
        _record(1, "bridge", "final answer is 1.", "1", True),
        _record(2, "bridge", "seed0 says 20", "20", False),
        _record(3, "bridge", "seed0 says 30", "30", False),
    ]
    seed1 = [
        _record(0, "bridge", "seed1 says 11", "11", False),
        _record(1, "bridge", "seed1 says 12", "12", False),
        _record(2, "bridge", "final answer is 2.", "2", True),
        _record(3, "bridge", "seed1 says 31", "31", False),
    ]
    seed2 = [
        _record(0, "bridge", "seed2 says 12", "12", False),
        _record(1, "bridge", "seed2 says 13", "13", False),
        _record(2, "bridge", "seed2 says 22", "22", False),
        _record(3, "bridge", "final answer is 3.", "3", True),
    ]
    record_sets = [baseline + seed0, baseline + seed1, baseline + seed2]

    records, metadata = gated.confidence_gated_route_records(
        record_sets,
        method="bridge",
        selection_policy="format_then_agreement",
        calibration_fraction=0.5,
        medium_budget=1,
        target_avg_seed_budget=1.5,
        budget_penalty=0.1,
        random_seed=17,
    )
    records_again, metadata_again = gated.confidence_gated_route_records(
        record_sets,
        method="bridge",
        selection_policy="format_then_agreement",
        calibration_fraction=0.5,
        medium_budget=1,
        target_avg_seed_budget=1.5,
        budget_penalty=0.1,
        random_seed=17,
    )
    assert records == records_again
    assert metadata == metadata_again

    methods = {row["method"] for row in records}
    assert methods == {
        "target_alone",
        "fixed_route_budget_0",
        "fixed_route_budget_1",
        "fixed_route_budget_3",
        "random_route_budget_matched",
        "confidence_gated_route_expansion",
    }
    assert metadata["calibration_indices"] == [0, 1]
    assert metadata["eval_indices"] == [2, 3]
    assert metadata["max_seed_budget"] == 3

    gated_rows = [row for row in records if row["method"] == "confidence_gated_route_expansion"]
    assert len(gated_rows) == 2
    for row in gated_rows:
        assert row["selector_split"] == "eval"
        assert row["selector_seed_budget"] in {0, 1, 3}
        assert row["selector_subset_candidate_count"] == row["selector_seed_budget"] + 1
        assert isinstance(row["selector_full_oracle_correct"], bool)
        assert row["selector_available_seed_count"] == 3
        assert "candidate_scores" in row

    results = gated.summarize_records(records)
    assert 0.0 <= results["confidence_gated_route_expansion"] <= 1.0
    assert "confidence_gated_route_expansion_avg_seed_budget" in results
    assert "confidence_gated_route_expansion_subgroups" in results


def test_confidence_gated_route_cli_writes_outputs(tmp_path) -> None:
    path0 = tmp_path / "salt0.jsonl"
    path1 = tmp_path / "salt1.jsonl"
    output = tmp_path / "out.jsonl"
    markdown = tmp_path / "out.md"

    rows0 = [
        _target(0, False, strong=False),
        _record(0, "bridge", "final answer is 0.", "0", True),
        _target(1, True, strong=True),
        _record(1, "bridge", "seed says 10", "10", False),
    ]
    rows1 = [
        _target(0, False, strong=False),
        _record(0, "bridge", "seed says 11", "11", False),
        _target(1, True, strong=True),
        _record(1, "bridge", "seed says 12", "12", False),
    ]
    path0.write_text("\n".join(json.dumps(row) for row in rows0) + "\n")
    path1.write_text("\n".join(json.dumps(row) for row in rows1) + "\n")

    argv = [
        "--inputs",
        str(path0),
        str(path1),
        "--method",
        "bridge",
        "--selection-policy",
        "format_then_agreement",
        "--calibration-fraction",
        "0.0",
        "--output-jsonl",
        str(output),
        "--output-md",
        str(markdown),
    ]
    gated.main(argv)

    written = [json.loads(line) for line in output.read_text().splitlines()]
    assert written
    assert output.with_suffix(".jsonl.meta.json").exists()
    assert "# Confidence-Gated Route Expansion Summary" in markdown.read_text()
