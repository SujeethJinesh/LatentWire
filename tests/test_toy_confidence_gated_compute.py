from __future__ import annotations

import json

from scripts import run_toy_confidence_gated_compute as bridge


def test_toy_confidence_gated_compute_rows_are_deterministic_and_interpretable() -> None:
    config = bridge.ToyConfidenceGatedComputeConfig(
        seed=7,
        train_examples=128,
        test_examples=64,
        dim=16,
        classes=5,
        pool_size=6,
        max_budget=4,
        probe_noise_floor=0.18,
        probe_noise_span=0.95,
        pool_noise_floor=0.12,
        pool_noise_span=1.05,
        tail_shape=1.2,
        target_avg_budget=2.0,
        budget_penalty=0.12,
        subgroup_bins=3,
    )

    payload = bridge.run_experiment(config)
    payload_again = bridge.run_experiment(config)
    assert payload == payload_again

    rows = payload["rows"]
    assert [row["method"] for row in rows] == [
        "fixed_budget_1",
        "fixed_budget_2",
        "fixed_budget_4",
        "random_budget_matched",
        "confidence_gated",
    ]

    lookup = {row["method"]: row for row in rows}
    gated = lookup["confidence_gated"]
    fixed_2 = lookup["fixed_budget_2"]
    random_matched = lookup["random_budget_matched"]
    oracle = max(rows, key=lambda row: row["oracle_accuracy"])

    for row in rows:
        assert 0.0 <= row["accuracy"] <= 1.0
        assert row["avg_budget"] > 0.0
        assert 0.0 <= row["compute_fraction"] <= 1.0
        assert 0.0 <= row["oracle_accuracy"] <= 1.0
        assert row["oracle_gap"] >= 0.0
        assert 0.0 <= row["probe_ece"] <= 1.0
        assert 0.0 <= row["selected_ece"] <= 1.0
        assert 0.0 <= row["probe_brier"] <= 1.0
        assert 0.0 <= row["selected_brier"] <= 1.0
        assert 0.0 <= row["probe_auroc"] <= 1.0
        assert 0.0 <= row["selected_auroc"] <= 1.0
        assert isinstance(row["budget_histogram"], dict)
        assert set(row["budget_histogram"]) == {"1", "2", "4"}

    assert oracle["oracle_accuracy"] >= gated["accuracy"]
    assert gated["accuracy"] >= random_matched["accuracy"]
    assert gated["accuracy"] >= fixed_2["accuracy"]
    assert abs(gated["avg_budget"] - config.target_avg_budget) <= 1.0
    assert gated["oracle_gap"] <= lookup["fixed_budget_1"]["oracle_gap"]

    subgroups = payload["subgroups"]["confidence_gated"]
    assert set(subgroups) == {"difficulty", "probe_confidence"}
    assert all("count" in row for group in subgroups.values() for row in group)
    total_count = sum(row["count"] for row in subgroups["difficulty"])
    assert total_count == config.test_examples
    assert len(subgroups["difficulty"]) >= 2
    assert len(subgroups["probe_confidence"]) >= 2

    calibration = payload["calibration"]
    assert calibration["train_examples"] == config.train_examples
    assert 0.0 <= calibration["low_threshold"] < calibration["high_threshold"] <= 1.0
    assert 0.0 <= calibration["train_accuracy"] <= 1.0
    assert calibration["train_avg_budget"] > 0.0


def test_toy_confidence_gated_compute_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "toy_confidence.json"
    markdown = tmp_path / "toy_confidence.md"

    payload = bridge.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "11",
            "--train-examples",
            "64",
            "--test-examples",
            "32",
            "--dim",
            "12",
            "--classes",
            "4",
            "--pool-size",
            "6",
            "--max-budget",
            "4",
            "--target-avg-budget",
            "2.0",
            "--budget-penalty",
            "0.12",
            "--subgroup-bins",
            "3",
        ]
    )

    on_disk = json.loads(output.read_text())
    assert on_disk == payload
    assert on_disk["config"]["seed"] == 11
    assert on_disk["config"]["test_examples"] == 32
    assert len(on_disk["rows"]) == 5
    assert on_disk["rows"][4]["method"] == "confidence_gated"
    assert "# Toy Confidence-Gated Compute" in markdown.read_text()
