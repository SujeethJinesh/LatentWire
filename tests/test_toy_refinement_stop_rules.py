from __future__ import annotations

import json

from scripts import run_toy_refinement_stop_rules as bridge


def test_toy_refinement_stop_rules_are_deterministic_and_interpretable() -> None:
    config = bridge.ToyRefinementStopRulesConfig(
        seed=5,
        examples=72,
        dim=20,
        classes=5,
        styles=6,
        quant_bits=4,
        bridge_noise=0.18,
        source_noise=0.07,
        refinement_rate=0.44,
        confidence_threshold=0.86,
        score_drift_threshold=0.20,
        verifier_harm_margin=0.015,
    )

    payload = bridge.run_experiment(config)
    assert payload == bridge.run_experiment(config)
    assert payload["methods"] == list(bridge.METHODS)

    rows = {row["method"]: row for row in payload["rows"]}
    assert list(rows) == list(bridge.METHODS)
    fixed_1 = rows["fixed_1_step"]
    fixed_2 = rows["fixed_2_step"]
    fixed_4 = rows["fixed_4_step"]
    confidence = rows["confidence_stop"]
    drift = rows["score_drift_stop"]
    verifier = rows["verifier_harm_stop"]
    oracle = rows["oracle_stop"]

    for row in rows.values():
        assert 0.0 <= row["task_accuracy"] <= 1.0
        assert row["mse"] >= 0.0
        assert row["regression_mse"] >= 0.0
        assert 1.0 <= row["average_steps"] <= 4.0
        assert row["compute_proxy"] >= config.dim
        assert row["bytes_proxy"] > 0.0
        assert 0.0 <= row["help_rate"] <= 1.0
        assert 0.0 <= row["harm_rate"] <= 1.0
        assert 0.0 <= row["mse_help_rate"] <= 1.0
        assert 0.0 <= row["mse_harm_rate"] <= 1.0
        assert 0.0 <= row["over_refinement_rate"] <= 1.0
        assert 0.0 <= row["mean_confidence"] <= 1.0
        assert row["confidence_ece"] >= 0.0
        assert sum(row["stop_reasons"].values()) == config.examples
        assert sum(row["stop_histogram"].values()) == config.examples

    assert fixed_1["average_steps"] == 1.0
    assert fixed_2["average_steps"] == 2.0
    assert fixed_4["average_steps"] == 4.0
    assert fixed_2["mse"] < fixed_1["mse"]
    assert fixed_4["over_refinement_rate"] > fixed_2["over_refinement_rate"]
    assert confidence["average_steps"] < fixed_4["average_steps"]
    assert drift["average_steps"] < fixed_4["average_steps"]
    assert verifier["average_steps"] <= fixed_4["average_steps"]
    assert oracle["mse"] <= min(row["mse"] for row in rows.values())
    assert oracle["over_refinement_rate"] == 0.0


def test_toy_refinement_stop_rules_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "refinement_stop_rules.json"
    markdown = tmp_path / "refinement_stop_rules.md"

    payload = bridge.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "13",
            "--examples",
            "24",
            "--dim",
            "16",
            "--classes",
            "4",
            "--styles",
            "5",
        ]
    )

    on_disk = json.loads(output.read_text())
    assert on_disk == payload
    assert on_disk["config"]["seed"] == 13
    assert on_disk["config"]["examples"] == 24
    assert len(on_disk["rows"]) == len(bridge.METHODS)
    assert "# Toy Refinement Stop Rules" in markdown.read_text()
