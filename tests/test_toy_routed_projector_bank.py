from __future__ import annotations

import json

from scripts import run_toy_routed_projector_bank as routed


def test_toy_routed_projector_bank_is_deterministic_and_schema_stable() -> None:
    config = routed.ToyRoutedProjectorBankConfig(
        seed=23,
        calibration_examples=96,
        test_examples=64,
        dim=16,
        classes=4,
        experts=4,
        styles=5,
        source_noise=0.03,
        target_noise=0.02,
        perturb_noise=0.02,
    )

    payload = routed.run_experiment(config)
    assert payload == routed.run_experiment(config)
    assert payload["methods"] == list(routed.METHODS)
    assert [row["method"] for row in payload["rows"]] == list(routed.METHODS)

    required_fields = {
        "method",
        "task_accuracy",
        "mse",
        "route_entropy",
        "expert_utilization",
        "route_stability",
        "help_vs_no_route",
        "harm_vs_no_route",
        "mse_help_vs_no_route",
        "mse_harm_vs_no_route",
        "route_accuracy",
        "bytes_proxy",
        "compute_proxy",
        "failure_tags",
    }
    for row in payload["rows"]:
        assert required_fields.issubset(row)
        assert 0.0 <= row["task_accuracy"] <= 1.0
        assert row["mse"] >= 0.0
        assert 0.0 <= row["route_entropy"] <= 2.1
        assert set(row["expert_utilization"]) == {str(idx) for idx in range(config.experts)}
        assert abs(sum(row["expert_utilization"].values()) - 1.0) < 1e-6
        assert 0.0 <= row["route_stability"] <= 1.0
        assert 0.0 <= row["help_vs_no_route"] <= 1.0
        assert 0.0 <= row["harm_vs_no_route"] <= 1.0
        assert 0.0 <= row["mse_help_vs_no_route"] <= 1.0
        assert 0.0 <= row["mse_harm_vs_no_route"] <= 1.0
        assert 0.0 <= row["route_accuracy"] <= 1.0
        assert row["bytes_proxy"] > 0.0
        assert row["compute_proxy"] > 0.0
        assert sum(row["failure_tags"].values()) == config.test_examples

    rows = {row["method"]: row for row in payload["rows"]}
    no_route = rows["no_route_baseline"]
    monolithic = rows["monolithic_projector"]
    oracle = rows["oracle_routed_bank"]
    confidence = rows["confidence_routed_bank"]
    feature = rows["feature_routed_bank"]
    random = rows["random_routed_bank"]

    assert no_route["help_vs_no_route"] == 0.0
    assert no_route["harm_vs_no_route"] == 0.0
    assert no_route["route_entropy"] == 0.0
    assert monolithic["mse"] < no_route["mse"]
    assert oracle["mse"] <= monolithic["mse"] + 1e-6
    assert oracle["route_accuracy"] == 1.0
    assert oracle["task_accuracy"] >= monolithic["task_accuracy"]
    assert feature["route_accuracy"] >= random["route_accuracy"]
    assert feature["mse"] <= random["mse"] + 1e-6
    assert confidence["task_accuracy"] >= no_route["task_accuracy"]
    assert random["route_entropy"] > 0.0


def test_toy_routed_projector_bank_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "routed_projector_bank.json"
    markdown = tmp_path / "routed_projector_bank.md"

    payload = routed.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "23",
            "--calibration-examples",
            "80",
            "--test-examples",
            "48",
            "--dim",
            "16",
            "--classes",
            "4",
            "--experts",
            "4",
            "--styles",
            "5",
        ]
    )

    loaded = json.loads(output.read_text())
    assert loaded == payload
    assert loaded["config"]["seed"] == 23
    assert loaded["methods"] == list(routed.METHODS)
    assert len(loaded["rows"]) == len(routed.METHODS)
    markdown_text = markdown.read_text()
    assert "# Toy Routed Projector Bank" in markdown_text
    assert "| Method | Accuracy | MSE | Route acc | Route entropy | Route stability | Expert utilization | Bytes proxy | Compute proxy | Help | Harm | MSE help | MSE harm |" in markdown_text
