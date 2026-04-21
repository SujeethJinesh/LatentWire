from __future__ import annotations

import json

from scripts import run_toy_router_stability_regularization as router


def test_toy_router_stability_regularization_is_deterministic_and_interpretable() -> None:
    config = router.ToyRouterStabilityRegularizationConfig(
        seed=19,
        calibration_examples=112,
        test_examples=72,
        dim=16,
        classes=4,
        experts=4,
        styles=5,
        source_noise=0.03,
        target_noise=0.02,
        route_noise=0.10,
        perturb_noise=0.55,
        dense_temperature=0.75,
        confidence_temperature=0.72,
        load_balance_strength=0.84,
        sticky_margin=0.04,
    )

    payload = router.run_experiment(config)
    assert payload == router.run_experiment(config)
    assert payload["methods"] == list(router.METHODS)
    assert [row["method"] for row in payload["rows"]] == list(router.METHODS)

    required_fields = {
        "method",
        "task_accuracy",
        "mse",
        "route_accuracy",
        "route_entropy",
        "gate_entropy",
        "load_balance",
        "collapse_rate",
        "perturbation_stability",
        "expert_utilization",
        "bytes_proxy",
        "compute_proxy",
        "help_rate",
        "harm_rate",
        "mse_help_rate",
        "mse_harm_rate",
        "failure_tags",
    }
    for row in payload["rows"]:
        assert required_fields.issubset(row)
        assert 0.0 <= row["task_accuracy"] <= 1.0
        assert row["mse"] >= 0.0
        assert 0.0 <= row["route_accuracy"] <= 1.0
        assert 0.0 <= row["route_entropy"] <= 2.1
        assert 0.0 <= row["gate_entropy"] <= 2.1
        assert 0.0 <= row["load_balance"] <= 1.0
        assert 0.0 <= row["collapse_rate"] <= 1.0
        assert 0.0 <= row["perturbation_stability"] <= 1.0
        assert set(row["expert_utilization"]) == {str(idx) for idx in range(config.experts)}
        assert abs(sum(row["expert_utilization"].values()) - 1.0) < 1e-6
        assert row["bytes_proxy"] > 0.0
        assert row["compute_proxy"] > 0.0
        assert 0.0 <= row["help_rate"] <= 1.0
        assert 0.0 <= row["harm_rate"] <= 1.0
        assert 0.0 <= row["mse_help_rate"] <= 1.0
        assert 0.0 <= row["mse_harm_rate"] <= 1.0
        assert sum(row["failure_tags"].values()) == config.test_examples

    rows = {row["method"]: row for row in payload["rows"]}
    hard = rows["hard_feature_routing"]
    confidence = rows["confidence_routing"]
    dense = rows["smoothed_dense_routing"]
    balanced = rows["load_balanced_routing"]
    sticky = rows["sticky_paraphrase_stable_routing"]
    random = rows["random_routing"]
    oracle = rows["oracle_routing"]

    assert hard["help_rate"] == 0.0
    assert hard["harm_rate"] == 0.0
    assert oracle["route_accuracy"] == 1.0
    assert oracle["perturbation_stability"] == 1.0
    assert oracle["mse"] <= hard["mse"] + 1e-6
    assert hard["route_accuracy"] >= random["route_accuracy"]
    assert hard["task_accuracy"] >= random["task_accuracy"]
    assert dense["gate_entropy"] > hard["gate_entropy"]
    assert dense["compute_proxy"] > hard["compute_proxy"]
    assert balanced["load_balance"] >= hard["load_balance"]
    assert balanced["collapse_rate"] <= hard["collapse_rate"] + 1e-6
    assert sticky["perturbation_stability"] > hard["perturbation_stability"]
    assert confidence["collapse_rate"] >= hard["collapse_rate"]


def test_toy_router_stability_regularization_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "router_stability_regularization.json"
    markdown = tmp_path / "router_stability_regularization.md"

    payload = router.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "7",
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
    assert loaded["config"]["seed"] == 7
    assert loaded["methods"] == list(router.METHODS)
    assert len(loaded["rows"]) == len(router.METHODS)
    markdown_text = markdown.read_text()
    assert "# Toy Router Stability Regularization" in markdown_text
    assert "| Method | Accuracy | MSE | Route acc | Route entropy | Gate entropy | Load balance | Collapse | Perturb stable | Utilization | Bytes | Compute | Help | Harm | MSE help | MSE harm |" in markdown_text
