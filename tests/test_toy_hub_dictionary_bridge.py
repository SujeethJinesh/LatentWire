from __future__ import annotations

import json

from scripts import run_toy_hub_dictionary_bridge as hub


def test_toy_hub_dictionary_bridge_is_deterministic_and_schema_stable() -> None:
    config = hub.ToyHubDictionaryBridgeConfig(
        seed=17,
        calibration_examples=180,
        test_examples=150,
        dim=16,
        atoms=8,
        classes=4,
        families=5,
        heldout_family=4,
        source_noise=0.03,
        target_noise=0.025,
    )

    payload = hub.run_experiment(config)
    assert payload == hub.run_experiment(config)
    assert payload["methods"] == list(hub.METHODS)
    assert [row["method"] for row in payload["rows"]] == list(hub.METHODS)
    assert payload["seen_pair_count"] == (config.families - 1) * (config.families - 2)
    assert payload["all_ordered_pair_count"] == config.families * (config.families - 1)
    assert payload["hub_adapter_count"] == 2 * config.families

    required_fields = {
        "method",
        "eval_examples",
        "task_accuracy",
        "mse",
        "atom_recovery",
        "hub_residual",
        "pairwise_residual",
        "heldout_fraction",
        "pair_seen_fraction",
        "adapter_count",
        "parameter_proxy",
        "bytes_proxy",
        "compute_proxy",
        "help_vs_monolithic",
        "harm_vs_monolithic",
        "mse_help_vs_monolithic",
        "mse_harm_vs_monolithic",
        "failure_tags",
    }
    for row in payload["rows"]:
        assert required_fields.issubset(row)
        assert row["eval_examples"] > 0
        assert 0.0 <= row["task_accuracy"] <= 1.0
        assert 0.0 <= row["atom_recovery"] <= 1.0
        assert row["mse"] >= 0.0
        assert row["hub_residual"] >= 0.0
        assert row["pairwise_residual"] >= 0.0
        assert 0.0 <= row["heldout_fraction"] <= 1.0
        assert 0.0 <= row["pair_seen_fraction"] <= 1.0
        assert row["adapter_count"] >= 0
        assert row["parameter_proxy"] >= 0.0
        assert row["bytes_proxy"] >= 0.0
        assert row["compute_proxy"] >= 0.0
        assert 0.0 <= row["help_vs_monolithic"] <= 1.0
        assert 0.0 <= row["harm_vs_monolithic"] <= 1.0
        assert 0.0 <= row["mse_help_vs_monolithic"] <= 1.0
        assert 0.0 <= row["mse_harm_vs_monolithic"] <= 1.0
        assert sum(row["failure_tags"].values()) == row["eval_examples"]

    rows = {row["method"]: row for row in payload["rows"]}
    monolithic = rows["monolithic_bridge"]
    pairwise = rows["pairwise_bridges"]
    hub_row = rows["hub_shared_dictionary"]
    heldout = rows["held_out_family_transfer"]
    random_hub = rows["random_hub"]
    oracle = rows["oracle"]

    assert monolithic["help_vs_monolithic"] == 0.0
    assert monolithic["harm_vs_monolithic"] == 0.0
    assert pairwise["adapter_count"] > hub_row["adapter_count"]
    assert pairwise["parameter_proxy"] > hub_row["parameter_proxy"]
    assert pairwise["pair_seen_fraction"] < 1.0
    assert pairwise["failure_tags"]["heldout_pair_missing"] > 0
    assert hub_row["adapter_count"] == 2 * config.families
    assert hub_row["task_accuracy"] >= pairwise["task_accuracy"]
    assert hub_row["atom_recovery"] >= pairwise["atom_recovery"]
    assert hub_row["hub_residual"] < pairwise["hub_residual"]
    assert heldout["heldout_fraction"] == 1.0
    assert heldout["task_accuracy"] >= pairwise["task_accuracy"]
    assert random_hub["task_accuracy"] < hub_row["task_accuracy"]
    assert oracle["mse"] == 0.0
    assert oracle["hub_residual"] <= hub_row["hub_residual"]


def test_toy_hub_dictionary_bridge_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "hub_dictionary_bridge.json"
    markdown = tmp_path / "hub_dictionary_bridge.md"

    payload = hub.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "17",
            "--calibration-examples",
            "150",
            "--test-examples",
            "120",
            "--dim",
            "16",
            "--atoms",
            "8",
            "--classes",
            "4",
            "--families",
            "5",
            "--heldout-family",
            "4",
        ]
    )

    loaded = json.loads(output.read_text())
    assert loaded == payload
    assert loaded["config"]["seed"] == 17
    assert loaded["methods"] == list(hub.METHODS)
    assert len(loaded["rows"]) == len(hub.METHODS)
    markdown_text = markdown.read_text()
    assert "# Toy Hub Dictionary Bridge" in markdown_text
    assert "| Method | Examples | Accuracy | MSE | Atom recovery | Hub residual | Pairwise residual |" in markdown_text
    assert "https://arxiv.org/abs/2602.15382" in markdown_text
