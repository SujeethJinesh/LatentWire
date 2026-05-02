from __future__ import annotations

import json

from scripts import run_toy_shared_feature_dictionary_bridge as bridge


def test_toy_shared_feature_dictionary_bridge_is_deterministic_and_well_typed() -> None:
    config = bridge.ToySharedFeatureDictionaryConfig(
        seed=19,
        train_examples=48,
        test_examples=24,
        dim=12,
        shared_features=4,
        source_private_features=3,
        target_private_features=3,
        sparsity=0.25,
        noise=0.02,
        dictionary_iters=5,
        dictionary_lam=1e-3,
        bridge_lam=1e-2,
    )

    rows = bridge.run_experiment(config)
    rows_again = bridge.run_experiment(config)
    assert rows_again == rows
    assert len(rows) == len(bridge.METHODS)
    assert [row["method"] for row in rows] == list(bridge.METHODS)

    required_keys = {
        "method",
        "seed",
        "dim",
        "train_examples",
        "test_examples",
        "shared_features",
        "source_private_features",
        "target_private_features",
        "sparsity",
        "noise",
        "train_mse",
        "test_mse",
        "train_accuracy",
        "test_accuracy",
        "shared_feature_recovery",
        "sparsity_rate",
        "dictionary_alignment_residual",
        "bytes_proxy",
        "compute_proxy",
        "accuracy_delta_vs_raw",
        "mse_delta_vs_raw",
        "help_vs_raw",
        "harm_vs_raw",
    }

    for row in rows:
        assert required_keys <= set(row)
        assert row["seed"] == 19
        assert row["dim"] == 12
        assert row["train_examples"] == 48
        assert row["test_examples"] == 24
        assert row["shared_features"] == 4
        assert row["source_private_features"] == 3
        assert row["target_private_features"] == 3
        assert 0.0 <= row["train_mse"]
        assert 0.0 <= row["test_mse"]
        assert 0.0 <= row["train_accuracy"] <= 1.0
        assert 0.0 <= row["test_accuracy"] <= 1.0
        assert 0.0 <= row["shared_feature_recovery"] <= 1.0
        assert 0.0 <= row["sparsity_rate"] <= 1.0
        assert row["bytes_proxy"] >= 0.0
        assert row["compute_proxy"] >= 0.0
        assert row["help_vs_raw"] >= 0.0
        assert row["harm_vs_raw"] >= 0.0
        assert row["help_vs_raw"] == max(0.0, row["accuracy_delta_vs_raw"])
        assert row["harm_vs_raw"] == max(0.0, -row["accuracy_delta_vs_raw"])

    raw = next(row for row in rows if row["method"] == "raw_residual_bridge")
    separate = next(row for row in rows if row["method"] == "separate_per_model_dictionaries")
    assert separate["test_accuracy"] > raw["test_accuracy"]
    assert separate["test_mse"] < raw["test_mse"]
    assert separate["help_vs_raw"] > 0.0


def test_toy_shared_feature_dictionary_bridge_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "toy_shared_feature_dictionary.json"
    markdown = tmp_path / "toy_shared_feature_dictionary.md"

    bridge.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "7",
            "--train-examples",
            "24",
            "--test-examples",
            "12",
            "--dim",
            "12",
            "--shared-features",
            "4",
            "--source-private-features",
            "3",
            "--target-private-features",
            "3",
            "--sparsity",
            "0.25",
            "--noise",
            "0.02",
            "--dictionary-iters",
            "4",
        ]
    )

    payload = json.loads(output.read_text())
    assert payload["config"]["dim"] == 12
    assert payload["config"]["shared_features"] == 4
    assert payload["methods"] == list(bridge.METHODS)
    assert len(payload["rows"]) == len(bridge.METHODS)
    assert payload["rows"][0]["method"] == "raw_residual_bridge"
    markdown_text = markdown.read_text()
    assert "Toy Shared Feature Dictionary Bridge" in markdown_text
    assert "MSE delta" in markdown_text
