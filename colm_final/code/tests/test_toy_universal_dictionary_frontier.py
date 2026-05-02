from __future__ import annotations

import json

from scripts import run_toy_universal_dictionary_frontier as frontier


def test_toy_universal_dictionary_frontier_is_deterministic_and_schema_stable() -> None:
    config = frontier.ToyUniversalDictionaryFrontierConfig(
        seed=17,
        calibration_examples=72,
        test_examples=56,
        atoms=30,
        dim=20,
        classes=4,
        universal_features=18,
        signal_atoms=8,
        distractor_atoms=10,
        protected_atoms=6,
        keep_fraction=0.70,
        low_bits=2,
        high_bits=8,
    )

    payload = frontier.run_experiment(config)
    payload_again = frontier.run_experiment(config)
    assert payload == payload_again

    rows = payload["rows"]
    assert payload["methods"] == list(frontier.METHODS)
    assert [row["method"] for row in rows] == list(frontier.METHODS)

    required_fields = {
        "method",
        "accuracy",
        "accuracy_delta_vs_prune_uniform_quant",
        "mse",
        "mse_delta_vs_prune_uniform_quant",
        "prune_rate",
        "kept_rate",
        "protected_rate",
        "kept_atoms",
        "protected_atoms",
        "bytes_proxy",
        "compute_proxy",
        "feature_overlap_persistence",
        "feature_overlap_lift",
        "top_persistent_feature_preservation",
        "protected_oracle_preservation_rate",
        "patch_rank_correlation",
        "protection_precision_rate",
        "missed_help_rate",
        "false_prune_rate",
        "help_vs_prune_uniform_quant",
        "harm_vs_prune_uniform_quant",
        "selector_stability",
    }
    for row in rows:
        assert required_fields.issubset(row)
        assert 0.0 <= row["accuracy"] <= 1.0
        assert row["mse"] >= 0.0
        assert 0.0 <= row["prune_rate"] <= 1.0
        assert 0.0 <= row["kept_rate"] <= 1.0
        assert 0.0 <= row["protected_rate"] <= 1.0
        assert row["bytes_proxy"] > 0.0
        assert row["compute_proxy"] > 0.0
        assert 0.0 <= row["feature_overlap_persistence"] <= 1.0
        assert -1.0 <= row["feature_overlap_lift"] <= 1.0
        assert 0.0 <= row["top_persistent_feature_preservation"] <= 1.0
        assert 0.0 <= row["protected_oracle_preservation_rate"] <= 1.0
        assert -1.0 <= row["patch_rank_correlation"] <= 1.0
        assert 0.0 <= row["protection_precision_rate"] <= 1.0
        assert 0.0 <= row["missed_help_rate"] <= 1.0
        assert 0.0 <= row["false_prune_rate"] <= 1.0
        assert 0.0 <= row["help_vs_prune_uniform_quant"] <= 1.0
        assert 0.0 <= row["harm_vs_prune_uniform_quant"] <= 1.0
        assert 0.0 <= row["selector_stability"] <= 1.0

    lookup = {row["method"]: row for row in rows}
    baseline = lookup["prune_uniform_quant"]
    raw = lookup["raw_activation_protect"]
    quant = lookup["quant_error_protect"]
    exact = lookup["exact_patch_effect_protect"]
    dictionary = lookup["universal_dictionary_persistence_protect"]
    random = lookup["random_protect"]
    utility = lookup["utility_oracle_protect"]

    assert baseline["protected_rate"] == 0.0
    assert baseline["accuracy_delta_vs_prune_uniform_quant"] == 0.0
    assert baseline["mse_delta_vs_prune_uniform_quant"] == 0.0
    assert exact["patch_rank_correlation"] >= quant["patch_rank_correlation"] - 1e-6
    assert exact["mse"] <= baseline["mse"] + 1e-6
    assert dictionary["feature_overlap_persistence"] > raw["feature_overlap_persistence"]
    assert dictionary["selector_stability"] >= random["selector_stability"]
    assert dictionary["protected_oracle_preservation_rate"] >= random["protected_oracle_preservation_rate"]
    assert dictionary["accuracy"] >= baseline["accuracy"]
    assert dictionary["mse"] <= baseline["mse"] + 1e-6
    assert quant["mse"] <= baseline["mse"] + 1e-6
    assert utility["feature_overlap_persistence"] >= random["feature_overlap_persistence"]


def test_toy_universal_dictionary_frontier_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "universal_dictionary_frontier.json"
    markdown = tmp_path / "universal_dictionary_frontier.md"

    payload = frontier.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "17",
            "--calibration-examples",
            "48",
            "--test-examples",
            "40",
            "--atoms",
            "30",
            "--dim",
            "20",
            "--classes",
            "4",
            "--universal-features",
            "18",
            "--signal-atoms",
            "8",
            "--distractor-atoms",
            "10",
            "--protected-atoms",
            "6",
            "--keep-fraction",
            "0.70",
            "--low-bits",
            "2",
            "--high-bits",
            "8",
        ]
    )

    loaded = json.loads(output.read_text())
    assert loaded["config"]["seed"] == 17
    assert loaded["config"]["atoms"] == 30
    assert loaded["methods"] == list(frontier.METHODS)
    assert len(loaded["rows"]) == len(frontier.METHODS)
    assert payload["rows"][0]["method"] == "prune_uniform_quant"
    markdown_text = markdown.read_text()
    assert "# Toy Universal Dictionary Frontier" in markdown_text
    assert "| Method | Accuracy | Acc delta | MSE | MSE delta | Feature persistence | Patch-rank corr | Selector stability | Protected-oracle preservation | Bytes proxy | Compute proxy | Help vs prune-uniform | Harm vs prune-uniform |" in markdown_text
