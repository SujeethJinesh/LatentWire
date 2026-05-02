from __future__ import annotations

import json

from scripts import run_toy_protected_frontier_selection as frontier


def test_toy_protected_frontier_selection_is_deterministic_and_schema_stable() -> None:
    config = frontier.ToyProtectedFrontierSelectionConfig(
        seed=31,
        calibration_examples=96,
        test_examples=72,
        atoms=28,
        dim=18,
        classes=4,
        signal_atoms=8,
        distractor_atoms=8,
        protected_atoms=6,
        keep_fraction=0.68,
        low_bits=2,
        high_bits=8,
        signal_scale=2.3,
        distractor_scale=7.0,
        verifier_noise=0.10,
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
        "missed_help_rate",
        "false_prune_rate",
        "top_atom_preservation_rate",
        "protected_oracle_preservation_rate",
        "patch_rank_correlation",
        "protection_precision_rate",
        "help_vs_prune_uniform_quant",
        "harm_vs_prune_uniform_quant",
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
        assert 0.0 <= row["missed_help_rate"] <= 1.0
        assert 0.0 <= row["false_prune_rate"] <= 1.0
        assert 0.0 <= row["top_atom_preservation_rate"] <= 1.0
        assert 0.0 <= row["protected_oracle_preservation_rate"] <= 1.0
        assert -1.0 <= row["patch_rank_correlation"] <= 1.0
        assert 0.0 <= row["protection_precision_rate"] <= 1.0
        assert 0.0 <= row["help_vs_prune_uniform_quant"] <= 1.0
        assert 0.0 <= row["harm_vs_prune_uniform_quant"] <= 1.0

    lookup = {row["method"]: row for row in rows}
    baseline = lookup["prune_uniform_quant"]
    global_activation = lookup["global_activation_protect"]
    quant_error = lookup["quant_error_protect"]
    exact_patch = lookup["exact_patch_effect_protect"]
    random = lookup["random_protect"]
    utility_oracle = lookup["utility_oracle_protect"]

    assert baseline["protected_rate"] == 0.0
    assert baseline["accuracy_delta_vs_prune_uniform_quant"] == 0.0
    assert baseline["mse_delta_vs_prune_uniform_quant"] == 0.0
    assert global_activation["protected_rate"] > baseline["protected_rate"]
    assert quant_error["accuracy"] >= baseline["accuracy"]
    assert quant_error["mse"] <= baseline["mse"]
    assert exact_patch["accuracy"] >= baseline["accuracy"] - 1e-6
    assert exact_patch["mse"] <= baseline["mse"] + 1e-6
    assert exact_patch["patch_rank_correlation"] >= quant_error["patch_rank_correlation"] - 1e-6
    assert quant_error["protected_oracle_preservation_rate"] >= random["protected_oracle_preservation_rate"]
    assert utility_oracle["protected_oracle_preservation_rate"] == 1.0


def test_toy_protected_frontier_selection_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "protected_frontier_selection.json"
    markdown = tmp_path / "protected_frontier_selection.md"

    payload = frontier.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "31",
            "--calibration-examples",
            "64",
            "--test-examples",
            "48",
            "--atoms",
            "28",
            "--dim",
            "18",
            "--classes",
            "4",
            "--signal-atoms",
            "8",
            "--distractor-atoms",
            "8",
            "--protected-atoms",
            "6",
            "--keep-fraction",
            "0.68",
            "--low-bits",
            "2",
            "--high-bits",
            "8",
        ]
    )

    loaded = json.loads(output.read_text())
    assert loaded["config"]["seed"] == 31
    assert loaded["config"]["atoms"] == 28
    assert loaded["methods"] == list(frontier.METHODS)
    assert len(loaded["rows"]) == len(frontier.METHODS)
    assert payload["rows"][0]["method"] == "prune_uniform_quant"
    markdown_text = markdown.read_text()
    assert "# Toy Protected Frontier Selection" in markdown_text
    assert "| Method | Accuracy | Acc delta | MSE | MSE delta | Prune rate | Protected rate | Missed help | False prune | Top-atom preservation | Protected-oracle preservation | Patch-rank corr | Protection precision | Bytes proxy | Compute proxy | Help vs prune-uniform | Harm vs prune-uniform |" in markdown_text
