from __future__ import annotations

import json

from scripts import run_toy_feature_atom_stack_bridge as bridge


def test_toy_feature_atom_stack_bridge_is_deterministic_and_interpretable() -> None:
    config = bridge.ToyFeatureAtomStackBridgeConfig(
        seed=17,
        train_examples=72,
        test_examples=48,
        dim=16,
        shared_features=6,
        route_families=4,
        atoms_per_family=4,
        source_private_features=4,
        target_private_features=4,
        shared_sparsity=0.35,
        private_sparsity=0.20,
        shared_scale=0.9,
        atom_scale=2.6,
        private_scale=0.45,
        noise=0.02,
        shared_iters=6,
        atom_iters=6,
        bridge_lam=1e-2,
        dictionary_lam=1e-3,
        codebook_temp=0.35,
        protected_shared=2,
        protected_atoms=3,
        classes=5,
    )

    rows = bridge.run_experiment(config)
    rows_again = bridge.run_experiment(config)
    assert rows == rows_again
    assert [row["method"] for row in rows] == list(bridge.METHODS)

    required_keys = {
        "method",
        "seed",
        "dim",
        "train_examples",
        "test_examples",
        "shared_features",
        "route_families",
        "atoms_per_family",
        "source_private_features",
        "target_private_features",
        "train_mse",
        "test_mse",
        "train_accuracy",
        "test_accuracy",
        "shared_feature_recovery",
        "atom_recovery",
        "shared_entropy",
        "shared_perplexity",
        "atom_entropy",
        "atom_perplexity",
        "bytes_proxy",
        "compute_proxy",
        "accuracy_delta_vs_raw",
        "mse_delta_vs_raw",
        "help_vs_raw",
        "harm_vs_raw",
    }

    for row in rows:
        assert required_keys <= set(row)
        assert row["seed"] == 17
        assert row["dim"] == 16
        assert row["train_examples"] == 72
        assert row["test_examples"] == 48
        assert row["shared_features"] == 6
        assert row["route_families"] == 4
        assert row["atoms_per_family"] == 4
        assert 0.0 <= row["train_mse"]
        assert 0.0 <= row["test_mse"]
        assert 0.0 <= row["train_accuracy"] <= 1.0
        assert 0.0 <= row["test_accuracy"] <= 1.0
        assert 0.0 <= row["shared_feature_recovery"] <= 1.0
        assert 0.0 <= row["atom_recovery"] <= 1.0
        assert row["shared_entropy"] >= 0.0
        assert row["shared_perplexity"] >= 1.0
        assert row["atom_entropy"] >= 0.0
        assert row["atom_perplexity"] >= 1.0
        assert row["bytes_proxy"] >= 0.0
        assert row["compute_proxy"] >= 0.0
        assert row["help_vs_raw"] >= 0.0
        assert row["harm_vs_raw"] >= 0.0
        assert row["help_vs_raw"] == max(0.0, row["accuracy_delta_vs_raw"])
        assert row["harm_vs_raw"] == max(0.0, -row["accuracy_delta_vs_raw"])

    raw = next(row for row in rows if row["method"] == "raw_ridge")
    shared = next(row for row in rows if row["method"] == "shared_feature_only")
    route = next(row for row in rows if row["method"] == "route_atom_only")
    stacked = next(row for row in rows if row["method"] == "stacked_feature_atom")
    protected = next(row for row in rows if row["method"] == "protected_stacked_feature_atom")
    oracle = next(row for row in rows if row["method"] == "oracle")

    assert shared["shared_feature_recovery"] >= raw["shared_feature_recovery"]
    assert route["atom_recovery"] >= raw["atom_recovery"]
    assert stacked["shared_feature_recovery"] >= shared["shared_feature_recovery"] - 1e-6
    assert stacked["test_accuracy"] > raw["test_accuracy"]
    assert stacked["test_accuracy"] > shared["test_accuracy"]
    assert stacked["test_accuracy"] > route["test_accuracy"]
    assert stacked["test_mse"] < shared["test_mse"]
    assert stacked["test_mse"] < route["test_mse"]
    assert protected["bytes_proxy"] > stacked["bytes_proxy"]
    assert protected["test_accuracy"] >= stacked["test_accuracy"] - 1e-6
    assert oracle["test_accuracy"] == 1.0
    assert oracle["test_mse"] == 0.0
    assert oracle["shared_feature_recovery"] == 1.0
    assert oracle["atom_recovery"] == 1.0
    assert oracle["help_vs_raw"] >= stacked["help_vs_raw"]


def test_toy_feature_atom_stack_bridge_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "feature_atom_stack_bridge.json"
    markdown = tmp_path / "feature_atom_stack_bridge.md"

    payload = bridge.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "7",
            "--train-examples",
            "48",
            "--test-examples",
            "24",
            "--dim",
            "16",
            "--shared-features",
            "6",
            "--route-families",
            "4",
            "--atoms-per-family",
            "4",
            "--protected-shared",
            "2",
            "--protected-atoms",
            "2",
        ]
    )

    on_disk = json.loads(output.read_text())
    assert on_disk["config"]["seed"] == 7
    assert on_disk["methods"] == list(bridge.METHODS)
    assert len(on_disk["rows"]) == len(bridge.METHODS)
    assert payload["rows"][0]["method"] == "raw_ridge"
    assert "Toy Feature Atom Stack Bridge" in markdown.read_text()
