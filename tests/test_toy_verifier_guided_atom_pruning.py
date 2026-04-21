from __future__ import annotations

import json

from scripts import run_toy_verifier_guided_atom_pruning as pruning


def test_toy_verifier_guided_atom_pruning_is_deterministic_and_interpretable() -> None:
    config = pruning.ToyVerifierGuidedAtomPruningConfig(
        seed=11,
        train_examples=64,
        test_examples=48,
        dim=12,
        atoms=15,
        steps=3,
        helpful_atoms=7,
        harmful_atoms=5,
        keep_fraction=0.55,
        activation_temperature=0.8,
        scalar_noise=0.25,
        step_noise=0.15,
        verifier_noise=0.08,
        cost_jitter=0.10,
        step_bonus_scale=0.30,
        verifier_bonus_scale=0.50,
    )

    rows = pruning.run_experiment(config)
    rows_again = pruning.run_experiment(config)
    assert rows == rows_again
    assert [row["method"] for row in rows] == list(pruning.METHODS)

    lookup = {row["method"]: row for row in rows}
    no_pruning = lookup["no_pruning"]
    scalar = lookup["scalar_score_pruning"]
    step_local = lookup["step_error_localized_pruning"]
    verifier = lookup["verifier_guided_frontier_pruning"]
    oracle = lookup["oracle_pruning"]

    for row in rows:
        assert 0.0 <= row["accuracy"] <= 1.0
        assert row["mse"] >= 0.0
        assert 0.0 <= row["prune_rate"] <= 1.0
        assert 0.0 <= row["missed_help_rate"] <= 1.0
        assert 0.0 <= row["false_prune_rate"] <= 1.0
        assert 0.0 <= row["atom_recovery"] <= 1.0
        assert row["bytes_proxy"] > 0.0
        assert row["compute_proxy"] > 0.0
        assert "help_vs_no_pruning" in row
        assert "harm_vs_no_pruning" in row
        assert row["accuracy_delta_vs_no_pruning"] == row["help_vs_no_pruning"] - row["harm_vs_no_pruning"]

    assert no_pruning["prune_rate"] == 0.0
    assert oracle["accuracy"] >= no_pruning["accuracy"] - 1e-6
    assert oracle["mse"] <= no_pruning["mse"] + 1e-6
    assert scalar["prune_rate"] > 0.0
    assert step_local["prune_rate"] > 0.0
    assert verifier["prune_rate"] > 0.0


def test_toy_verifier_guided_atom_pruning_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "verifier_guided_atom_pruning.json"
    markdown = tmp_path / "verifier_guided_atom_pruning.md"

    payload = pruning.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "5",
            "--train-examples",
            "32",
            "--test-examples",
            "16",
            "--dim",
            "10",
            "--atoms",
            "12",
            "--steps",
            "3",
            "--keep-fraction",
            "0.5",
        ]
    )

    on_disk = json.loads(output.read_text())
    assert on_disk["config"]["seed"] == 5
    assert on_disk["methods"] == list(pruning.METHODS)
    assert len(on_disk["rows"]) == len(pruning.METHODS)
    assert payload["rows"][0]["method"] == "no_pruning"

    md = markdown.read_text()
    assert "# Toy Verifier-Guided Atom Pruning" in md
    assert "| Method | Accuracy | MSE | Prune rate | Missed help | False prune | Atom recovery | Bytes proxy | Compute proxy | Help vs no-pruning | Harm vs no-pruning |" in md
