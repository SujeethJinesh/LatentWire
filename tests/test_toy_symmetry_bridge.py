from __future__ import annotations

import json

from scripts import run_toy_symmetry_bridge as bridge


def test_toy_symmetry_bridge_returns_deterministic_rows() -> None:
    config = bridge.ToySymmetryConfig(
        seed=11,
        train_examples=24,
        test_examples=12,
        dim=8,
        noise=0.02,
        ridge_lam=1e-2,
        nonlinear_strength=0.15,
    )

    rows = bridge.run_experiment(config, bridge.SCENARIOS)
    assert len(rows) == len(bridge.SCENARIOS) * len(bridge.METHODS)
    assert {(row["scenario"], row["method"]) for row in rows} == {
        (scenario, method) for scenario in bridge.SCENARIOS for method in bridge.METHODS
    }

    for row in rows:
        assert row["seed"] == 11
        assert row["dim"] == 8
        assert row["train_examples"] == 24
        assert row["test_examples"] == 12
        assert row["train_mse"] >= 0.0
        assert row["test_mse"] >= 0.0
        assert -1.0 <= row["train_cosine"] <= 1.0
        assert -1.0 <= row["test_cosine"] <= 1.0
        assert 0.0 <= row["train_pair_recall_at_1"] <= 1.0
        assert 0.0 <= row["test_pair_recall_at_1"] <= 1.0
        if row["method"] in {"permutation_only", "permutation_plus_procrustes"}:
            if row["scenario"] in {"permutation", "permutation_rotation"}:
                assert row["permutation_accuracy"] is not None
                assert row["permutation_exact_match"] in {True, False}
            else:
                assert row["permutation_accuracy"] is None
                assert row["permutation_exact_match"] is None
        else:
            assert row["permutation_accuracy"] is None
            assert row["permutation_exact_match"] is None

    rows_again = bridge.run_experiment(config, bridge.SCENARIOS)
    assert rows_again == rows

    permutation_row = next(
        row for row in rows if row["scenario"] == "permutation" and row["method"] == "permutation_only"
    )
    assert permutation_row["permutation_exact_match"] is True
    assert permutation_row["test_pair_recall_at_1"] == 1.0

    rotation_identity = next(
        row for row in rows if row["scenario"] == "orthogonal_rotation" and row["method"] == "identity"
    )
    rotation_procrustes = next(
        row for row in rows if row["scenario"] == "orthogonal_rotation" and row["method"] == "orthogonal_procrustes"
    )
    assert rotation_procrustes["test_mse"] <= rotation_identity["test_mse"]


def test_toy_symmetry_bridge_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "toy_symmetry.json"
    markdown = tmp_path / "toy_symmetry.md"

    bridge.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "5",
            "--train-examples",
            "16",
            "--test-examples",
            "8",
            "--dim",
            "8",
            "--noise",
            "0.01",
            "--ridge-lam",
            "0.01",
            "--nonlinear-strength",
            "0.12",
            "--scenarios",
            "identity",
            "permutation",
        ]
    )

    payload = json.loads(output.read_text())
    assert payload["config"]["dim"] == 8
    assert payload["scenarios"] == ["identity", "permutation"]
    assert payload["methods"] == list(bridge.METHODS)
    assert len(payload["rows"]) == 2 * len(bridge.METHODS)
    assert payload["rows"][0]["scenario"] == "identity"
    assert "Toy Symmetry Bridge" in markdown.read_text()
