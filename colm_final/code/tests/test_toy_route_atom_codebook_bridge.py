from __future__ import annotations

import json

from scripts import run_toy_route_atom_codebook_bridge as bridge


def test_toy_route_atom_codebook_bridge_is_deterministic_and_interpretable() -> None:
    config = bridge.ToyRouteAtomCodebookBridgeConfig(
        seed=13,
        train_examples=96,
        test_examples=64,
        dim=16,
        route_families=4,
        atoms_per_family=4,
        source_private_atoms=4,
        target_private_atoms=4,
        route_coeff_scale=1.7,
        private_coeff_scale=0.55,
        outlier_scale=2.5,
        noise=0.05,
        dictionary_iters=8,
        codebook_temp=0.35,
        ridge_lam=1e-3,
        protected_atoms=3,
    )

    rows = bridge.run_experiment(config)
    rows_again = bridge.run_experiment(config)
    assert rows == rows_again
    assert [row["method"] for row in rows] == list(bridge.METHODS)

    raw = rows[0]
    learned = next(row for row in rows if row["method"] == "learned_shared_codebook")
    route = next(row for row in rows if row["method"] == "route_conditioned_codebook")
    protected = next(row for row in rows if row["method"] == "protected_outlier_atoms")
    oracle = next(row for row in rows if row["method"] == "oracle")

    for row in rows:
        assert 0.0 <= row["accuracy"] <= 1.0
        assert row["mse"] >= 0.0
        assert 0.0 <= row["atom_recovery"] <= 1.0
        assert row["codebook_entropy"] >= 0.0
        assert row["codebook_perplexity"] >= 1.0
        assert row["bytes_proxy"] > 0.0
        assert row["compute_proxy"] > 0.0
        assert row["help_vs_raw"] >= 0.0
        assert row["harm_vs_raw"] >= 0.0
        assert row["accuracy_delta_vs_raw"] == row["help_vs_raw"] - row["harm_vs_raw"]

    assert learned["atom_recovery"] >= raw["atom_recovery"]
    assert learned["atom_recovery"] >= next(row for row in rows if row["method"] == "uniform_codebook_quantization")["atom_recovery"]
    assert route["compute_proxy"] < learned["compute_proxy"]
    assert protected["bytes_proxy"] > learned["bytes_proxy"]
    assert protected["accuracy"] >= raw["accuracy"] - 1e-6
    assert oracle["accuracy"] == 1.0
    assert oracle["mse"] == 0.0
    assert oracle["atom_recovery"] == 1.0
    assert oracle["help_vs_raw"] >= max(
        learned["help_vs_raw"], route["help_vs_raw"], protected["help_vs_raw"]
    )


def test_toy_route_atom_codebook_bridge_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "route_atom_codebook_bridge.json"
    markdown = tmp_path / "route_atom_codebook_bridge.md"

    payload = bridge.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "7",
            "--train-examples",
            "64",
            "--test-examples",
            "32",
            "--dim",
            "16",
            "--route-families",
            "4",
            "--atoms-per-family",
            "4",
            "--protected-atoms",
            "2",
        ]
    )

    on_disk = json.loads(output.read_text())
    assert on_disk["config"]["seed"] == 7
    assert on_disk["methods"] == list(bridge.METHODS)
    assert len(on_disk["rows"]) == len(bridge.METHODS)
    assert payload["rows"][0]["method"] == "raw_ridge"
    assert "Toy Route Atom Codebook Bridge" in markdown.read_text()
