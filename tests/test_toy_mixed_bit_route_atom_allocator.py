from __future__ import annotations

import json

from scripts import run_toy_mixed_bit_route_atom_allocator as allocator


def test_toy_mixed_bit_route_atom_allocator_is_deterministic_and_schema_stable() -> None:
    config = allocator.ToyMixedBitRouteAtomAllocatorConfig(
        seed=19,
        calibration_examples=56,
        test_examples=48,
        atoms=32,
        dim=20,
        classes=4,
        universal_features=18,
        signal_atoms=8,
        outlier_atoms=10,
        low_bits=3,
        mid_bits=4,
        high_bits=8,
        target_bpw=4.0,
    )

    payload = allocator.run_experiment(config)
    payload_again = allocator.run_experiment(config)
    assert payload == payload_again

    rows = payload["rows"]
    assert payload["methods"] == list(allocator.METHODS)
    assert [row["method"] for row in rows] == list(allocator.METHODS)

    required_fields = {
        "method",
        "accuracy",
        "accuracy_delta_vs_uniform_3_bit",
        "mse",
        "mse_delta_vs_uniform_3_bit",
        "achieved_bpw",
        "target_bpw",
        "bit_allocation_histogram",
        "high_bit_atoms",
        "bytes_proxy",
        "compute_proxy",
        "patch_rank_correlation",
        "exact_patch_overlap",
        "feature_persistence_overlap",
        "mean_feature_persistence",
        "mean_quant_error",
        "outlier_protection",
        "help_vs_uniform_3_bit",
        "harm_vs_uniform_3_bit",
        "stability",
    }
    for row in rows:
        assert required_fields.issubset(row)
        assert 0.0 <= row["accuracy"] <= 1.0
        assert row["mse"] >= 0.0
        assert row["achieved_bpw"] > 0.0
        assert isinstance(row["bit_allocation_histogram"], dict)
        assert sum(row["bit_allocation_histogram"].values()) == config.atoms
        assert row["bytes_proxy"] > 0.0
        assert row["compute_proxy"] > 0.0
        assert -1.0 <= row["patch_rank_correlation"] <= 1.0
        assert 0.0 <= row["exact_patch_overlap"] <= 1.0
        assert 0.0 <= row["feature_persistence_overlap"] <= 1.0
        assert 0.0 <= row["mean_feature_persistence"] <= 1.0
        assert row["mean_quant_error"] >= 0.0
        assert 0.0 <= row["outlier_protection"] <= 1.0
        assert 0.0 <= row["help_vs_uniform_3_bit"] <= 1.0
        assert 0.0 <= row["harm_vs_uniform_3_bit"] <= 1.0
        assert 0.0 <= row["stability"] <= 1.0

    lookup = {row["method"]: row for row in rows}
    uniform3 = lookup["uniform_3_bit"]
    uniform4 = lookup["uniform_4_bit"]
    quant = lookup["quant_error_target_bpw_allocator"]
    exact = lookup["exact_patch_target_bpw_allocator"]
    persistence = lookup["universal_feature_persistence_allocator"]
    random = lookup["random_allocator"]
    oracle = lookup["oracle_allocator"]

    assert uniform3["bit_allocation_histogram"] == {"3": config.atoms}
    assert uniform4["bit_allocation_histogram"] == {"4": config.atoms}
    assert uniform3["accuracy_delta_vs_uniform_3_bit"] == 0.0
    assert uniform3["mse_delta_vs_uniform_3_bit"] == 0.0
    assert quant["achieved_bpw"] <= uniform4["achieved_bpw"]
    assert exact["patch_rank_correlation"] >= quant["patch_rank_correlation"] - 1e-6
    assert quant["outlier_protection"] >= random["outlier_protection"]
    assert persistence["feature_persistence_overlap"] >= random["feature_persistence_overlap"]
    assert oracle["mse"] <= random["mse"] + 1e-6
    assert uniform4["accuracy"] >= uniform3["accuracy"]
    assert quant["accuracy"] >= uniform3["accuracy"]


def test_toy_mixed_bit_route_atom_allocator_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "mixed_bit_route_atom_allocator.json"
    markdown = tmp_path / "mixed_bit_route_atom_allocator.md"

    payload = allocator.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "19",
            "--calibration-examples",
            "48",
            "--test-examples",
            "40",
            "--atoms",
            "32",
            "--dim",
            "20",
            "--classes",
            "4",
            "--universal-features",
            "18",
            "--signal-atoms",
            "8",
            "--outlier-atoms",
            "10",
            "--low-bits",
            "3",
            "--mid-bits",
            "4",
            "--high-bits",
            "8",
            "--target-bpw",
            "4.0",
        ]
    )

    loaded = json.loads(output.read_text())
    assert loaded["config"]["seed"] == 19
    assert loaded["methods"] == list(allocator.METHODS)
    assert len(loaded["rows"]) == len(allocator.METHODS)
    assert payload["rows"][0]["method"] == "uniform_3_bit"
    markdown_text = markdown.read_text()
    assert "# Toy Mixed-Bit Route-Atom Allocator" in markdown_text
    assert "| Method | Accuracy | Acc delta | MSE | MSE delta | Achieved bpw | Bit histogram | Patch-rank corr | Outlier protection | Exact overlap | Feature overlap | Stability | Bytes proxy | Compute proxy | Help vs 3-bit | Harm vs 3-bit |" in markdown_text
