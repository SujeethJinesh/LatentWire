from __future__ import annotations

import json

from scripts import run_toy_verified_mixed_precision_stack as stack


def test_toy_verified_mixed_precision_stack_is_deterministic_and_schema_stable() -> None:
    config = stack.ToyVerifiedMixedPrecisionStackConfig(
        seed=23,
        calibration_examples=96,
        test_examples=72,
        atoms=20,
        dim=12,
        signal_atoms=5,
        harmful_atoms=5,
        protected_atoms=5,
        keep_fraction=0.56,
        low_bits=3,
        high_bits=8,
        signal_scale=3.4,
        harmful_scale=7.8,
        activation_spike=1.2,
        verifier_noise=0.12,
        calibration_noise=0.05,
        cost_jitter=0.10,
    )

    payload = stack.run_experiment(config)
    payload_again = stack.run_experiment(config)
    assert payload == payload_again

    rows = payload["rows"]
    assert [row["method"] for row in rows] == list(stack.METHODS)

    required_fields = {
        "method",
        "accuracy",
        "mse",
        "prune_rate",
        "kept_rate",
        "protected_rate",
        "bytes_proxy",
        "compute_proxy",
        "missed_help_rate",
        "false_prune_rate",
        "top_atom_preservation_rate",
        "protected_atom_preservation_rate",
        "help_vs_full_precision",
        "harm_vs_full_precision",
        "accuracy_delta_vs_full_precision",
        "mse_delta_vs_full_precision",
        "kept_atoms",
        "protected_atoms",
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
        assert 0.0 <= row["protected_atom_preservation_rate"] <= 1.0
        assert 0.0 <= row["help_vs_full_precision"] <= 1.0
        assert 0.0 <= row["harm_vs_full_precision"] <= 1.0

    lookup = {row["method"]: row for row in rows}
    full = lookup["full_precision"]
    uniform = lookup["uniform_low_bit"]
    activation = lookup["activation_aware_quant_only"]
    prune_only = lookup["verifier_prune_only"]
    prune_uniform = lookup["prune_then_uniform_quant"]
    prune_activation = lookup["prune_then_activation_aware_quant"]
    oracle = lookup["oracle_stack"]

    assert full["protected_rate"] == 1.0
    assert full["prune_rate"] == 0.0
    assert uniform["protected_rate"] == 0.0
    assert prune_only["prune_rate"] > 0.0
    assert prune_only["protected_rate"] > 0.0
    assert activation["protected_rate"] > uniform["protected_rate"]
    assert prune_uniform["bytes_proxy"] <= prune_only["bytes_proxy"]
    assert prune_activation["bytes_proxy"] <= prune_only["bytes_proxy"]
    assert prune_activation["protected_rate"] >= prune_uniform["protected_rate"]
    assert oracle["accuracy"] >= prune_activation["accuracy"] - 1e-6
    assert oracle["mse"] <= prune_activation["mse"] + 1e-6
    assert oracle["top_atom_preservation_rate"] >= prune_only["top_atom_preservation_rate"] - 1e-6
    assert oracle["protected_atom_preservation_rate"] >= activation["protected_atom_preservation_rate"] - 1e-6


def test_toy_verified_mixed_precision_stack_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "verified_mixed_precision_stack.json"
    markdown = tmp_path / "verified_mixed_precision_stack.md"

    payload = stack.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "23",
            "--calibration-examples",
            "48",
            "--test-examples",
            "32",
            "--atoms",
            "20",
            "--dim",
            "12",
            "--protected-atoms",
            "5",
            "--keep-fraction",
            "0.56",
            "--low-bits",
            "3",
            "--high-bits",
            "8",
        ]
    )

    loaded = json.loads(output.read_text())
    assert loaded["config"]["seed"] == 23
    assert loaded["config"]["atoms"] == 20
    assert loaded["methods"] == list(stack.METHODS)
    assert len(loaded["rows"]) == len(stack.METHODS)
    assert payload["rows"][0]["method"] == "full_precision"
    markdown_text = markdown.read_text()
    assert "# Toy Verified Mixed-Precision Stack" in markdown_text
    assert "| Method | Accuracy | MSE | Prune rate | Kept rate | Protected rate | Missed help | False prune | Top-atom preservation | Protected-top preservation | Bytes proxy | Compute proxy | Help vs full | Harm vs full |" in markdown_text
