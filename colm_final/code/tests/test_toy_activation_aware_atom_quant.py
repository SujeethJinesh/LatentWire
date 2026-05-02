from __future__ import annotations

import json

from scripts import run_toy_activation_aware_atom_quant as bridge


def test_toy_activation_aware_atom_quant_is_deterministic_and_schema_stable() -> None:
    config = bridge.ToyActivationAwareAtomQuantConfig(
        seed=19,
        calibration_samples=160,
        test_samples=192,
        atoms=32,
        signal_atoms=6,
        outlier_atoms=4,
        protected_atoms=8,
        low_bits=3,
        high_bits=8,
        signal_scale=3.5,
        outlier_scale=9.0,
        label_noise=0.12,
        activation_spike=1.4,
    )

    payload = bridge.run_experiment(config)
    payload_again = bridge.run_experiment(config)
    assert payload == payload_again

    rows = payload["rows"]
    assert [row["method"] for row in rows] == [
        "full_precision",
        "uniform_low_bit",
        "random_mixed_precision",
        "activation_aware_mixed_precision",
        "protected_outlier_mixed_precision",
        "oracle_mixed_precision",
    ]

    required_fields = {
        "method",
        "accuracy",
        "mse",
        "cosine",
        "bit_budget",
        "bytes_proxy",
        "protected_rate",
        "outlier_protected_rate",
        "top_atom_preservation_rate",
        "help_vs_full_precision",
        "harm_vs_full_precision",
        "selected_atoms",
        "oracle_atoms",
        "outlier_atom_indices",
    }
    for row in rows:
        assert required_fields.issubset(row)
        assert 0.0 <= row["accuracy"] <= 1.0
        assert row["mse"] >= 0.0
        assert -1.0 <= row["cosine"] <= 1.0
        assert row["bit_budget"] > 0.0
        assert row["bytes_proxy"] > 0.0
        assert 0.0 <= row["protected_rate"] <= 1.0
        assert 0.0 <= row["outlier_protected_rate"] <= 1.0
        assert 0.0 <= row["top_atom_preservation_rate"] <= 1.0
        assert 0.0 <= row["help_vs_full_precision"] <= 1.0
        assert 0.0 <= row["harm_vs_full_precision"] <= 1.0
        assert isinstance(row["selected_atoms"], list)
        assert isinstance(row["oracle_atoms"], list)
        assert isinstance(row["outlier_atom_indices"], list)

    full = rows[0]
    uniform = rows[1]
    random = rows[2]
    activation = rows[3]
    protected = rows[4]
    oracle = rows[5]

    assert full["protected_rate"] == 1.0
    assert full["top_atom_preservation_rate"] == 1.0
    assert full["outlier_protected_rate"] == 1.0
    assert uniform["top_atom_preservation_rate"] == 0.0
    assert random["bit_budget"] == activation["bit_budget"] == protected["bit_budget"] == oracle["bit_budget"]

    assert activation["accuracy"] >= random["accuracy"] - 1e-6
    assert protected["accuracy"] >= random["accuracy"] - 1e-6
    assert oracle["accuracy"] >= activation["accuracy"] - 1e-6
    assert oracle["accuracy"] >= protected["accuracy"] - 1e-6
    assert activation["top_atom_preservation_rate"] >= random["top_atom_preservation_rate"]
    assert protected["outlier_protected_rate"] >= random["outlier_protected_rate"]
    assert oracle["top_atom_preservation_rate"] >= activation["top_atom_preservation_rate"]
    assert oracle["top_atom_preservation_rate"] >= protected["top_atom_preservation_rate"]
    assert uniform["mse"] >= full["mse"]
    assert random["mse"] >= activation["mse"] - 1e-6 or random["mse"] >= protected["mse"] - 1e-6


def test_toy_activation_aware_atom_quant_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "activation_aware_atom_quant.json"
    markdown = tmp_path / "activation_aware_atom_quant.md"

    payload = bridge.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "19",
            "--calibration-samples",
            "96",
            "--test-samples",
            "96",
            "--atoms",
            "32",
            "--signal-atoms",
            "6",
            "--outlier-atoms",
            "4",
            "--protected-atoms",
            "8",
            "--low-bits",
            "3",
            "--high-bits",
            "8",
            "--signal-scale",
            "3.5",
            "--outlier-scale",
            "9.0",
            "--label-noise",
            "0.12",
        ]
    )

    loaded = json.loads(output.read_text())
    assert loaded["config"]["seed"] == 19
    assert loaded["config"]["atoms"] == 32
    assert len(loaded["rows"]) == 6
    assert payload["rows"][0]["method"] == "full_precision"
    markdown_text = markdown.read_text()
    assert "# Toy Activation-Aware Atom Quant" in markdown_text
    assert "| Method | Acc | MSE | Cosine | Bit budget | Bytes proxy | Protected rate | Outlier protected | Top-atom preservation | Help vs full | Harm vs full |" in markdown_text
