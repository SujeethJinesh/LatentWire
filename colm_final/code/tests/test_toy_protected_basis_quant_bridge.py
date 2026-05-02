from __future__ import annotations

import json

from scripts import run_toy_protected_basis_quant_bridge as bridge


def test_toy_protected_basis_quant_bridge_is_deterministic_and_has_expected_telemetry() -> None:
    config = bridge.ToyProtectedBasisQuantConfig(
        seed=21,
        calibration_samples=128,
        test_samples=192,
        dim=32,
        bits_uniform=4,
        low_bits=3,
        high_bits=8,
        protected_channels=2,
        mixed_high_channels=6,
        outlier_channels=4,
        outlier_scale=7.5,
        signal_scale=2.25,
        label_noise=0.15,
    )

    payload = bridge.run_experiment(config)
    payload_again = bridge.run_experiment(config)
    assert payload == payload_again

    rows = payload["rows"]
    assert [row["method"] for row in rows] == [
        "uniform_low_bit",
        "protected_salient_channels",
        "incoherent_preprocess",
        "mixed_bit_allocation",
    ]

    lookup = {row["method"]: row for row in rows}
    uniform = lookup["uniform_low_bit"]
    protected = lookup["protected_salient_channels"]
    incoherent = lookup["incoherent_preprocess"]
    mixed = lookup["mixed_bit_allocation"]

    for row in rows:
        assert 0.0 <= row["accuracy"] <= 1.0
        assert row["mse"] >= 0.0
        assert -1.0 <= row["cosine"] <= 1.0
        assert 0.0 <= row["outlier_mass"] <= 1.0
        assert row["bytes_estimate"] > 0.0
        assert 0.0 <= row["help_vs_uniform"] <= 1.0
        assert 0.0 <= row["harm_vs_uniform"] <= 1.0
        assert isinstance(row["salient_indices"], list)
        assert isinstance(row["selected_channels"], list)

    byte_band = [row["bytes_estimate"] for row in rows]
    assert max(byte_band) - min(byte_band) <= 4.5

    assert protected["accuracy"] >= uniform["accuracy"] - 1e-6
    assert mixed["accuracy"] >= uniform["accuracy"] - 1e-6
    assert incoherent["mse"] <= uniform["mse"] + 1e-6

    assert protected["help_vs_uniform"] >= 0.0
    assert mixed["help_vs_uniform"] >= 0.0
    assert incoherent["harm_vs_uniform"] <= 1.0


def test_toy_protected_basis_quant_bridge_cli_writes_jsonl_and_markdown(tmp_path) -> None:
    output = tmp_path / "protected_basis_quant_bridge.jsonl"
    markdown = tmp_path / "protected_basis_quant_bridge.md"

    payload = bridge.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "11",
            "--calibration-samples",
            "64",
            "--test-samples",
            "96",
            "--dim",
            "32",
            "--bits-uniform",
            "4",
            "--low-bits",
            "3",
            "--high-bits",
            "8",
            "--protected-channels",
            "2",
            "--mixed-high-channels",
            "6",
            "--outlier-channels",
            "4",
        ]
    )

    on_disk = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    assert len(on_disk) == 4
    assert on_disk[0]["method"] == "uniform_low_bit"
    assert payload["config"]["seed"] == 11
    assert payload["config"]["dim"] == 32
    assert "# Toy Protected-Basis Quant Bridge" in markdown.read_text()
    assert "| Method | Accuracy | MSE | Cosine | Outlier mass | Bytes estimate | Help vs uniform | Harm vs uniform |" in markdown.read_text()
