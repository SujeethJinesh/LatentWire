from __future__ import annotations

import json

from scripts import run_toy_quantization_bridge as bridge


def test_toy_quantization_bridge_returns_deterministic_rows() -> None:
    config = bridge.ToyQuantizationConfig(
        seed=7,
        samples=48,
        dim=8,
        bits=4,
        protected_channels=2,
        outlier_channels=2,
        outlier_scale=6.0,
    )

    rows = bridge.run_experiment(config, ["none", "random", "hadamard"])

    assert {(row["rotation"], row["quantizer"]) for row in rows} == {
        ("none", "uniform"),
        ("none", "protected_outlier"),
        ("random", "uniform"),
        ("random", "protected_outlier"),
        ("hadamard", "uniform"),
        ("hadamard", "protected_outlier"),
    }
    for row in rows:
        assert row["method"] == f'{row["rotation"]}_{row["quantizer"]}'
        assert row["bits"] == 4
        assert row["dim"] == 8
        assert row["samples"] == 48
        assert row["protected_channels"] == 2
        assert row["mse"] >= 0.0
        assert -1.0 <= row["cosine"] <= 1.0
        assert 0.0 <= row["outlier_energy_retained"]
        assert row["bytes_estimate"] > 0.0

    none_uniform = next(row for row in rows if row["rotation"] == "none" and row["quantizer"] == "uniform")
    none_protected = next(
        row for row in rows if row["rotation"] == "none" and row["quantizer"] == "protected_outlier"
    )
    assert none_protected["bytes_estimate"] >= none_uniform["bytes_estimate"]
    assert none_protected["mse"] <= none_uniform["mse"]
    assert none_protected["outlier_energy_retained"] >= none_uniform["outlier_energy_retained"]

    rows_again = bridge.run_experiment(config, ["none", "random", "hadamard"])
    assert rows_again == rows


def test_toy_quantization_bridge_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "toy_quant.json"
    markdown = tmp_path / "toy_quant.md"

    bridge.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "5",
            "--samples",
            "16",
            "--dim",
            "8",
            "--bits",
            "4",
            "--protected-channels",
            "2",
            "--outlier-channels",
            "2",
            "--rotations",
            "none",
            "hadamard",
        ]
    )

    payload = json.loads(output.read_text())
    assert payload["config"]["dim"] == 8
    assert payload["config"]["rotations"] == ["none", "hadamard"]
    assert len(payload["rows"]) == 4
    assert payload["rows"][0]["method"] == "none_uniform"
    assert "Toy Quantization Bridge" in markdown.read_text()
