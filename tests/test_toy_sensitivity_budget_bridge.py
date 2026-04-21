from __future__ import annotations

import json

from scripts import run_toy_sensitivity_budget_bridge as bridge


def test_toy_sensitivity_budget_bridge_is_deterministic_and_uses_matched_budget() -> None:
    config = bridge.ToySensitivityBudgetConfig(
        seed=7,
        calibration_examples=48,
        test_examples=64,
        slots=10,
        channels=16,
        classes=4,
        protected_slots=2,
        protected_channels=3,
        bits=4,
        signal_scale=2.1,
        noise_scale=0.12,
        outlier_scale=6.5,
    )

    rows = bridge.run_experiment(config)
    rows_again = bridge.run_experiment(config)
    assert rows == rows_again
    assert [row["scenario"] for row in rows] == [
        "aligned",
        "aligned",
        "aligned",
        "rotated",
        "rotated",
        "rotated",
        "outlier",
        "outlier",
        "outlier",
        "slot_permuted",
        "slot_permuted",
        "slot_permuted",
    ]
    assert [row["method"] for row in rows[:3]] == [
        "uniform_allocation",
        "sensitivity_protected",
        "oracle_allocation",
    ]

    scenario_lookup: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        scenario_lookup.setdefault(row["scenario"], {})[row["method"]] = row
        assert 0.0 <= row["accuracy"] <= 1.0
        assert 0.0 <= row["uniform_accuracy"] <= 1.0
        assert -1.0 <= row["accuracy_delta_vs_uniform"] <= 1.0
        assert 0.0 <= row["help_rate_vs_uniform"] <= 1.0
        assert 0.0 <= row["harm_rate_vs_uniform"] <= 1.0
        assert 0.0 <= row["protected_fraction"] <= 1.0
        assert row["allocation_entropy"] >= 0.0
        assert 0.0 <= row["outlier_mass"] <= 1.0
        assert row["bytes_estimate"] > 0.0
        assert len(row["selected_slot_indices"]) == config.protected_slots
        assert len(row["selected_channel_indices"]) == config.protected_channels
        assert row["protected_slots"] == config.protected_slots
        assert row["protected_channels"] == config.protected_channels

    for scenario, methods in scenario_lookup.items():
        uniform = methods["uniform_allocation"]
        sensitivity = methods["sensitivity_protected"]
        oracle = methods["oracle_allocation"]
        assert uniform["bytes_estimate"] == sensitivity["bytes_estimate"] == oracle["bytes_estimate"]
        assert oracle["accuracy"] >= sensitivity["accuracy"] - 1e-6
        assert sensitivity["accuracy"] >= uniform["accuracy"] - 1e-6
        assert sensitivity["help_rate_vs_uniform"] >= 0.0
        assert sensitivity["harm_rate_vs_uniform"] >= 0.0
        assert oracle["help_rate_vs_uniform"] >= 0.0

    assert scenario_lookup["outlier"]["oracle_allocation"]["outlier_mass"] >= scenario_lookup["aligned"]["oracle_allocation"]["outlier_mass"] - 1e-6


def test_toy_sensitivity_budget_bridge_cli_writes_jsonl_and_markdown(tmp_path) -> None:
    output = tmp_path / "sensitivity_budget_bridge.jsonl"
    markdown = tmp_path / "sensitivity_budget_bridge.md"

    payload = bridge.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "11",
            "--calibration-examples",
            "24",
            "--test-examples",
            "32",
            "--slots",
            "8",
            "--channels",
            "12",
            "--protected-slots",
            "2",
            "--protected-channels",
            "3",
            "--scenarios",
            "aligned",
            "outlier",
        ]
    )

    on_disk = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    assert len(on_disk) == 6
    assert on_disk[0]["scenario"] == "aligned"
    assert on_disk[0]["method"] == "uniform_allocation"
    assert payload["config"]["seed"] == 11
    assert payload["config"]["scenarios"] == ["aligned", "outlier"]
    assert "# Toy Sensitivity Budget Bridge" in markdown.read_text()
    assert "| Scenario | Method | Accuracy | Uniform acc. | Δ acc. | Help vs uniform | Harm vs uniform | Protected fraction | Allocation entropy | Outlier mass | Bytes estimate | Selected slots | Selected channels |" in markdown.read_text()
