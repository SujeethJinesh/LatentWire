from __future__ import annotations

import json

from scripts import run_toy_iterative_latent_refinement as bridge


def test_toy_iterative_latent_refinement_is_deterministic_and_interpretable() -> None:
    config = bridge.ToyIterativeLatentRefinementConfig(
        seed=7,
        examples=72,
        dim=20,
        classes=5,
        styles=6,
        quant_bits=4,
        bridge_noise=0.18,
        source_noise=0.07,
        refinement_rate=0.44,
        gate_threshold=0.68,
        diffusion_steps=3,
        diffusion_noise=0.06,
        oracle_fraction=0.35,
    )

    payload = bridge.run_experiment(config)
    assert payload == bridge.run_experiment(config)
    assert payload["methods"] == list(bridge.METHODS)

    rows = {row["method"]: row for row in payload["rows"]}
    assert list(rows) == list(bridge.METHODS)
    one_pass = rows["one_pass_bridge"]
    fixed_2 = rows["fixed_2_step_refinement"]
    fixed_4 = rows["fixed_4_step_refinement"]
    gated = rows["confidence_gated_refinement"]
    diffusion = rows["noisy_diffusion_refinement"]
    oracle = rows["oracle_refinement"]

    for row in rows.values():
        assert 0.0 <= row["task_accuracy"] <= 1.0
        assert row["mse"] >= 0.0
        assert row["regression_mse"] >= 0.0
        assert row["refinement_steps"] >= 1.0
        assert row["compute_proxy"] >= 0.0
        assert row["bytes_proxy"] > 0.0
        assert 0.0 <= row["help_rate"] <= 1.0
        assert 0.0 <= row["harm_rate"] <= 1.0
        assert 0.0 <= row["mse_help_rate"] <= 1.0
        assert 0.0 <= row["mse_harm_rate"] <= 1.0
        assert 0.0 <= row["mean_confidence"] <= 1.0
        assert row["confidence_ece"] >= 0.0
        assert len(row["trajectory_mse"]) == int(row["refinement_steps"])
        assert row["trajectory_best_mse"] <= max(row["trajectory_mse"])
        assert sum(row["failure_reasons"].values()) == config.examples

    assert one_pass["help_rate"] == 0.0
    assert one_pass["harm_rate"] == 0.0
    assert fixed_2["mse"] < one_pass["mse"]
    assert fixed_4["trajectory_best_mse"] <= fixed_2["mse"]
    assert fixed_4["mse_harm_rate"] > 0.0
    assert diffusion["mse"] < one_pass["mse"]
    assert oracle["mse"] < fixed_4["mse"]
    assert oracle["task_accuracy"] >= fixed_4["task_accuracy"]
    assert gated["compute_proxy"] <= fixed_2["compute_proxy"]
    assert 0.0 <= gated["gate_fraction"] <= 1.0
    assert gated["gated_examples"] <= config.examples


def test_toy_iterative_latent_refinement_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "iterative_latent_refinement.json"
    markdown = tmp_path / "iterative_latent_refinement.md"

    payload = bridge.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "11",
            "--examples",
            "24",
            "--dim",
            "16",
            "--classes",
            "4",
            "--styles",
            "5",
            "--diffusion-steps",
            "2",
        ]
    )

    on_disk = json.loads(output.read_text())
    assert on_disk == payload
    assert on_disk["config"]["seed"] == 11
    assert on_disk["config"]["examples"] == 24
    assert len(on_disk["rows"]) == len(bridge.METHODS)
    assert "# Toy Iterative Latent Refinement" in markdown.read_text()
