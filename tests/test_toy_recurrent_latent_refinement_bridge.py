from __future__ import annotations

import json

from scripts import run_toy_recurrent_latent_refinement_bridge as bridge


def test_toy_recurrent_latent_refinement_bridge_is_deterministic_and_schema_complete() -> None:
    config = bridge.ToyRecurrentLatentRefinementConfig(
        seed=7,
        examples=96,
        dim=24,
        classes=5,
        block_size=4,
        bridge_noise=0.18,
        residual_noise=0.09,
        gate_fraction=0.25,
        diffusion_steps=4,
        diffusion_shrink=0.65,
        quant_bits=4,
    )

    rows = bridge.run_experiment(config)
    rows_again = bridge.run_experiment(config)
    assert rows == rows_again
    assert [row["method"] for row in rows] == [
        "one_shot_bridge",
        "two_step_residual_refinement",
        "gated_refinement",
        "blockwise_diffusion_denoise",
        "oracle_upper_bound",
    ]

    lookup = {row["method"]: row for row in rows}
    one_shot = lookup["one_shot_bridge"]
    two_step = lookup["two_step_residual_refinement"]
    gated = lookup["gated_refinement"]
    diffusion = lookup["blockwise_diffusion_denoise"]
    oracle = lookup["oracle_upper_bound"]

    for row in rows:
        expected_len = int(row["steps"]) if row["steps"] else 1
        assert 0.0 <= row["accuracy"] <= 1.0
        assert row["mse"] >= 0.0
        assert -1.0 <= row["cosine"] <= 1.0
        assert row["reg_mse"] >= 0.0
        assert 0.0 <= row["confidence"] <= 1.0
        assert row["bytes_estimate"] >= 0.0
        assert row["compute_proxy"] >= 0.0
        assert row["steps"] >= 0.0
        assert 0.0 <= row["help_rate"] <= 1.0
        assert 0.0 <= row["harm_rate"] <= 1.0
        assert -1.0 <= row["net_help_rate"] <= 1.0
        assert len(row["trajectory_mse"]) == expected_len
        assert len(row["trajectory_cosine"]) == len(row["trajectory_mse"])
        assert len(row["trajectory_residual_norm"]) == len(row["trajectory_mse"])
        assert row["trajectory_initial_mse"] >= 0.0
        assert row["trajectory_final_mse"] >= 0.0
        assert row["trajectory_best_mse"] >= 0.0
        assert row["trajectory_convergence_ratio"] >= 0.0

    assert one_shot["help_rate"] == 0.0
    assert one_shot["harm_rate"] == 0.0
    assert oracle["accuracy"] >= diffusion["accuracy"]
    assert oracle["mse"] <= diffusion["mse"]
    assert oracle["cosine"] >= diffusion["cosine"]

    assert (
        two_step["mse"] < one_shot["mse"]
        or gated["mse"] < one_shot["mse"]
        or diffusion["mse"] < one_shot["mse"]
    )
    assert (
        two_step["accuracy"] >= one_shot["accuracy"]
        or gated["accuracy"] >= one_shot["accuracy"]
        or diffusion["accuracy"] >= one_shot["accuracy"]
    )

    assert diffusion["trajectory_best_mse"] <= diffusion["trajectory_initial_mse"]
    assert diffusion["trajectory_best_mse"] <= diffusion["trajectory_final_mse"]

    assert "gate_entropy" in gated
    assert "gate_fraction_effective" in gated


def test_toy_recurrent_latent_refinement_bridge_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "toy_recurrent_latent.json"
    markdown = tmp_path / "toy_recurrent_latent.md"

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
            "--block-size",
            "4",
            "--diffusion-steps",
            "3",
        ]
    )

    on_disk = json.loads(output.read_text())
    assert on_disk == payload
    assert on_disk["config"]["seed"] == 11
    assert on_disk["config"]["examples"] == 24
    assert len(on_disk["rows"]) == 5
    assert on_disk["rows"][0]["method"] == "one_shot_bridge"
    assert "# Toy Recurrent Latent Refinement Bridge" in markdown.read_text()
