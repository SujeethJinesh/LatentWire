from __future__ import annotations

import json

from scripts import run_toy_latent_refinement_bridge as bridge


def test_toy_latent_refinement_bridge_rows_are_deterministic_and_ordered() -> None:
    config = bridge.ToyLatentRefinementConfig(
        seed=7,
        examples=96,
        dim=24,
        classes=5,
        codebook_size=6,
        query_bank_size=4,
        refinement_steps=3,
        gate_fraction=0.25,
        bridge_noise=0.12,
        residual_noise=0.06,
        soft_temperature=0.75,
    )

    rows = bridge.run_experiment(config)
    rows_again = bridge.run_experiment(config)

    assert rows == rows_again
    assert [row["method"] for row in rows] == [
        "one_shot_noisy_bridge",
        "iterative_residual_refinement",
        "gated_refinement",
        "soft_token_mixture_projection",
        "coarse_to_fine_query_bank",
    ]

    lookup = {row["method"]: row for row in rows}
    one_shot = lookup["one_shot_noisy_bridge"]
    iterative = lookup["iterative_residual_refinement"]
    gated = lookup["gated_refinement"]
    soft = lookup["soft_token_mixture_projection"]
    coarse = lookup["coarse_to_fine_query_bank"]

    for row in rows:
        assert 0.0 <= row["accuracy"] <= 1.0
        assert row["mse"] >= 0.0
        assert row["entropy"] >= 0.0
        assert 0.0 <= row["confidence"] <= 1.0
        assert row["bytes_estimate"] > 0.0
        assert row["steps"] >= 1.0

    assert iterative["accuracy"] >= one_shot["accuracy"]
    assert iterative["mse"] <= one_shot["mse"]
    assert gated["accuracy"] >= one_shot["accuracy"]
    assert gated["mse"] <= one_shot["mse"]
    assert coarse["accuracy"] >= one_shot["accuracy"]
    assert coarse["mse"] <= one_shot["mse"]
    assert coarse["bytes_estimate"] < one_shot["bytes_estimate"]
    assert soft["entropy"] >= one_shot["entropy"]

    assert "gate_entropy" in gated
    assert "fine_bits" in coarse


def test_toy_latent_refinement_bridge_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "toy_latent.json"
    markdown = tmp_path / "toy_latent.md"

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
            "--codebook-size",
            "5",
            "--query-bank-size",
            "3",
            "--refinement-steps",
            "2",
        ]
    )

    on_disk = json.loads(output.read_text())
    assert on_disk == payload
    assert on_disk["config"]["seed"] == 11
    assert on_disk["config"]["examples"] == 24
    assert len(on_disk["rows"]) == 5
    assert on_disk["rows"][0]["method"] == "one_shot_noisy_bridge"

    md = markdown.read_text()
    assert "# Toy Latent Refinement Bridge" in md
    assert "| Method | Accuracy | MSE | Entropy | Confidence | Bytes estimate | Steps |" in md
