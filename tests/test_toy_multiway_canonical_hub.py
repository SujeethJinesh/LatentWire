from __future__ import annotations

import json

from scripts import run_toy_multiway_canonical_hub as sweep


def test_toy_multiway_canonical_hub_is_deterministic_and_improves_low_shot_transfer() -> None:
    config = sweep.ToyMultiwayCanonicalHubConfig(
        seed=11,
        dim=18,
        classes=5,
        families=5,
        heldout_family=4,
        anchor_family=0,
        seen_shots_per_class=20,
        heldout_shots=(1, 2, 4),
        test_examples_per_class=40,
        class_noise=0.22,
        source_noise=0.035,
        scale_jitter=0.07,
        bias_scale=0.26,
        style_strength=0.08,
        ridge_lam=1e-2,
        gpa_iters=6,
    )

    payload = sweep.run_experiment(config)
    payload_again = sweep.run_experiment(config)
    assert payload == payload_again
    assert payload["methods"] == list(sweep.METHODS)
    assert list(payload["rows"][0].keys()) == list(sweep.ROW_KEY_ORDER)
    assert len(payload["rows"]) == len(config.heldout_shots) * len(sweep.METHODS)

    for row in payload["rows"]:
        assert row["shot"] in config.heldout_shots
        assert 0.0 <= row["accuracy"] <= 1.0
        assert row["mse"] >= 0.0
        assert -1.0 <= row["accuracy_delta_vs_fewshot"] <= 1.0
        assert row["centroid_cosine"] <= 1.0
        assert row["centroid_cosine"] >= -1.0
        assert row["canonical_gap"] >= 0.0

    lookup = {(row["shot"], row["method"]): row for row in payload["rows"]}
    for shot in config.heldout_shots:
        fewshot = lookup[(shot, "heldout_fewshot_ridge")]
        global_seen = lookup[(shot, "global_seen_ridge")]
        multiway = lookup[(shot, "multiway_gpa_canonical")]
        oracle = lookup[(shot, "oracle_family_ridge")]
        assert fewshot["accuracy_delta_vs_fewshot"] == 0.0
        assert multiway["shared_basis"] is True
        assert oracle["accuracy"] >= fewshot["accuracy"] - 1e-6
        assert multiway["accuracy"] >= global_seen["accuracy"] - 1e-6

    assert lookup[(1, "multiway_gpa_canonical")]["accuracy"] >= lookup[(1, "heldout_fewshot_ridge")]["accuracy"]
    assert lookup[(1, "multiway_gpa_canonical")]["mse"] <= lookup[(1, "heldout_fewshot_ridge")]["mse"] + 1e-6


def test_toy_multiway_canonical_hub_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "multiway_canonical_hub.json"
    markdown = tmp_path / "multiway_canonical_hub.md"

    payload = sweep.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "11",
            "--dim",
            "18",
            "--classes",
            "5",
            "--families",
            "5",
            "--heldout-family",
            "4",
            "--anchor-family",
            "0",
            "--seen-shots-per-class",
            "20",
            "--heldout-shots",
            "1",
            "2",
            "4",
            "--test-examples-per-class",
            "24",
            "--class-noise",
            "0.22",
            "--source-noise",
            "0.035",
            "--scale-jitter",
            "0.07",
            "--bias-scale",
            "0.26",
            "--style-strength",
            "0.08",
            "--ridge-lam",
            "0.01",
            "--gpa-iters",
            "6",
        ]
    )

    loaded = json.loads(output.read_text())
    assert loaded == payload
    text = markdown.read_text()
    assert "# Toy Multi-Way Canonical Hub Sweep" in text
    assert "multiway_gpa_canonical" in text
    assert "https://arxiv.org/abs/2602.06205" in text
