from __future__ import annotations

import json

from scripts import run_toy_gauge_fix_quotient_bridge as sweep


def test_toy_gauge_fix_quotient_bridge_is_deterministic_and_schema_stable() -> None:
    config = sweep.ToyGaugeFixQuotientBridgeConfig(
        seed=7,
        heads=4,
        head_dim=6,
        classes=5,
        families=5,
        heldout_family=4,
        anchor_family=0,
        seen_shots_per_class=20,
        heldout_shots=(1, 2, 4),
        test_examples_per_class=40,
        class_noise=0.16,
        source_noise=0.03,
        nuisance_rank=3,
        nuisance_strength=0.25,
        head_scale_jitter=0.18,
        head_bias_scale=0.18,
        ridge_lam=1e-2,
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
        assert -1.0 <= row["centroid_cosine"] <= 1.0
        assert row["gauge_residual"] >= 0.0
        if row["method"] in {"gauge_fix_then_bridge", "quotient_match_after_fix"}:
            assert row["head_match_accuracy"] is not None
            assert 0.0 <= row["head_match_accuracy"] <= 1.0
            assert row["shared_basis"] is True
        else:
            assert row["shared_basis"] is False

    lookup = {(row["shot"], row["method"]): row for row in payload["rows"]}
    assert lookup[(1, "quotient_match_after_fix")]["mse"] <= lookup[(1, "global_seen_ridge")]["mse"]
    assert lookup[(1, "quotient_match_after_fix")]["mse"] <= lookup[(1, "gauge_fix_then_bridge")]["mse"]
    assert lookup[(1, "quotient_match_after_fix")]["head_match_accuracy"] >= lookup[(1, "gauge_fix_then_bridge")]["head_match_accuracy"]


def test_toy_gauge_fix_quotient_bridge_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "gauge_fix_quotient_bridge.json"
    markdown = tmp_path / "gauge_fix_quotient_bridge.md"

    payload = sweep.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "7",
            "--heads",
            "4",
            "--head-dim",
            "6",
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
        ]
    )

    loaded = json.loads(output.read_text())
    assert loaded == payload
    text = markdown.read_text()
    assert "# Toy Gauge-Fix Quotient Bridge Sweep" in text
    assert "quotient_match_after_fix" in text
    assert "https://openreview.net/forum?id=KrkbYbK0cH" in text
