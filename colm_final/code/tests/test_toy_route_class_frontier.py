from __future__ import annotations

import json

from scripts import run_toy_hub_sticky_frontier_stack as stack
from scripts import run_toy_route_class_frontier as sweep


def test_toy_route_class_frontier_is_deterministic_and_schema_stable() -> None:
    config = stack.ToyHubStickyFrontierStackConfig(
        seed=37,
        calibration_examples=72,
        test_examples=56,
        dim=18,
        atoms=12,
        families=5,
        classes=4,
        source_noise=0.025,
        target_noise=0.020,
        route_noise=0.30,
        perturb_noise=0.50,
        route_code_strength=0.70,
        source_style_strength=0.24,
        target_style_strength=0.20,
        family_scale_jitter=0.14,
        family_bias_scale=0.50,
        hub_snap_strength=0.35,
        route_temperature=0.82,
        confidence_temperature=0.72,
        sticky_margin=0.03,
        keep_fraction=0.70,
        low_bits=3,
        high_bits=8,
        protected_atoms=4,
        verifier_noise=0.06,
        verifier_harm_margin=0.01,
        verifier_stop_threshold=0.85,
    )

    payload = sweep.run_experiment(config)
    payload_again = sweep.run_experiment(config)
    assert payload == payload_again
    assert payload["methods"] == list(sweep.METHODS)
    assert [row["method"] for row in payload["rows"]] == list(sweep.METHODS)
    assert list(payload["rows"][0].keys()) == list(sweep.ROW_KEY_ORDER)

    for row in payload["rows"]:
        assert 0.0 <= row["accuracy"] <= 1.0
        assert row["mse"] >= 0.0
        assert 0.0 <= row["route_accuracy"] <= 1.0
        assert -1.0 <= row["patch_rank_correlation"] <= 1.0
        assert 0.0 <= row["protected_oracle_preservation_rate"] <= 1.0
        assert row["selected_atom_count"] > 0.0
        assert row["protected_atom_count"] >= 0.0
        assert row["average_stop_steps"] >= 1.0
        assert 0.0 <= row["over_refinement_rate"] <= 1.0
        assert row["bytes_proxy"] > 0.0
        assert row["compute_proxy"] > 0.0

    lookup = {row["method"]: row for row in payload["rows"]}
    conditional_base = lookup["conditional_prior_base"]
    conditional_quant = lookup["conditional_prior_quant_error_frontier"]
    conditional_patch = lookup["conditional_prior_route_class_patch_protect"]
    conditional_frontier = lookup["conditional_prior_route_class_patch_frontier"]
    conditional_stop = lookup["conditional_prior_route_class_patch_frontier_stop"]
    oracle_base = lookup["oracle_base"]
    oracle_patch = lookup["oracle_route_class_patch_protect"]
    oracle_stop = lookup["oracle_route_class_patch_frontier_stop"]

    assert conditional_base["selected_atom_count"] == config.atoms
    assert conditional_patch["selected_atom_count"] == config.atoms
    assert conditional_frontier["selected_atom_count"] < config.atoms
    assert conditional_stop["average_stop_steps"] > 1.0
    assert oracle_base["route_accuracy"] >= conditional_base["route_accuracy"]
    assert conditional_patch["patch_rank_correlation"] >= conditional_quant["patch_rank_correlation"] - 1e-6
    assert oracle_patch["patch_rank_correlation"] >= 0.0
    assert oracle_stop["protected_atom_count"] == float(config.protected_atoms)


def test_toy_route_class_frontier_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "route_class_frontier.json"
    markdown = tmp_path / "route_class_frontier.md"

    payload = sweep.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "37",
            "--calibration-examples",
            "48",
            "--test-examples",
            "36",
            "--dim",
            "18",
            "--atoms",
            "12",
            "--families",
            "5",
            "--classes",
            "4",
            "--source-noise",
            "0.025",
            "--target-noise",
            "0.020",
            "--route-noise",
            "0.30",
            "--perturb-noise",
            "0.50",
            "--route-code-strength",
            "0.70",
            "--source-style-strength",
            "0.24",
            "--target-style-strength",
            "0.20",
            "--family-scale-jitter",
            "0.14",
            "--family-bias-scale",
            "0.50",
            "--hub-snap-strength",
            "0.35",
            "--route-temperature",
            "0.82",
            "--confidence-temperature",
            "0.72",
            "--sticky-margin",
            "0.03",
            "--keep-fraction",
            "0.70",
            "--low-bits",
            "3",
            "--high-bits",
            "8",
            "--protected-atoms",
            "4",
            "--verifier-noise",
            "0.06",
            "--verifier-harm-margin",
            "0.01",
            "--verifier-stop-threshold",
            "0.85",
        ]
    )

    loaded = json.loads(output.read_text())
    assert loaded == payload
    text = markdown.read_text()
    assert "# Toy Route-Class Frontier Sweep" in text
    assert "route_class_patch_frontier" in text
    assert "https://arxiv.org/abs/2502.03714" in text
