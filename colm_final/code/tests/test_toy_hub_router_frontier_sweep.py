from __future__ import annotations

import json

from scripts import run_toy_hub_router_frontier_sweep as sweep
from scripts import run_toy_hub_sticky_frontier_stack as stack


def test_toy_hub_router_frontier_sweep_is_deterministic_and_interpretable() -> None:
    config = stack.ToyHubStickyFrontierStackConfig(
        seed=31,
        calibration_examples=96,
        test_examples=72,
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

    rows = payload["rows"]
    assert list(rows[0].keys()) == list(sweep.ROW_KEY_ORDER)
    assert rows[0]["method"] == "raw_pairwise_bridge"
    assert len(rows) == 1 + len(sweep.ROUTER_METHODS) * 3
    assert len(payload["router_summary"]) == len(sweep.ROUTER_METHODS)

    for row in rows:
        assert 0.0 <= row["accuracy"] <= 1.0
        assert row["mse"] >= 0.0
        assert 0.0 <= row["route_accuracy"] <= 1.0
        assert 0.0 <= row["perturbation_stability"] <= 1.0
        assert 0.0 <= row["atom_recovery"] <= 1.0
        assert row["average_stop_steps"] >= 1.0
        assert 0.0 <= row["over_refinement_rate"] <= 1.0
        assert row["bytes_proxy"] > 0.0
        assert row["compute_proxy"] > 0.0

    lookup = {row["method"]: row for row in rows}
    raw_pairwise = lookup["raw_pairwise_bridge"]
    feature_base = lookup["feature_router_base"]
    sticky_base = lookup["sticky_router_base"]
    sticky_frontier = lookup["sticky_router_frontier"]
    sticky_stop = lookup["sticky_router_frontier_stop"]
    random_base = lookup["random_router_base"]
    oracle_base = lookup["oracle_router_base"]
    oracle_frontier = lookup["oracle_router_frontier"]
    oracle_stop = lookup["oracle_router_frontier_stop"]

    assert raw_pairwise["accuracy_delta_vs_raw_pairwise"] == 0.0
    assert raw_pairwise["frontier_delta_vs_same_router_base"] == 0.0
    assert sticky_base["perturbation_stability"] >= feature_base["perturbation_stability"] - 1e-6
    assert sticky_frontier["atom_recovery"] > 0.0
    assert oracle_base["route_accuracy"] >= random_base["route_accuracy"]
    assert oracle_frontier["frontier_delta_vs_same_router_base"] == oracle_frontier["accuracy"] - oracle_base["accuracy"]
    assert oracle_stop["stop_delta_vs_same_router_frontier"] == oracle_stop["accuracy"] - oracle_frontier["accuracy"]
    assert sticky_stop["frontier_delta_vs_same_router_base"] == sticky_frontier["accuracy"] - sticky_base["accuracy"]


def test_toy_hub_router_frontier_sweep_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "hub_router_frontier_sweep.json"
    markdown = tmp_path / "hub_router_frontier_sweep.md"

    payload = sweep.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "31",
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
    assert "# Toy Hub Router / Frontier Sweep" in text
    assert "## Router Summary" in text
    assert "https://arxiv.org/abs/2506.14038" in text
