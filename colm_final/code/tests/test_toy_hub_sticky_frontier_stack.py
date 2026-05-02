from __future__ import annotations

import json

from scripts import run_toy_hub_sticky_frontier_stack as stack


def test_toy_hub_sticky_frontier_stack_is_deterministic_and_schema_stable() -> None:
    config = stack.ToyHubStickyFrontierStackConfig(
        seed=29,
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

    payload = stack.run_experiment(config)
    payload_again = stack.run_experiment(config)
    assert payload == payload_again
    assert payload["methods"] == list(stack.METHODS)
    assert list(payload.keys()) == [
        "config",
        "methods",
        "rows",
        "interpretation",
        "sources_consulted",
    ]
    assert [row["method"] for row in payload["rows"]] == list(stack.METHODS)
    assert list(payload["rows"][0].keys()) == list(stack.ROW_KEY_ORDER)

    required_fields = {
        "method",
        "route_policy",
        "seed",
        "accuracy",
        "mse",
        "accuracy_delta_vs_raw_pairwise",
        "mse_delta_vs_raw_pairwise",
        "route_accuracy",
        "route_entropy",
        "route_load",
        "perturbation_stability",
        "atom_recovery",
        "selected_atom_count",
        "protected_atom_count",
        "bit_histogram",
        "route_histogram",
        "average_stop_steps",
        "over_refinement_rate",
        "stop_reasons",
        "stop_histogram",
        "bytes_proxy",
        "compute_proxy",
        "help_vs_raw_pairwise",
        "harm_vs_raw_pairwise",
    }
    for row in payload["rows"]:
        assert required_fields.issubset(row)
        assert 0.0 <= row["accuracy"] <= 1.0
        assert row["mse"] >= 0.0
        assert 0.0 <= row["route_accuracy"] <= 1.0
        assert row["route_entropy"] >= 0.0
        assert 0.0 <= row["route_load"] <= 1.0
        assert 0.0 <= row["perturbation_stability"] <= 1.0
        assert 0.0 <= row["atom_recovery"] <= 1.0
        assert row["selected_atom_count"] >= 0.0
        assert row["protected_atom_count"] >= 0.0
        assert isinstance(row["bit_histogram"], dict)
        assert isinstance(row["route_histogram"], dict)
        assert row["average_stop_steps"] >= 1.0
        assert 0.0 <= row["over_refinement_rate"] <= 1.0
        assert isinstance(row["stop_reasons"], dict)
        assert isinstance(row["stop_histogram"], dict)
        assert row["bytes_proxy"] > 0.0
        assert row["compute_proxy"] > 0.0
        assert 0.0 <= row["help_vs_raw_pairwise"] <= 1.0
        assert 0.0 <= row["harm_vs_raw_pairwise"] <= 1.0

    lookup = {row["method"]: row for row in payload["rows"]}
    raw_pairwise = lookup["raw_pairwise_bridge"]
    monolithic = lookup["monolithic_bridge"]
    hub_only = lookup["hub_dictionary_only"]
    feature = lookup["hub_feature_router"]
    sticky = lookup["hub_sticky_router"]
    frontier = lookup["hub_sticky_protected_mixed_bit_frontier"]
    stop = lookup["hub_sticky_frontier_verifier_stop"]
    random = lookup["random_router_control"]
    confidence = lookup["confidence_router_control"]
    oracle = lookup["oracle_router_control"]

    assert raw_pairwise["accuracy_delta_vs_raw_pairwise"] == 0.0
    assert abs(raw_pairwise["mse_delta_vs_raw_pairwise"]) <= 1e-6
    assert sticky["perturbation_stability"] >= feature["perturbation_stability"] - 1e-6
    assert frontier["atom_recovery"] >= sticky["atom_recovery"] - 1e-6
    assert frontier["bit_histogram"].get(str(config.high_bits), 0) > 0
    assert stop["average_stop_steps"] > 1.0
    assert stop["average_stop_steps"] <= 4.0
    assert sum(stop["stop_reasons"].values()) == config.test_examples
    assert stop["stop_reasons"]["verifier_harm"] > 0
    assert sum(stop["stop_histogram"].values()) == config.test_examples
    assert feature["route_accuracy"] >= random["route_accuracy"] - 1e-6
    assert oracle["route_accuracy"] >= random["route_accuracy"] - 1e-6
    assert confidence["route_accuracy"] >= random["route_accuracy"] - 1e-6
    assert monolithic["bytes_proxy"] <= raw_pairwise["bytes_proxy"]


def test_toy_hub_sticky_frontier_stack_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "toy_hub_sticky_frontier_stack.json"
    markdown = tmp_path / "toy_hub_sticky_frontier_stack.md"

    payload = stack.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "29",
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
    assert loaded["config"]["seed"] == 29
    assert loaded["methods"] == list(stack.METHODS)
    assert len(loaded["rows"]) == len(stack.METHODS)
    markdown_text = markdown.read_text()
    assert "# Toy Hub Sticky Frontier Stack" in markdown_text
    assert "| Method | Accuracy | MSE | Route acc | Route entropy | Route load | Perturb stability | Atom recovery | Avg stop steps | Over-refine | Bytes proxy | Compute proxy | Help vs raw pairwise | Harm vs raw pairwise |" in markdown_text
    assert "## Bit Histograms" in markdown_text
    assert "https://arxiv.org/abs/2502.03714" in markdown_text
