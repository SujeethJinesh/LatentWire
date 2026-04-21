from __future__ import annotations

import json

from scripts import run_toy_shared_byte_span_route_atom_remap as bridge


def test_toy_shared_byte_span_route_atom_remap_is_deterministic_and_interpretable() -> None:
    config = bridge.ToySharedByteSpanRouteAtomRemapConfig(
        seed=9,
        calibration_examples=48,
        test_examples=24,
        min_atoms=4,
        max_atoms=6,
        protected_atoms=3,
        remap_capacity=10,
        low_bits=3,
        high_bits=8,
        signal_scale=2.6,
        distractor_scale=8.2,
        activation_noise=0.03,
        calibration_noise=0.02,
        label_noise=0.20,
    )

    payload = bridge.run_experiment(config)
    payload_again = bridge.run_experiment(config)
    assert payload == payload_again
    assert payload["methods"] == list(bridge.METHODS)
    assert payload["remap_table_size"] >= 0
    assert payload["remap_table_bytes"] > 0

    rows = {row["method"]: row for row in payload["rows"]}
    token_id = rows["token_id"]
    regroup = rows["regroup_baseline"]
    shared = rows["shared_byte_span_remap_route_atoms"]
    oracle = rows["oracle_shared_byte_span_route_atoms"]

    for row in rows.values():
        assert 0.0 <= row["task_accuracy"] <= 1.0
        assert row["mse"] >= 0.0
        assert 0.0 <= row["source_target_boundary_f1"] <= 1.0
        assert 0.0 <= row["remap_coverage"] <= 1.0
        assert 0.0 <= row["atom_recovery"] <= 1.0
        assert row["bytes_proxy"] > 0.0
        assert row["protected_atoms"] == 3

    assert shared["remap_coverage"] > 0.0
    assert shared["atom_recovery"] >= token_id["atom_recovery"]
    assert shared["task_accuracy"] >= token_id["task_accuracy"] - 1e-6
    assert shared["task_accuracy"] >= regroup["task_accuracy"] - 1e-6
    assert shared["mse"] < token_id["mse"]
    assert shared["source_target_boundary_f1"] >= token_id["source_target_boundary_f1"] - 1e-6
    assert shared["bytes_proxy"] <= regroup["bytes_proxy"] + 1e-6
    assert oracle["mse"] <= shared["mse"]
    assert oracle["atom_recovery"] >= shared["atom_recovery"]


def test_toy_shared_byte_span_route_atom_remap_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "shared_byte_span_route_atom_remap.json"
    markdown = tmp_path / "shared_byte_span_route_atom_remap.md"

    payload = bridge.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "5",
            "--calibration-examples",
            "24",
            "--test-examples",
            "12",
            "--protected-atoms",
            "3",
            "--remap-capacity",
            "4",
        ]
    )

    on_disk = json.loads(output.read_text())
    assert on_disk == payload
    assert on_disk["config"]["seed"] == 5
    assert len(on_disk["rows"]) == len(bridge.METHODS)
    assert "Toy Shared Byte/Span Route Atom Remap" in markdown.read_text()
