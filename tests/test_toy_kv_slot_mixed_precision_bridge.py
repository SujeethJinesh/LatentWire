from __future__ import annotations

import json

from scripts import run_toy_kv_slot_mixed_precision_bridge as bridge


def test_toy_kv_slot_mixed_precision_bridge_is_deterministic_and_schema_stable() -> None:
    config = bridge.ToyKVSlotMixedPrecisionConfig(
        seed=13,
        train_examples=96,
        test_examples=64,
        slots=12,
        dim=16,
        classes=5,
        low_bits=3,
        key_high_bits=6,
        value_high_bits=5,
        protected_key_channels=2,
        protected_value_channels=2,
        route_signal_channels=4,
        answer_signal_channels=4,
        route_signal_scale=4.5,
        answer_signal_scale=4.0,
        key_noise=0.18,
        value_noise=0.28,
        query_noise=0.22,
    )

    payload = bridge.run_experiment(config)
    payload_again = bridge.run_experiment(config)
    assert payload == payload_again

    rows = payload["rows"]
    assert [row["method"] for row in rows] == [
        "uniform_low_bit",
        "key_protected_value_low",
        "value_protected_key_low",
        "mixed_kv_precision",
        "incoherent_basis_rotation",
    ]

    required_fields = {
        "method",
        "basis",
        "route_accuracy",
        "answer_accuracy",
        "key_mse",
        "value_mse",
        "bytes_estimate",
        "route_help_vs_uniform",
        "route_harm_vs_uniform",
        "answer_help_vs_uniform",
        "answer_harm_vs_uniform",
        "route_delta_vs_uniform",
        "answer_delta_vs_uniform",
        "key_selected_indices",
        "value_selected_indices",
        "basis_orthogonality_error",
    }
    for row in rows:
        assert required_fields.issubset(row)
        assert 0.0 <= row["route_accuracy"] <= 1.0
        assert 0.0 <= row["answer_accuracy"] <= 1.0
        assert row["key_mse"] >= 0.0
        assert row["value_mse"] >= 0.0
        assert row["bytes_estimate"] > 0.0
        assert 0.0 <= row["route_help_vs_uniform"] <= 1.0
        assert 0.0 <= row["route_harm_vs_uniform"] <= 1.0
        assert 0.0 <= row["answer_help_vs_uniform"] <= 1.0
        assert 0.0 <= row["answer_harm_vs_uniform"] <= 1.0
        assert isinstance(row["key_selected_indices"], list)
        assert isinstance(row["value_selected_indices"], list)

    uniform = rows[0]
    key_protected = rows[1]
    value_protected = rows[2]
    mixed = rows[3]
    rotated = rows[4]

    byte_band = [row["bytes_estimate"] for row in rows]
    assert max(byte_band) - min(byte_band) <= 0.35 * uniform["bytes_estimate"]

    assert key_protected["route_accuracy"] >= uniform["route_accuracy"] - 1e-6
    assert value_protected["answer_accuracy"] >= uniform["answer_accuracy"] - 1e-6
    assert mixed["route_accuracy"] >= uniform["route_accuracy"] - 1e-6
    assert mixed["answer_accuracy"] >= uniform["answer_accuracy"] - 1e-6
    assert rotated["basis"] in {"hadamard", "orthogonal"}
    assert rotated["basis_orthogonality_error"] >= 0.0

    assert (
        max(row["route_accuracy"] for row in rows[1:]) > uniform["route_accuracy"]
        or max(row["answer_accuracy"] for row in rows[1:]) > uniform["answer_accuracy"]
    )


def test_toy_kv_slot_mixed_precision_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "kv_slot_mixed_precision_bridge.json"
    markdown = tmp_path / "kv_slot_mixed_precision_bridge.md"

    payload = bridge.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "5",
            "--train-examples",
            "64",
            "--test-examples",
            "32",
            "--slots",
            "8",
            "--dim",
            "16",
            "--classes",
            "4",
            "--low-bits",
            "3",
            "--key-high-bits",
            "6",
            "--value-high-bits",
            "5",
            "--protected-key-channels",
            "2",
            "--protected-value-channels",
            "2",
        ]
    )

    loaded = json.loads(output.read_text())
    assert loaded["config"]["seed"] == 5
    assert loaded["config"]["dim"] == 16
    assert len(loaded["rows"]) == 5
    assert payload["rows"][0]["method"] == "uniform_low_bit"
    assert "# Toy K/V Slot Mixed-Precision Bridge" in markdown.read_text()
    assert "| Method | Basis | Route acc | Answer acc | K MSE | V MSE | Bytes estimate | Route help | Answer help |" in markdown.read_text()
