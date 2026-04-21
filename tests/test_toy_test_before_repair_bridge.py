from __future__ import annotations

import json

from scripts import run_toy_test_before_repair_bridge as bridge


def test_toy_test_before_repair_bridge_is_deterministic_and_has_required_telemetry() -> None:
    config = bridge.ToyTestBeforeRepairConfig(
        seed=7,
        examples=96,
        pool_size=6,
        chain_length=4,
        severity_bins=3,
        test_threshold=0.69,
        repair_threshold=0.55,
        repair_noise=0.09,
        test_noise=0.05,
        near_miss_bias=0.18,
        route_noise_bias=0.22,
    )

    payload = bridge.run_experiment(config)
    payload_again = bridge.run_experiment(config)
    assert payload == payload_again

    rows = payload["rows"]
    assert [row["method"] for row in rows] == ["repair_only", "test_before_repair", "oracle"]

    lookup = {row["method"]: row for row in rows}
    repair_only = lookup["repair_only"]
    tbr = lookup["test_before_repair"]
    oracle = lookup["oracle"]

    for row in rows:
        assert 0.0 <= row["accuracy"] <= 1.0
        assert 0.0 <= row["repair_application_rate"] <= 1.0
        assert 0.0 <= row["repair_change_rate"] <= 1.0
        assert 0.0 <= row["test_pass_rate"] <= 1.0
        assert 0.0 <= row["bytes_estimate"]
        assert 0.0 <= row["test_bytes_estimate"]
        assert 0.0 <= row["repair_bytes_estimate"]

    assert tbr["accuracy"] >= repair_only["accuracy"]
    assert oracle["accuracy"] >= tbr["accuracy"]
    assert tbr["help_vs_repair_only"] >= 0.0
    assert tbr["harm_vs_repair_only"] >= 0.0
    assert tbr["test_pass_rate"] > 0.0

    example_rows = payload["example_rows"]
    assert len(example_rows) == config.examples * 3
    assert all("selected_test_score" in row for row in example_rows)
    assert all("selected_test_pass" in row for row in example_rows)
    assert all("repair_changed_answer" in row for row in example_rows)
    assert all("selection_bytes_estimate" in row for row in example_rows)

    subgroups = payload["subgroups"]["severity"]
    assert len(subgroups) >= 2
    assert sum(row["count"] for row in subgroups) == config.examples
    assert all("test_before_repair_accuracy" in row for row in subgroups)
    assert all("test_before_repair_pass_rate" in row for row in subgroups)
    weighted_tbr_accuracy = sum(row["test_before_repair_accuracy"] * row["count"] for row in subgroups) / config.examples
    assert abs(weighted_tbr_accuracy - tbr["accuracy"]) < 1e-9


def test_toy_test_before_repair_threshold_controls_pass_rate() -> None:
    low_threshold = bridge.ToyTestBeforeRepairConfig(seed=3, examples=72, test_threshold=0.35)
    high_threshold = bridge.ToyTestBeforeRepairConfig(seed=3, examples=72, test_threshold=0.92)

    low_payload = bridge.run_experiment(low_threshold)
    high_payload = bridge.run_experiment(high_threshold)
    low_tbr = next(row for row in low_payload["rows"] if row["method"] == "test_before_repair")
    high_tbr = next(row for row in high_payload["rows"] if row["method"] == "test_before_repair")

    assert low_tbr["test_pass_rate"] >= high_tbr["test_pass_rate"]


def test_toy_test_before_repair_bridge_cli_writes_jsonl_and_markdown(tmp_path) -> None:
    output = tmp_path / "test_before_repair.jsonl"
    markdown = tmp_path / "test_before_repair.md"

    payload = bridge.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "11",
            "--examples",
            "48",
            "--pool-size",
            "6",
            "--chain-length",
            "4",
            "--severity-bins",
            "3",
        ]
    )

    lines = output.read_text().strip().splitlines()
    assert len(lines) == payload["config"]["examples"] * 3
    first_row = json.loads(lines[0])
    assert first_row["method"] in {"repair_only", "test_before_repair", "oracle"}
    assert "selected_test_pass" in first_row
    assert "bytes_estimate" in first_row
    assert "# Toy Test-Before-Repair Bridge" in markdown.read_text()
