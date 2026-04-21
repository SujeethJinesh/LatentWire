from __future__ import annotations

import json

from scripts import run_toy_process_repair_bridge as bridge


def test_toy_process_repair_bridge_is_deterministic_and_has_required_telemetry() -> None:
    config = bridge.ToyProcessRepairConfig(
        seed=7,
        examples=96,
        pool_size=5,
        chain_length=4,
        severity_bins=3,
        repair_threshold=0.55,
        repair_noise=0.11,
        rerank_noise=0.06,
        near_miss_bias=0.16,
    )

    payload = bridge.run_experiment(config)
    payload_again = bridge.run_experiment(config)
    assert payload == payload_again

    rows = payload["rows"]
    assert [row["method"] for row in rows] == ["rerank_only", "process_aware_repair", "oracle"]

    lookup = {row["method"]: row for row in rows}
    rerank_only = lookup["rerank_only"]
    repair = lookup["process_aware_repair"]
    oracle = lookup["oracle"]

    for row in rows:
        assert 0.0 <= row["accuracy"] <= 1.0
        assert 0.0 <= row["oracle_accuracy"] <= 1.0
        assert 0.0 <= row["repair_application_rate"] <= 1.0
        assert 0.0 <= row["false_repair_rate"] <= 1.0
        assert row["mean_selected_severity"] >= 0.0

    assert repair["accuracy"] >= rerank_only["accuracy"]
    assert oracle["accuracy"] >= repair["accuracy"]
    assert repair["repair_application_rate"] > 0.0
    assert repair["false_repair_rate"] < 0.25

    subgroups = payload["subgroups"]["severity"]
    assert len(subgroups) >= 2
    assert sum(row["count"] for row in subgroups) == config.examples
    assert all("repair_accuracy" in row for row in subgroups)
    assert all("false_repair_rate" in row for row in subgroups)


def test_toy_process_repair_threshold_controls_application_rate() -> None:
    low_threshold = bridge.ToyProcessRepairConfig(seed=3, examples=72, repair_threshold=0.25)
    high_threshold = bridge.ToyProcessRepairConfig(seed=3, examples=72, repair_threshold=0.95)

    low_payload = bridge.run_experiment(low_threshold)
    high_payload = bridge.run_experiment(high_threshold)
    low_repair = next(row for row in low_payload["rows"] if row["method"] == "process_aware_repair")
    high_repair = next(row for row in high_payload["rows"] if row["method"] == "process_aware_repair")

    assert low_repair["repair_application_rate"] >= high_repair["repair_application_rate"]


def test_toy_process_repair_bridge_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "toy_process_repair.json"
    markdown = tmp_path / "toy_process_repair.md"

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
            "5",
            "--chain-length",
            "4",
            "--severity-bins",
            "3",
        ]
    )

    on_disk = json.loads(output.read_text())
    assert on_disk == payload
    assert on_disk["config"]["seed"] == 11
    assert len(on_disk["rows"]) == 3
    assert on_disk["rows"][1]["method"] == "process_aware_repair"
    assert "# Toy Process Repair Bridge" in markdown.read_text()
