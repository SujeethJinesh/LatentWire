from __future__ import annotations

import json

from scripts import run_toy_tokenizer_frontier_bridge as bridge


def test_toy_tokenizer_frontier_bridge_is_deterministic_and_shows_boundary_gap() -> None:
    config = bridge.ToyTokenizerFrontierBridgeConfig(
        seed=7,
        train_examples=48,
        test_examples=24,
        min_segments=3,
        max_segments=5,
        remap_capacity=6,
    )

    rows = bridge.run_experiment(config, bridge.METHODS)
    rows_again = bridge.run_experiment(config, bridge.METHODS)
    assert rows == rows_again
    assert [row["method"] for row in rows] == list(bridge.METHODS)

    lookup = {row["method"]: row for row in rows}
    token_id = lookup["token_id"]
    frontier = lookup["frontier_regroup"]
    learned = lookup["learned_remap"]

    assert token_id["exact_reconstruction"] < 1.0
    assert frontier["exact_reconstruction"] == 1.0
    assert learned["exact_reconstruction"] == 1.0
    assert token_id["decoded_boundary_f1"] < frontier["decoded_boundary_f1"]
    assert frontier["decoded_boundary_f1"] == 1.0
    assert learned["decoded_boundary_f1"] == 1.0
    assert token_id["source_target_boundary_f1"] < 1.0
    assert frontier["bytes_per_example"] <= token_id["bytes_per_example"]
    assert learned["bytes_per_example"] <= frontier["bytes_per_example"]
    assert learned["learned_remap_coverage"] > 0.0

    for row in rows:
        assert 0.0 <= row["decoded_boundary_f1"] <= 1.0
        assert 0.0 <= row["source_target_boundary_f1"] <= 1.0
        assert row["bytes_per_example"] > 0.0
        assert 0.0 <= row["token_accuracy"] <= 1.0


def test_toy_tokenizer_frontier_bridge_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "tokenizer_frontier.json"
    markdown = tmp_path / "tokenizer_frontier.md"

    payload = bridge.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "11",
            "--train-examples",
            "24",
            "--test-examples",
            "12",
            "--min-segments",
            "3",
            "--max-segments",
            "4",
            "--remap-capacity",
            "4",
        ]
    )

    on_disk = json.loads(output.read_text())
    assert on_disk == payload
    assert on_disk["config"]["seed"] == 11
    assert on_disk["config"]["train_examples"] == 24
    assert on_disk["config"]["test_examples"] == 12
    assert len(on_disk["rows"]) == len(bridge.METHODS)
    assert on_disk["rows"][0]["method"] == "token_id"
    assert "Toy Tokenizer Frontier Bridge" in markdown.read_text()
