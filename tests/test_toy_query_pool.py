from __future__ import annotations

import json

from scripts import run_toy_query_pool


def test_toy_query_pool_experiment_returns_interpretable_rows() -> None:
    config = run_toy_query_pool.ToyConfig(
        seed=3,
        train_examples=24,
        test_examples=12,
        dim=8,
        slots=6,
        classes=3,
        top_k=2,
        pool_slots=2,
        epochs=3,
        lr=1e-2,
        rec_weight=0.1,
    )

    rows = run_toy_query_pool.run_experiment(config, ["aligned", "rotated"])

    assert {(row["scenario"], row["method"]) for row in rows} == {
        ("aligned", "topk"),
        ("aligned", "query_pool"),
        ("rotated", "topk"),
        ("rotated", "query_pool"),
    }
    for row in rows:
        assert 0.0 <= row["task_acc"] <= 1.0
        assert row["rec_mse"] >= 0.0
        assert row["route_entropy"] >= 0.0
        assert 0.0 <= row["slot_collision_rate"] <= 1.0
        assert 0.0 <= row["dead_slot_rate"] <= 1.0


def test_toy_query_pool_cli_writes_json(tmp_path) -> None:
    output = tmp_path / "toy.json"
    markdown = tmp_path / "toy.md"

    # Exercise the core serialization shape without paying for the default CLI run.
    config = run_toy_query_pool.ToyConfig(
        seed=1,
        train_examples=16,
        test_examples=8,
        dim=8,
        slots=5,
        classes=3,
        top_k=2,
        pool_slots=2,
        epochs=2,
    )
    payload = {
        "config": config.__dict__,
        "rows": run_toy_query_pool.run_experiment(config, ["outlier"]),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    run_toy_query_pool.write_markdown_summary(payload["rows"], markdown)
    loaded = json.loads(output.read_text())

    assert loaded["config"]["pool_slots"] == 2
    assert {row["method"] for row in loaded["rows"]} == {"topk", "query_pool"}
    assert "| outlier | query_pool |" in markdown.read_text()
