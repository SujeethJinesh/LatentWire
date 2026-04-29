from __future__ import annotations

import json

from scripts.run_source_private_tool_trace_compression_baselines import run_gate


def test_tool_trace_compression_baselines_write_artifacts(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path,
        train_examples=64,
        eval_examples=32,
        train_family_set="all",
        eval_family_set="all",
        candidates=4,
        feature_dim=128,
        budgets=[6],
        train_seed=5,
        eval_seed=6,
        ridge=1e-2,
    )

    summary = json.loads((tmp_path / "summary.json").read_text())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    row = payload["budget_summaries"][0]
    assert payload["exact_id_parity"] is True
    assert summary["budget_summaries"][0]["budget_bytes"] == 6
    assert "predictions_budget6.jsonl" in manifest["artifacts"]
    assert (tmp_path / "predictions_budget6.jsonl").exists()
    assert "best_compression_baseline_accuracy" in row


def test_tool_trace_compression_baselines_include_source_destroying_controls(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path,
        train_examples=128,
        eval_examples=64,
        train_family_set="all",
        eval_family_set="all",
        candidates=4,
        feature_dim=256,
        budgets=[6],
        train_seed=3,
        eval_seed=4,
        ridge=1e-2,
    )

    metrics = payload["budget_summaries"][0]["metrics"]
    assert metrics["zero_source"]["accuracy"] == metrics["target_only"]["accuracy"]
    assert "scalar_quantized_source" in metrics
    assert "scalar_shuffled_source" in metrics
    assert "scalar_answer_masked_source" in metrics
    assert "raw_source_sign_sketch" in metrics
