from __future__ import annotations

from scripts.run_source_private_tool_trace_compression_baselines import run_gate
from scripts.summarize_source_private_slot_packet_bootstrap import run_summary


def test_slot_packet_bootstrap_summary(tmp_path) -> None:
    result_dir = tmp_path / "gate"
    run_gate(
        output_dir=result_dir,
        train_examples=128,
        eval_examples=64,
        train_family_set="all",
        eval_family_set="all",
        candidates=4,
        feature_dim=256,
        budgets=[6],
        train_seed=7,
        eval_seed=8,
        ridge=1e-2,
        candidate_view="slot",
        fit_intercept=False,
    )

    payload = run_summary(
        result_dirs=[result_dir],
        output_dir=tmp_path / "summary",
        budget=6,
        bootstrap_samples=100,
        seed=11,
    )

    assert payload["pass_gate"] is True
    assert payload["rows"][0]["scalar_accuracy"] >= 0.90
    assert payload["rows"][0]["paired_bootstrap"]["target_only"]["ci95_low"] > 0.15
    assert (tmp_path / "summary" / "summary.md").exists()
