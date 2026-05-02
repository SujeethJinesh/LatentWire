from __future__ import annotations

from scripts.run_source_private_tool_trace_compression_baselines import run_gate
from scripts.summarize_source_private_relative_score_bootstrap import run_summary


def test_relative_score_bootstrap_summary_passes_on_smoke(tmp_path) -> None:
    result_dir = tmp_path / "relative"
    run_gate(
        output_dir=result_dir,
        train_examples=64,
        eval_examples=32,
        train_family_set="all",
        eval_family_set="all",
        candidates=4,
        feature_dim=128,
        budgets=[4],
        train_seed=5,
        eval_seed=6,
        ridge=1e-2,
        candidate_view="slot",
        fit_intercept=False,
        packet_variants=["relative_scores"],
    )
    payload = run_summary(
        result_dirs=[result_dir],
        output_dir=tmp_path / "summary",
        budget=4,
        bootstrap_samples=100,
        seed=11,
    )

    assert payload["rows"][0]["relative_payload_bytes"] == 4
    assert payload["rows"][0]["relative_accuracy"] >= payload["rows"][0]["target_accuracy"]
    assert (tmp_path / "summary" / "summary.md").exists()


def test_relative_score_bootstrap_summary_supports_canonical_method(tmp_path) -> None:
    result_dir = tmp_path / "relative_canonical"
    run_gate(
        output_dir=result_dir,
        train_examples=64,
        eval_examples=32,
        train_family_set="all",
        eval_family_set="all",
        candidates=4,
        feature_dim=128,
        budgets=[4],
        train_seed=5,
        eval_seed=6,
        ridge=1e-2,
        candidate_view="slot",
        fit_intercept=False,
        remap_slot_seed=101,
        packet_variants=["relative_scores_canonical"],
    )
    payload = run_summary(
        result_dirs=[result_dir],
        output_dir=tmp_path / "summary_canonical",
        budget=4,
        bootstrap_samples=100,
        seed=11,
        method_condition="relative_canonical_score_source",
    )

    assert payload["method_condition"] == "relative_canonical_score_source"
    assert payload["rows"][0]["relative_payload_bytes"] == 4
    assert "relative_canonical_random_same_byte" in payload["rows"][0]["paired_bootstrap"]
    assert (tmp_path / "summary_canonical" / "summary.md").exists()
