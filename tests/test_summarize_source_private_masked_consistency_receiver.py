from __future__ import annotations

from scripts.summarize_source_private_masked_consistency_receiver import DEFAULT_RUN_DIRS, summarize_runs


def test_summarize_masked_consistency_receiver_runs(tmp_path) -> None:
    payload = summarize_runs(run_dirs=list(DEFAULT_RUN_DIRS), output_dir=tmp_path / "summary")

    assert payload["gate"] == "source_private_masked_consistency_receiver_summary"
    assert payload["headline"]["pass_gate"] is True
    assert payload["headline"]["n256_runs"] == 2
    assert payload["headline"]["min_n256_learned_matched_accuracy"] >= 0.95
    assert payload["headline"]["min_n256_lift_vs_best_control"] >= 0.65
    assert payload["headline"]["min_n256_ci95_low_vs_best_control"] >= 0.60
    assert payload["headline"]["all_exact_id_parity"] is True
    assert (tmp_path / "summary" / "summary.json").exists()
    assert (tmp_path / "summary" / "summary.md").exists()
