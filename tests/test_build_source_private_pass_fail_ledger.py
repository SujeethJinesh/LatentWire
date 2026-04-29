from __future__ import annotations

import pathlib

from scripts.build_source_private_pass_fail_ledger import build_pass_fail_ledger


def test_pass_fail_ledger_keeps_promoted_and_failed_rows(tmp_path) -> None:
    payload = build_pass_fail_ledger(
        frontier_json=pathlib.Path("results/source_private_cpu_systems_frontier_20260429/cpu_systems_frontier.json"),
        output_dir=tmp_path / "pass_fail_ledger",
    )
    rows = {row["row_id"]: row for row in payload["rows"]}

    assert payload["total_rows"] >= 104
    assert payload["by_bucket"]["paper_ready_evidence"] >= 3
    assert payload["by_bucket"]["positive_needs_more_evidence"] >= 50
    assert payload["by_bucket"]["failed_or_pruned"] >= 40

    n160 = rows["endpoint_label_strict_n160_paired_uncertainty"]
    assert n160["reviewer_bucket"] == "paper_ready_evidence"
    assert n160["ci95_low_vs_target"] >= 0.35
    assert n160["ci95_low_vs_comparator"] >= 0.35
    assert n160["evidence_complete"] is True

    learned_same = rows["candidate_embedding_receiver_diagnostic_budget8_seed37_38"]
    assert learned_same["reviewer_bucket"] == "positive_needs_more_evidence"
    assert learned_same["matched_minus_best_control"] >= 0.15

    learned_cross = rows["candidate_embedding_receiver_heldout_code_similarity_budget8_seed29_30"]
    assert learned_cross["reviewer_bucket"] == "failed_or_pruned"
    assert "insufficient matched-control delta" in learned_cross["pruning_reason"]

    assert (tmp_path / "pass_fail_ledger" / "pass_fail_ledger.json").exists()
    assert (tmp_path / "pass_fail_ledger" / "pass_fail_ledger.csv").exists()
    assert (tmp_path / "pass_fail_ledger" / "pass_fail_ledger.md").exists()
