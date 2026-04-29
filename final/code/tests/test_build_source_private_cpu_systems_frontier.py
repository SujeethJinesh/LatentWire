from __future__ import annotations

from scripts.build_source_private_cpu_systems_frontier import build_cpu_frontier


def test_cpu_systems_frontier_includes_passes_and_failures(tmp_path) -> None:
    payload = build_cpu_frontier(output_dir=tmp_path / "frontier")
    contributions = {row["contribution"] for row in payload["rows"]}
    statuses = {row["status"] for row in payload["rows"]}

    assert payload["headline"]["total_rows"] == len(payload["rows"])
    assert "byte-rate systems frontier" in contributions
    assert "learned scalar packet" in contributions
    assert "canonical RASP cross-family falsification" in contributions
    assert "consistency posterior negative ablation" in contributions
    assert "Mac endpoint-proxy byte/TTFT frontier" in contributions
    assert "endpoint paired uncertainty" in contributions
    assert "learned target-preserving receiver" in contributions
    assert "pass" in statuses
    assert "fail" in statuses
    assert (tmp_path / "frontier" / "cpu_systems_frontier.csv").exists()

    endpoint_rows = {row["row_id"]: row for row in payload["rows"]}
    strict_core = endpoint_rows["endpoint_proxy_core_n32_audit_strict_controls"]
    strict_holdout = endpoint_rows["endpoint_proxy_holdout_n32_audit_strict_controls"]
    label_core = endpoint_rows["endpoint_proxy_core_n16_label_strict_controls"]
    label_holdout = endpoint_rows["endpoint_proxy_holdout_n16_label_strict_controls"]
    label_core_n32 = endpoint_rows["endpoint_proxy_core_n32_label_strict_controls"]
    label_holdout_n32 = endpoint_rows["endpoint_proxy_holdout_n32_label_strict_controls"]
    label_core_n64 = endpoint_rows["endpoint_proxy_core_n64_label_strict_controls"]
    label_holdout_n64 = endpoint_rows["endpoint_proxy_holdout_n64_label_strict_controls"]
    label_core_n160 = endpoint_rows["endpoint_proxy_core_n160_label_strict_controls"]
    label_holdout_n160 = endpoint_rows["endpoint_proxy_holdout_n160_label_strict_controls"]
    uncertainty = endpoint_rows["endpoint_label_strict_n64_paired_uncertainty"]
    uncertainty_n160 = endpoint_rows["endpoint_core_label_strict_n160_paired_uncertainty"]
    uncertainty_n160_both = endpoint_rows["endpoint_label_strict_n160_paired_uncertainty"]
    learned_receiver = endpoint_rows["candidate_embedding_receiver_gated_budget4_seed29_30"]
    learned_receiver_8b = endpoint_rows["candidate_embedding_receiver_diagnostic_budget8_seed37_38"]
    learned_receiver_heldout = endpoint_rows["candidate_embedding_receiver_heldout_core_to_holdout_budget8_seed29_30"]
    learned_receiver_code_similarity = endpoint_rows["candidate_embedding_receiver_heldout_code_similarity_budget8_seed29_30"]
    learned_receiver_anchor_similarity = endpoint_rows[
        "candidate_embedding_receiver_heldout_anchor_relative_code_similarity_budget8_seed29_30"
    ]
    learned_receiver_anchor_ridge = endpoint_rows["candidate_embedding_receiver_heldout_anchor_relative_ridge_budget8_seed29_30"]
    n64_audit = endpoint_rows["endpoint_proxy_core_n64_audit_payload_gated_nearmiss"]
    assert payload["headline"]["total_rows"] >= 104
    assert strict_core["status"] == "fail"
    assert strict_core["accuracy"] > strict_core["best_control_accuracy"]
    assert strict_core["best_control_accuracy"] == 0.21875
    assert strict_holdout["status"] == "fail"
    assert strict_holdout["accuracy"] > strict_holdout["best_control_accuracy"]
    assert strict_holdout["best_control_accuracy"] == 0.1875
    assert n64_audit["status"] == "fail"
    assert n64_audit["accuracy"] > n64_audit["best_control_accuracy"]
    assert label_core["status"] == "pass"
    assert label_core["accuracy"] > label_core["best_control_accuracy"]
    assert label_holdout["status"] == "pass"
    assert label_holdout["accuracy"] > label_holdout["best_control_accuracy"]
    assert label_core_n32["status"] == "pass"
    assert label_core_n32["accuracy"] > label_core_n32["best_control_accuracy"]
    assert label_holdout_n32["status"] == "pass"
    assert label_holdout_n32["accuracy"] > label_holdout_n32["best_control_accuracy"]
    assert label_core_n64["status"] == "pass"
    assert label_core_n64["accuracy"] > label_core_n64["best_control_accuracy"]
    assert label_holdout_n64["status"] == "pass"
    assert label_holdout_n64["accuracy"] > label_holdout_n64["best_control_accuracy"]
    assert label_core_n160["status"] == "pass"
    assert label_core_n160["accuracy"] > label_core_n160["best_control_accuracy"]
    assert label_holdout_n160["status"] == "pass"
    assert label_holdout_n160["accuracy"] > label_holdout_n160["best_control_accuracy"]
    assert uncertainty["status"] == "pass"
    assert uncertainty["ci95_low_vs_target"] >= 0.296
    assert uncertainty["ci95_low_vs_comparator"] >= 0.296
    assert uncertainty["valid_rate"] == 1.0
    assert uncertainty_n160["status"] == "pass"
    assert uncertainty_n160["ci95_low_vs_target"] >= 0.35
    assert uncertainty_n160["ci95_low_vs_comparator"] >= 0.35
    assert uncertainty_n160_both["status"] == "pass"
    assert uncertainty_n160_both["ci95_low_vs_target"] >= 0.35
    assert uncertainty_n160_both["ci95_low_vs_comparator"] >= 0.35
    assert learned_receiver["status"] == "pass"
    assert learned_receiver["accuracy"] >= learned_receiver["target_accuracy"] + 0.15
    assert learned_receiver["best_control_accuracy"] <= learned_receiver["target_accuracy"] + 0.05
    assert learned_receiver_8b["status"] == "pass"
    assert learned_receiver_8b["accuracy"] >= learned_receiver_8b["target_accuracy"] + 0.15
    assert learned_receiver_heldout["status"] == "fail"
    assert learned_receiver_heldout["accuracy"] > learned_receiver_heldout["target_accuracy"]
    assert learned_receiver_code_similarity["status"] == "fail"
    assert learned_receiver_anchor_similarity["status"] == "fail"
    assert learned_receiver_anchor_ridge["status"] == "fail"
