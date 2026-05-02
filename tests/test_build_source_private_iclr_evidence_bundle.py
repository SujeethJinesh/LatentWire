from __future__ import annotations

import csv
import json

from scripts import build_source_private_iclr_evidence_bundle as bundle


def test_build_bundle_passes_and_writes_expected_artifacts(tmp_path) -> None:
    payload = bundle.build_bundle(output_dir=tmp_path)

    assert payload["pass_gate"] is True
    assert len(payload["contribution_rows"]) >= 5
    assert len(payload["novelty_matrix"]) >= 8
    assert all(check["pass"] for check in payload["pass_checks"])
    assert (tmp_path / "iclr_evidence_bundle.json").exists()
    assert (tmp_path / "iclr_evidence_bundle.md").exists()
    assert (tmp_path / "novelty_matrix.csv").exists()
    assert (tmp_path / "contribution_matrix.csv").exists()
    assert (tmp_path / "reproduce_iclr_evidence_bundle.sh").exists()
    assert (tmp_path / "manifest.json").exists()


def test_bundle_highlights_source_private_and_systems_axes(tmp_path) -> None:
    payload = bundle.build_bundle(output_dir=tmp_path)
    novelty_rows = {row["comparison"]: row for row in payload["novelty_matrix"]}
    contribution_rows = {row["contribution"]: row for row in payload["contribution_rows"]}

    assert novelty_rows["LatentWire source-private packet"]["source_private"] is True
    assert novelty_rows["C2C cache-to-cache communication"]["requires_model_internals"] is True
    assert "QJL" in novelty_rows["QJL 1-bit sign sketch"]["comparison"]
    assert novelty_rows["Prefix / prompt tuning"]["source_private"] is False
    assert "not a token" in novelty_rows["Prefix / prompt tuning"]["paper_role"]
    assert "1000x" in next(
        check["check"] for check in payload["pass_checks"] if check["check"].endswith("above_1000x")
    )
    assert contribution_rows["Systems byte/KV-cache accounting frontier"]["status"]
    assert contribution_rows["Train-only sender source-prioritized packet builder"]["status"] == (
        "strongest current generalization-facing method"
    )
    assert contribution_rows["Unified train-only train-donor anti-shuffle packet method"]["status"] == (
        "strongest live ICLR method branch"
    )
    assert contribution_rows["Validation-locked train-donor rate frontier"]["status"] == (
        "new reviewer-facing model-selection audit"
    )
    assert contribution_rows["Global stable-gap validation selector"]["status"] == (
        "new strongest model-selection audit"
    )
    assert contribution_rows["ARC-Challenge public bridge contract"]["status"] == (
        "new public-benchmark readiness gate"
    )
    assert contribution_rows["ARC-Challenge fixed-12B public benchmark transfer"]["status"] == (
        "new positive public-benchmark gate"
    )
    assert contribution_rows["ARC-Challenge shared-basis source-computable endpoint"]["status"] == (
        "new strongest public benchmark endpoint"
    )
    assert contribution_rows["OpenBookQA 3B shared-basis second public benchmark"]["status"] == (
        "new second public-benchmark positive gate and stronger rate point"
    )
    assert contribution_rows["OpenBookQA train-only packet/target receiver"]["status"] == (
        "new positive receiver-fusion method gate"
    )
    assert contribution_rows["SciQ text-saturation diagnostic"]["status"] == (
        "documented benchmark limitation, not a promoted headline benchmark"
    )
    assert contribution_rows["CommonsenseQA non-science validation probe"]["status"] == (
        "live non-science diagnostic, not a strict headline benchmark yet"
    )
    assert contribution_rows["HellaSwag 2B non-science adversarial continuation gate"]["status"] == (
        "strong non-science slice, weakened by top-label-copy control"
    )
    assert "5-seed stability=5/5" in contribution_rows[
        "HellaSwag 2B non-science adversarial continuation gate"
    ]["headline_evidence"]
    assert "5B framed record" in contribution_rows[
        "HellaSwag 2B non-science adversarial continuation gate"
    ]["main_metric"]
    assert contribution_rows["HellaSwag label-copy and score-packet headroom diagnostic"]["status"] == (
        "new reviewer-risk diagnostic / branch weakened"
    )
    assert "top-2 oracle heldout" in contribution_rows[
        "HellaSwag label-copy and score-packet headroom diagnostic"
    ]["main_metric"]
    assert contribution_rows["HellaSwag train-only public receiver repair falsification"]["status"] == (
        "new negative method gate / branch pruned"
    )
    assert "best repair minus source-label copy" in contribution_rows[
        "HellaSwag train-only public receiver repair falsification"
    ]["main_metric"]
    assert contribution_rows["HellaSwag train-source-score repair falsification"]["status"] == (
        "new negative source-score method gate / branch weakened"
    )
    assert "selected minus best label-copy" in contribution_rows[
        "HellaSwag train-source-score repair falsification"
    ]["main_metric"]
    assert contribution_rows["HellaSwag source-hidden summary repair falsification"]["status"] == (
        "new negative hidden-summary method gate / branch weakened"
    )
    assert "hidden packet minus source-label copy" in contribution_rows[
        "HellaSwag source-hidden summary repair falsification"
    ]["main_metric"]
    assert contribution_rows["HellaSwag source-hidden innovation repair"]["status"] == (
        "new positive hard-surface method gate feeding anchored stability"
    )
    assert "selected minus best label-copy" in contribution_rows[
        "HellaSwag source-hidden innovation repair"
    ]["main_metric"]
    assert contribution_rows["HellaSwag anchored hidden-innovation split stability"]["status"] == (
        "positive cached split gate, weakened by train-sample stress"
    )
    assert "anchored split seeds pass=5/5" in contribution_rows[
        "HellaSwag anchored hidden-innovation split stability"
    ]["headline_evidence"]
    assert "min delta vs best label-copy" in contribution_rows[
        "HellaSwag anchored hidden-innovation split stability"
    ]["main_metric"]
    assert contribution_rows["HellaSwag hidden-innovation train-sample stress"]["status"] == (
        "new negative robustness gate / HellaSwag branch demoted from headline"
    )
    assert "sample pass map" in contribution_rows[
        "HellaSwag hidden-innovation train-sample stress"
    ]["headline_evidence"]
    assert "mean/min delta vs best label-copy" in contribution_rows[
        "HellaSwag hidden-innovation train-sample stress"
    ]["main_metric"]
    assert contribution_rows["HellaSwag bagged hidden-innovation packet"]["status"] == (
        "new positive third-sample robustness gate / live ICLR method candidate"
    )
    assert "component models" in contribution_rows[
        "HellaSwag bagged hidden-innovation packet"
    ]["headline_evidence"]
    assert "jackknife subbags pass=3/3" in contribution_rows[
        "HellaSwag bagged hidden-innovation packet"
    ]["headline_evidence"]
    assert "delta vs score-only bagged" in contribution_rows[
        "HellaSwag bagged hidden-innovation packet"
    ]["main_metric"]
    assert "jackknife min delta/CI low" in contribution_rows[
        "HellaSwag bagged hidden-innovation packet"
    ]["main_metric"]
    assert contribution_rows["HellaSwag heldout-slice hidden-innovation stress"]["status"] == (
        "new positive frozen heldout-slice gate / HellaSwag headline-candidate"
    )
    assert "slice=1024:2048" in contribution_rows[
        "HellaSwag heldout-slice hidden-innovation stress"
    ]["headline_evidence"]
    assert "delta vs score-only bagged" in contribution_rows[
        "HellaSwag heldout-slice hidden-innovation stress"
    ]["main_metric"]
    assert contribution_rows["HellaSwag multi-slice hidden-innovation stress"]["status"] == (
        "new positive 9-slice gate / stronger HellaSwag headline-candidate"
    )
    assert "slices=9/9 contiguous" in contribution_rows[
        "HellaSwag multi-slice hidden-innovation stress"
    ]["headline_evidence"]
    assert "min delta vs score-only/zero-hidden" in contribution_rows[
        "HellaSwag multi-slice hidden-innovation stress"
    ]["main_metric"]
    assert contribution_rows["HellaSwag anchor-relative common-basis stress"]["status"] == (
        "new negative common-basis diagnostic / HellaSwag branch blocker"
    )
    assert "weighted selected/best-label/score-only" in contribution_rows[
        "HellaSwag anchor-relative common-basis stress"
    ]["headline_evidence"]
    assert "anchor controls below label-copy" in contribution_rows[
        "HellaSwag anchor-relative common-basis stress"
    ]["main_metric"]
    assert contribution_rows["HellaSwag PQ hidden-code branch kill"]["status"] == (
        "new negative hidden-code/codebook gate / HellaSwag receiver-improvement cut"
    )
    assert "default/packet-only" in contribution_rows[
        "HellaSwag PQ hidden-code branch kill"
    ]["headline_evidence"]
    assert "best scout delta/CI95 low" in contribution_rows[
        "HellaSwag PQ hidden-code branch kill"
    ]["main_metric"]
    assert contribution_rows["HellaSwag repair systems acceptance card"]["status"] == (
        "superseded local acceptance row / demoted by later HellaSwag branch-kill gates"
    )
    assert "delta vs source-label copy" in contribution_rows[
        "HellaSwag repair systems acceptance card"
    ]["main_metric"]
    assert contribution_rows["Cross-benchmark source-state byte-floor systems comparator"]["status"] == (
        "new cross-benchmark systems/accounting contribution"
    )
    assert "one-token QJL 1-bit KV floor" in contribution_rows[
        "Cross-benchmark source-state byte-floor systems comparator"
    ]["main_metric"]
    assert contribution_rows["Native vLLM/SGLang systems benchmark plan"]["status"] == (
        "new native systems runbook and acceptance schema"
    )
    assert "TTFT, TPOT, goodput" in contribution_rows[
        "Native vLLM/SGLang systems benchmark plan"
    ]["main_metric"]
    assert contribution_rows["Qwen-hidden to BGE source-latent endpoint diagnostic"]["status"] == (
        "negative diagnostic / branch weakened"
    )
    assert contribution_rows["ARC-Challenge shared-basis systems trace"]["status"] == (
        "new Mac-local systems trace for the public endpoint"
    )
    assert "receiver decode p50/p95" in contribution_rows[
        "ARC-Challenge shared-basis systems trace"
    ]["headline_evidence"]
    assert "seed stability validation/test=5/5 and 5/5" in contribution_rows[
        "ARC-Challenge shared-basis source-computable endpoint"
    ]["headline_evidence"]
    assert contribution_rows["ARC-Challenge projection-seed stability"]["status"] == (
        "new public-benchmark robustness gate"
    )
    assert "test 5/5 seeds pass" in contribution_rows[
        "ARC-Challenge projection-seed stability"
    ]["headline_evidence"]
    assert "test matched/target/text" in contribution_rows[
        "ARC-Challenge fixed-12B public benchmark transfer"
    ]["headline_evidence"]
    assert "1172" in contribution_rows["ARC-Challenge public bridge contract"]["headline_evidence"]
    assert "6/6" in contribution_rows["Global stable-gap validation selector"]["headline_evidence"]
    assert "6/6" in contribution_rows["Validation-locked train-donor rate frontier"]["headline_evidence"]
    assert "6/6" in contribution_rows["Unified train-only train-donor anti-shuffle packet method"]["headline_evidence"]
    assert contribution_rows["Train-only receiver permuted-null gap decoder"]["status"] == (
        "new live receiver-basis method candidate"
    )
    assert "6/6" in contribution_rows["Train-only receiver permuted-null gap decoder"]["headline_evidence"]
    assert "9/9" in contribution_rows["Train-only sender source-prioritized packet builder"]["headline_evidence"]
    assert contribution_rows["Source-prioritized innovation packet builder"]["status"] == (
        "new strict held-out-family positive method"
    )
    assert "9/9" in contribution_rows["Source-prioritized innovation packet builder"]["headline_evidence"]
    assert contribution_rows["Public-disjoint source-to-candidate packet builder"]["status"] == "high-accuracy adaptation row"
    assert contribution_rows["Source-private candidate-local residual packet"]["status"] == "current live positive method"
    assert "6/6" in contribution_rows["Same-family versus cross-family separation gate"]["headline_evidence"]
    assert "0.45-0.48" in contribution_rows["Candidate-local threshold frontier"]["headline_evidence"]
    assert "Procrustes matched/control" in contribution_rows["Candidate-local margin atlas"]["headline_evidence"]
    assert "4887" in contribution_rows["Candidate-local systems boundary trace"]["headline_evidence"]
    assert any(check["check"] == "candidate_local_no_source_text_or_kv" and check["pass"] for check in payload["pass_checks"])
    assert any(
        check["check"] == "candidate_local_threshold_0_48_clean_9_of_9" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "candidate_local_margin_beats_best_control_by_2x" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "train_sender_packet_builder_3_seed_cross_family_pass" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "train_receiver_permuted_null_gap_3_seed_cross_family_pass" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "train_receiver_permuted_null_gap_6_of_6_cross_rows_pass" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "train_sender_packet_builder_rate_has_cross_family_pass" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "source_prioritized_packet_builder_loo_3_seed_cross_family_pass" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "train_donor_antishuffle_3_seed_cross_family_pass" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "train_donor_antishuffle_seed47_n512_cross_family_pass" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "train_donor_antishuffle_3_seed_n512_cross_family_pass" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "train_donor_locked_rate_frontier_6_of_6_selected_eval_rows" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "train_donor_stable_gap_selector_selects_global_12b" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "arc_challenge_bridge_has_official_splits" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "arc_challenge_fixed_packet_test_passes" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "arc_challenge_fixed_packet_test_beats_same_byte_text" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "arc_challenge_seed_stability_test_5_of_5" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "arc_challenge_seed_stability_test_ci_positive" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "arc_challenge_common_basis_test_passes" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "arc_challenge_common_basis_seed_test_5_of_5" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "arc_challenge_anchor_relative_seed_test_5_of_5" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "arc_challenge_anchor_relative_seed_test_beats_same_byte_text" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "arc_challenge_anchor_id_shuffle_test_collapses" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "arc_challenge_anchor_value_shuffle_test_collapses" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "arc_challenge_random_anchors_test_passes" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "arc_challenge_source_latent_endpoint_not_overclaimed" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "arc_challenge_systems_trace_source_private_boundary" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "openbookqa_seed_test_3b_5_of_5" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "openbookqa_seed_test_3b_beats_same_byte_text" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "openbookqa_receiver_headroom_candidate_passes" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "openbookqa_receiver_default_beats_packet_and_target" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert payload["openbookqa_receiver_headroom_headline"]["default_seed_matched"]["receiver_accuracy"] > payload[
        "openbookqa_receiver_headroom_headline"
    ]["default_seed_matched"]["base_accuracy"]
    assert any(
        check["check"] == "sciq_validation_text_saturation_documented" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "commonsenseqa_text_saturation_not_overclaimed" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "commonsenseqa_2b_relaxed_margin_seed_stable" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_bridge_has_labeled_splits" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_fixed_validation1024_2b_passes" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_seed_validation1024_2b_5_of_5" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_seed_validation1024_2b_beats_same_byte_text" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_seed_validation1024_2b_record_is_5b" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_control_suite_metadata_controls_clean" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_control_suite_label_copy_threat_documented" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_score_packet_headroom_not_overclaimed" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_score_packet_top2_oracle_has_method_headroom" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_public_receiver_repair_not_overclaimed" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_public_receiver_repair_top2_headroom_persists" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_train_source_score_repair_not_overclaimed" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_train_source_score_repair_top2_headroom_persists" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_hidden_summary_repair_not_overclaimed" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_hidden_summary_repair_top2_headroom_persists" and check["pass"]
        for check in payload["pass_checks"]
    )
    hidden_headline = payload["hellaswag_hidden_summary_repair_headline"]
    assert hidden_headline["pass_gate"] is False
    assert hidden_headline["selected_layer"] == -1
    assert hidden_headline["selected_packet_raw_bytes"] == 2
    assert hidden_headline["selected_packet_framed_bytes"] == 5
    assert hidden_headline["hidden_packet_minus_source_label_copy"] < 0
    assert hidden_headline["total_wall_time_sec"] > 0
    hidden_innovation_headline = payload["hellaswag_hidden_innovation_repair_headline"]
    assert hidden_innovation_headline["pass_gate"] is True
    assert hidden_innovation_headline["selected_packet_raw_bytes"] == 2
    assert hidden_innovation_headline["selected_packet_framed_bytes"] == 5
    assert hidden_innovation_headline["selected_minus_best_label_copy"] >= 0.02
    assert hidden_innovation_headline["paired_ci95_selected_vs_best_label_copy"]["ci95_low"] > 0.0
    assert hidden_innovation_headline["selected_minus_wrong_example_hidden_control"] > 0.0
    assert hidden_innovation_headline["total_wall_time_sec"] > 0
    stability_headline = payload["hellaswag_hidden_innovation_stability_headline"]
    assert stability_headline["pass_gate"] is True
    assert stability_headline["pass_count"] == 5
    assert stability_headline["split_seed_count"] == 5
    assert stability_headline["selected_view_counts"] == {"score_hidden_residual": 5}
    assert stability_headline["selected_packet_raw_bytes"] == 2
    assert stability_headline["selected_packet_framed_bytes"] == 5
    assert stability_headline["delta_vs_best_label_copy_min"] >= 0.02
    assert stability_headline["paired_ci95_low_vs_best_label_copy_min"] > 0.0
    assert stability_headline["unrestricted_diagnostic"]["pass_count"] < 5
    assert stability_headline["total_wall_time_sec"] > 0
    train_sample_stress = payload["hellaswag_hidden_innovation_train_sample_stress_headline"]
    assert train_sample_stress["pass_gate"] is False
    assert train_sample_stress["new_train_sample_seed_count"] == 1
    assert train_sample_stress["sample_pass"] == {"1729": True, "2027": False}
    assert train_sample_stress["pass_count"] == 5
    assert train_sample_stress["split_rows"] == 6
    assert train_sample_stress["selected_packet_raw_bytes"] == 2
    assert train_sample_stress["selected_packet_framed_bytes"] == 5
    assert train_sample_stress["delta_vs_best_label_copy_min"] < 0.0
    assert train_sample_stress["total_wall_time_sec"] > 0
    bagged_headline = payload["hellaswag_hidden_innovation_bagged_gate_headline"]
    assert bagged_headline["pass_gate"] is True
    assert bagged_headline["component_model_count"] == 9
    assert bagged_headline["new_train_sample_seed_count"] == 2
    assert bagged_headline["jackknife_summary"]["pass_count"] == 3
    assert bagged_headline["jackknife_summary"]["row_count"] == 3
    assert bagged_headline["selected_packet_raw_bytes"] == 2
    assert bagged_headline["selected_packet_framed_bytes"] == 5
    assert bagged_headline["selected_minus_best_label_copy"] >= 0.02
    assert bagged_headline["selected_minus_score_only_bagged_control"] >= 0.02
    assert bagged_headline["selected_minus_zero_hidden_control"] >= 0.02
    assert bagged_headline["paired_ci95_low_vs_best_label_copy"] > 0.0
    assert bagged_headline["total_wall_time_sec"] > 0
    eval_slice_headline = payload["hellaswag_hidden_innovation_eval_slice_stress_headline"]
    assert eval_slice_headline["pass_gate"] is True
    assert eval_slice_headline["eval_slice_start"] == 1024
    assert eval_slice_headline["eval_slice_end_exclusive"] == 2048
    assert eval_slice_headline["eval_rows"] == 1024
    assert eval_slice_headline["selected_minus_best_label_copy"] >= 0.02
    assert eval_slice_headline["paired_ci95_low_vs_best_label_copy"] > 0.0
    assert eval_slice_headline["selected_minus_score_only_bagged_control"] >= 0.02
    assert eval_slice_headline["jackknife_pass_count"] == eval_slice_headline["jackknife_row_count"]
    assert eval_slice_headline["raw_payload_bytes"] == 2
    assert eval_slice_headline["framed_record_bytes"] == 5
    eval_slice_2048_headline = payload["hellaswag_hidden_innovation_eval_slice_stress_2048_3072_headline"]
    assert eval_slice_2048_headline["pass_gate"] is True
    assert eval_slice_2048_headline["eval_slice_start"] == 2048
    assert eval_slice_2048_headline["eval_slice_end_exclusive"] == 3072
    assert eval_slice_2048_headline["selected_minus_best_label_copy"] >= 0.02
    multi_slice_headline = payload["hellaswag_hidden_innovation_multi_slice_stress_headline"]
    assert multi_slice_headline["pass_gate"] is True
    assert multi_slice_headline["slice_count"] == 9
    assert multi_slice_headline["pass_slice_count"] == 9
    assert multi_slice_headline["total_eval_rows"] == 9216
    assert multi_slice_headline["contiguous_validation_prefix"] is True
    assert multi_slice_headline["min_delta_vs_best_label_copy"] >= 0.02
    assert multi_slice_headline["min_delta_vs_score_only_bagged"] >= 0.02
    assert multi_slice_headline["source_private_packet"] is True
    assert len(multi_slice_headline["slice_artifacts"]) == 9
    acceptance_headline = payload["hellaswag_repair_systems_acceptance_headline"]
    assert acceptance_headline["pass_gate"] is True
    assert acceptance_headline["rows"] == 7
    assert acceptance_headline["method_gate_pass"] is True
    assert acceptance_headline["systems_audit_pass"] is True
    assert acceptance_headline["native_queue_allowed"] is False
    assert acceptance_headline["best_repair_row_id"] == "hidden_innovation_repair"
    assert acceptance_headline["best_delta_vs_source_label_copy"] >= 0.02
    assert acceptance_headline["trained_label_copy_control_rows"] >= 1
    assert acceptance_headline["best_delta_vs_trained_label_copy"] >= 0.02
    assert any(
        check["check"] == "hellaswag_hidden_innovation_repair_passes" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_hidden_innovation_repair_beats_label_copy" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_hidden_innovation_repair_controls_collapse" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_hidden_innovation_repair_source_private_packet" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_hidden_innovation_stability_passes" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_hidden_innovation_stability_5_of_5_cached_splits" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_hidden_innovation_stability_controls_collapse" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_hidden_innovation_stability_unrestricted_selector_not_overclaimed"
        and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_hidden_innovation_stability_source_private_packet" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_hidden_innovation_train_sample_stress_not_overclaimed" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_hidden_innovation_train_sample_stress_records_fresh_sample" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_hidden_innovation_train_sample_stress_source_private_packet" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_hidden_innovation_bagged_gate_passes" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_hidden_innovation_bagged_gate_beats_label_and_score_only" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_hidden_innovation_bagged_gate_controls_collapse" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_hidden_innovation_bagged_gate_uses_fresh_train_sample" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_hidden_innovation_bagged_gate_jackknife_3_of_3" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_hidden_innovation_bagged_gate_source_private_packet" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_hidden_innovation_eval_slice_stress_passes" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_hidden_innovation_eval_slice_stress_beats_label_and_score_only"
        and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_hidden_innovation_eval_slice_stress_controls_and_jackknife"
        and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_hidden_innovation_multi_slice_stress_passes" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_hidden_innovation_multi_slice_stress_has_3_contiguous_slices"
        and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_hidden_innovation_multi_slice_stress_has_4_contiguous_slices"
        and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_hidden_innovation_multi_slice_stress_has_5_contiguous_slices"
        and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_hidden_innovation_multi_slice_stress_beats_label_score_zero"
        and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_hidden_innovation_multi_slice_stress_controls_and_jackknife"
        and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_hidden_innovation_multi_slice_stress_source_private_packet"
        and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_anchor_relative_hidden_innovation_multi_slice_recorded"
        and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_anchor_relative_hidden_innovation_common_basis_demoted"
        and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_anchor_relative_hidden_innovation_source_private_packet"
        and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_pq_hidden_innovation_codec_not_overclaimed" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_pq_hidden_innovation_codec_source_private_packet" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_pq_hidden_innovation_codec_controls_collapse" and check["pass"]
        for check in payload["pass_checks"]
    )
    pq_headline = payload["hellaswag_pq_hidden_innovation_codec_headline"]
    assert pq_headline["pass_gate"] is False
    assert pq_headline["default_delta_vs_packet_only"] < 0
    assert pq_headline["best_scout_delta_vs_packet_only"] < 0.010
    assert pq_headline["packet_contract"]["raw_payload_bytes"] == 1
    assert pq_headline["systems_packet_sideband"]["native_gpu_claims_allowed"] is False
    assert any(
        check["check"] == "hellaswag_repair_systems_acceptance_card_passes" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_repair_systems_acceptance_method_promoted" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_repair_systems_acceptance_trained_label_control_clears" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_repair_systems_acceptance_audit_passes" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_repair_systems_acceptance_native_queue_blocked" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "hellaswag_repair_systems_acceptance_has_byte_latency_exposure_fields" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "mac_packet_ring_transport_passes" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "cross_benchmark_systems_comparator_passes" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "cross_benchmark_systems_qjl_floor_above_50x" and check["pass"]
        for check in payload["pass_checks"]
    )
    cross_headline = payload["cross_benchmark_systems_comparator_headline"]
    assert cross_headline["headline_eligible_benchmarks"] >= 2
    assert cross_headline["native_systems_complete"] is False
    assert any(
        check["check"] == "native_systems_benchmark_plan_passes" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "native_systems_benchmark_plan_nonclaim" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "native_systems_benchmark_plan_has_required_baselines" and check["pass"]
        for check in payload["pass_checks"]
    )
    assert any(
        check["check"] == "native_systems_benchmark_plan_has_hardware_metrics" and check["pass"]
        for check in payload["pass_checks"]
    )
    native_plan_headline = payload["native_systems_benchmark_plan_headline"]
    assert native_plan_headline["native_systems_complete"] is False
    assert native_plan_headline["required_baseline_count"] >= 10
    assert native_plan_headline["required_metric_count"] >= 20
    assert native_plan_headline["serving_substrates"] == ["vLLM", "SGLang"]


def test_written_bundle_is_valid_json_and_csv(tmp_path) -> None:
    bundle.build_bundle(output_dir=tmp_path)
    summary = json.loads((tmp_path / "iclr_evidence_bundle.json").read_text(encoding="utf-8"))
    with (tmp_path / "novelty_matrix.csv").open(encoding="utf-8", newline="") as handle:
        novelty_rows = list(csv.DictReader(handle))

    assert summary["pass_gate"] is True
    assert novelty_rows
    assert "comparison" in novelty_rows[0]
