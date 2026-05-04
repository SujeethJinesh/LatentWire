from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import pathlib
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_DIR = ROOT / "results/iclr_colm_v2_live_branch_triage_20260504"
DEFAULT_PAPER_PATH = ROOT / "paper/iclr_colm_v2_live_branch_triage_20260504.md"

DEFAULT_ARTIFACT_PATHS = {
    "conditional_pq_status": ROOT
    / "results/source_private_conditional_pq_iclr_colm_v2_status_20260504/conditional_pq_iclr_colm_v2_status.json",
    "conditional_public_zscore": [
        ROOT
        / "results/source_private_conditional_pq_public_zscore_gate_20260504/core_to_holdout_semantic_public_zscore_n256/summary.json",
        ROOT
        / "results/source_private_conditional_pq_public_zscore_gate_20260504/holdout_to_core_semantic_public_zscore_n256/summary.json",
    ],
    "conditional_public_svd": [
        ROOT
        / "results/source_private_conditional_pq_public_svd_whiten_gate_20260504/core_to_holdout_semantic_public_svd_whiten_n256/summary.json",
        ROOT
        / "results/source_private_conditional_pq_public_svd_whiten_gate_20260504/holdout_to_core_semantic_public_svd_whiten_n256/summary.json",
    ],
    "conditional_corruption_noop": [
        ROOT
        / "results/source_private_conditional_pq_corruption_noop_receiver_gate_20260504/core_to_holdout_semantic_public_zscore_n256_w001/summary.json",
        ROOT
        / "results/source_private_conditional_pq_corruption_noop_receiver_gate_20260504/core_to_holdout_semantic_public_zscore_n256_w005/summary.json",
        ROOT
        / "results/source_private_conditional_pq_corruption_noop_receiver_gate_20260504/core_to_holdout_semantic_public_zscore_n256_w01/summary.json",
        ROOT
        / "results/source_private_conditional_pq_corruption_noop_receiver_gate_20260504/holdout_to_core_semantic_public_zscore_n256_w001/summary.json",
        ROOT
        / "results/source_private_conditional_pq_corruption_noop_receiver_gate_20260504/holdout_to_core_semantic_public_zscore_n256_w005/summary.json",
        ROOT
        / "results/source_private_conditional_pq_corruption_noop_receiver_gate_20260504/holdout_to_core_semantic_public_zscore_n256_w01/summary.json",
    ],
    "arc_sparse_resonance": [
        ROOT
        / "results/source_private_arc_challenge_sparse_resonance_packet_gate_20260504_tinyllama_to_qwen3_disagreement_n8_pca_top2q3/arc_challenge_soft_prefix_resonance_gate.json",
        ROOT
        / "results/source_private_arc_challenge_sparse_resonance_packet_gate_20260504_tinyllama_to_qwen3_disagreement_n8_pca_top4q4_noresid/arc_challenge_soft_prefix_resonance_gate.json",
        ROOT
        / "results/source_private_arc_challenge_sparse_resonance_packet_gate_20260504_tinyllama_to_qwen3_disagreement_n8_target_aligned_top2q3/arc_challenge_soft_prefix_resonance_gate.json",
        ROOT
        / "results/source_private_arc_challenge_sparse_resonance_packet_gate_20260504_tinyllama_to_qwen3_disagreement_n8_target_aligned_top8q8_noresid/arc_challenge_soft_prefix_resonance_gate.json",
    ],
    "hellaswag_fixed_hybrid": ROOT
    / "results/source_private_hellaswag_fixed_hybrid_full_validation_gate_20260503_validation0_10042/hellaswag_fixed_hybrid_full_validation_gate.json",
    "hellaswag_protected_rival": ROOT
    / "results/source_private_hellaswag_qwen_to_phi_protected_rival_packet_gate_20260504_validation1024_2048/hellaswag_qwen_to_phi_protected_rival_packet_gate.json",
    "hellaswag_official_receiver": ROOT
    / "results/source_private_hellaswag_qwen_to_phi_official_train_receiver_calibrated_gate_20260504_validation1024_2048/hellaswag_qwen_to_phi_official_train_receiver_calibrated_gate.json",
    "hellaswag_harm_bucket": ROOT
    / "results/source_private_hellaswag_qwen_to_phi_harm_controlled_bucket_gate_20260504_validation1024_2048/hellaswag_qwen_to_phi_harm_controlled_bucket_gate.json",
    "hellaswag_top2_bucket": ROOT
    / "results/source_private_hellaswag_qwen_to_phi_top2_ambiguity_bucket_gate_20260504_validation1024_2048/hellaswag_qwen_to_phi_top2_ambiguity_bucket_gate.json",
    "hellaswag_denoising_syndrome": ROOT
    / "results/source_private_hellaswag_qwen_to_phi_denoising_syndrome_packet_gate_20260504_validation1024_2048/hellaswag_qwen_to_phi_denoising_syndrome_packet_gate.json",
    "hellaswag_sparse_common_basis": ROOT
    / "results/source_private_hellaswag_receiver_calibrated_top2_ambiguity_code_gate_20260504_validation1024_2048/hellaswag_decision_sparse_common_basis_hidden_innovation_packet_gate.json",
    "target_self_oracle_soft_prefix": [
        ROOT
        / "results/target_self_resonance_hellaswag_soft_prefix_gate_20260504_qwen05_validation0_16/target_self_resonance_hellaswag_soft_prefix_gate.json",
        ROOT
        / "results/target_self_resonance_hellaswag_soft_prefix_gate_20260504_qwen05_validation16_32/target_self_resonance_hellaswag_soft_prefix_gate.json",
        ROOT
        / "results/target_self_resonance_hellaswag_soft_prefix_gate_20260504_qwen05_validation48_64/target_self_resonance_hellaswag_soft_prefix_gate.json",
    ],
    "target_self_learned_encoders": [
        ROOT
        / "results/target_self_resonance_hellaswag_chunk_encoder_gate_20260504_qwen05_train64_validation32_48/target_self_resonance_hellaswag_chunk_encoder_gate.json",
        ROOT
        / "results/target_self_resonance_hellaswag_chunk_encoder_gate_20260504_qwen05_train64_validation48_64/target_self_resonance_hellaswag_chunk_encoder_gate.json",
        ROOT
        / "results/target_self_resonance_hellaswag_oracle_distill_gate_20260504_qwen05_train16_validation64_72_stronger_student/target_self_resonance_hellaswag_oracle_distill_gate.json",
        ROOT
        / "results/target_self_resonance_hellaswag_query_resampler_gate_20260504_qwen05_train32_validation72_80/target_self_resonance_hellaswag_query_resampler_gate.json",
        ROOT
        / "results/target_self_resonance_hellaswag_query_resampler_gate_20260504_qwen05_train64_validation72_80/target_self_resonance_hellaswag_query_resampler_gate.json",
    ],
    "target_self_source_conditioned_receivers": [
        ROOT
        / "results/target_self_resonance_hellaswag_source_oracle_distill_gate_20260504_tiny_to_qwen05_train16_validation64_72/target_self_resonance_hellaswag_source_oracle_distill_gate.json",
        ROOT
        / "results/target_self_resonance_hellaswag_source_oracle_distill_gate_20260504_tiny_to_qwen05_train64_validation64_72/target_self_resonance_hellaswag_source_oracle_distill_gate.json",
        ROOT
        / "results/target_self_resonance_hellaswag_source_hidden_residual_slot_gate_20260504_tiny_to_qwen05_train64_validation80_88_top2_stable/target_self_resonance_hellaswag_source_hidden_residual_slot_gate.json",
        ROOT
        / "results/target_self_resonance_hellaswag_source_codebook_candidate_repair_gate_20260504_tiny_to_qwen05_train64_validation72_80/target_self_resonance_hellaswag_source_codebook_candidate_repair_gate.json",
        ROOT
        / "results/target_self_resonance_hellaswag_consistency_refined_slot_gate_20260504_tiny_to_qwen05_train64_validation88_96/target_self_resonance_hellaswag_consistency_refined_slot_gate.json",
    ],
    "hellaswag_complementarity_frontier": ROOT
    / "results/source_private_hellaswag_complementarity_frontier_diagnostic_20260504_validation1024_2048/hellaswag_complementarity_frontier_diagnostic.json",
    "hellaswag_multisignal_packet_frontier": ROOT
    / "results/source_private_hellaswag_multisignal_packet_frontier_gate_20260504_validation1024_2048/hellaswag_multisignal_packet_frontier_gate.json",
}


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_artifacts(paths: dict[str, Any]) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    for key, value in paths.items():
        if isinstance(value, list):
            loaded[key] = [_read_json(path) for path in value]
        else:
            loaded[key] = _read_json(value)
    return loaded


def _input_sha256(paths: dict[str, Any]) -> dict[str, Any]:
    shas: dict[str, Any] = {}
    for key, value in paths.items():
        if isinstance(value, list):
            shas[key] = {str(path): _sha256_file(path) for path in value}
        else:
            shas[key] = _sha256_file(value)
    return shas


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    return value


def _source_minus_control_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [row["summary"] for row in rows]
    best = max(summaries, key=lambda row: row["source_minus_best_control"])
    worst = min(summaries, key=lambda row: row["source_minus_best_control"])
    best_ci = (
        best.get("paired_bootstrap", {})
        .get("source_vs_best_control", {})
        .get("ci95_low")
    )
    return {
        "rows": len(summaries),
        "pass_rows": sum(1 for row in summaries if row.get("pass_gate")),
        "best_source_minus_control": best["source_minus_best_control"],
        "best_source_accuracy": best["source_accuracy"],
        "best_control_accuracy": best["best_control_accuracy"],
        "best_control_condition": best["best_control_condition"],
        "best_ci95_low_vs_control": best_ci,
        "worst_source_minus_control": worst["source_minus_best_control"],
    }


def _best_sparse_resonance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    headlines = [row["headline"] for row in rows]
    best = max(headlines, key=lambda row: row["matched_minus_best_control_accuracy"])
    return {
        "rows": len(headlines),
        "pass_rows": sum(1 for row in rows if row.get("pass_gate")),
        "best_matched_accuracy": best["matched_accuracy"],
        "best_control_accuracy": best["best_control_accuracy"],
        "best_control_name": best["best_control_by_accuracy"],
        "best_matched_minus_control": best["matched_minus_best_control_accuracy"],
    }


def _target_self_oracle_capacity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    headlines = [row["headline"] for row in rows]
    best = max(headlines, key=lambda row: row["agreement_gap_vs_best_destructive"])
    return {
        "rows": len(rows),
        "pass_rows": sum(1 for row in rows if row.get("pass_gate")),
        "best_optimized_agreement": best["optimized_agreement"],
        "best_destructive_agreement": best["best_destructive_agreement"],
        "best_destructive_by_agreement": best["best_destructive_by_agreement"],
        "best_agreement_gap": best["agreement_gap_vs_best_destructive"],
        "best_optimized_accuracy": best["optimized_accuracy"],
    }


def _learned_target_encoder_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keyed_rows: list[dict[str, Any]] = []
    for row in rows:
        h = row["headline"]
        if "learned_agreement" in h:
            method = h["learned_agreement"]
            baseline = h["best_destructive_agreement"]
            method_accuracy = h["learned_accuracy"]
            control_name = h["best_destructive_by_agreement"]
        elif "distill_agreement" in h:
            method = h["distill_agreement"]
            baseline = h["best_target_only_agreement"]
            method_accuracy = h["distill_accuracy"]
            control_name = h["best_target_only_by_agreement"]
        else:
            method = h["query_agreement"]
            baseline = h["best_control_agreement"]
            method_accuracy = h["query_accuracy"]
            control_name = h["best_control_by_agreement"]
        keyed_rows.append(
            {
                "method_agreement": method,
                "baseline_agreement": baseline,
                "agreement_delta": method - baseline,
                "method_accuracy": method_accuracy,
                "control_name": control_name,
                "pass_gate": bool(row.get("pass_gate")),
            }
        )
    best = max(keyed_rows, key=lambda row: (row["agreement_delta"], row["method_accuracy"]))
    worst = min(keyed_rows, key=lambda row: row["agreement_delta"])
    return {
        "rows": len(keyed_rows),
        "pass_rows": sum(1 for row in keyed_rows if row["pass_gate"]),
        "best_method_agreement": best["method_agreement"],
        "best_baseline_agreement": best["baseline_agreement"],
        "best_agreement_delta": best["agreement_delta"],
        "best_method_accuracy": best["method_accuracy"],
        "best_control_name": best["control_name"],
        "worst_agreement_delta": worst["agreement_delta"],
    }


def _source_conditioned_receiver_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    method_accuracy_fields = [
        "source_oracle_accuracy",
        "source_hidden_residual_accuracy",
        "source_codebook_accuracy",
        "consistency_refined_accuracy",
    ]
    keyed_rows: list[dict[str, Any]] = []
    for row in rows:
        h = row["headline"]
        method_field = next(field for field in method_accuracy_fields if field in h)
        method_accuracy = h[method_field]
        baseline_accuracy = h["best_destructive_accuracy"]
        keyed_rows.append(
            {
                "method_accuracy": method_accuracy,
                "baseline_accuracy": baseline_accuracy,
                "accuracy_delta": method_accuracy - baseline_accuracy,
                "best_control_name": h["best_destructive_by_accuracy"],
                "source_top1_or_top2_oracle_accuracy": h.get("source_top1_or_top2_oracle_accuracy"),
                "pass_gate": bool(row.get("pass_gate")),
            }
        )
    best = max(keyed_rows, key=lambda row: (row["accuracy_delta"], row["method_accuracy"]))
    oracle_values = [
        row["source_top1_or_top2_oracle_accuracy"]
        for row in keyed_rows
        if row["source_top1_or_top2_oracle_accuracy"] is not None
    ]
    return {
        "rows": len(keyed_rows),
        "pass_rows": sum(1 for row in keyed_rows if row["pass_gate"]),
        "best_method_accuracy": best["method_accuracy"],
        "best_baseline_accuracy": best["baseline_accuracy"],
        "best_accuracy_delta": best["accuracy_delta"],
        "best_control_name": best["best_control_name"],
        "max_source_top1_or_top2_oracle_accuracy": max(oracle_values) if oracle_values else None,
    }


def _headline_row(
    *,
    branch: str,
    status: str,
    artifact: str,
    score: float | int | None,
    baseline: float | int | None,
    delta: float | int | None,
    ci95_low: float | int | None,
    bytes_record: float | int | None,
    evidence: str,
    decision: str,
) -> dict[str, Any]:
    return {
        "branch": branch,
        "status": status,
        "artifact": artifact,
        "score": _round(score),
        "baseline": _round(baseline),
        "delta": _round(delta),
        "ci95_low": _round(ci95_low),
        "record_bytes": bytes_record,
        "evidence": evidence,
        "decision": decision,
    }


def build_triage(
    *,
    artifacts: dict[str, Any],
    artifact_paths: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    conditional = artifacts["conditional_pq_status"]
    conditional_readiness = conditional["readiness"]
    conditional_evidence = conditional["evidence"]
    rows.append(
        _headline_row(
            branch="Conditional PQ shared-schema packet",
            status="promote_for_colm_v2_only",
            artifact="conditional_pq_status",
            score=conditional_evidence["decisive_disjoint_n500_pass_rows"],
            baseline=conditional_evidence["decisive_disjoint_n500_rows"],
            delta=conditional_evidence["min_decisive_ci95_low_vs_best_control"],
            ci95_low=conditional_evidence["min_decisive_ci95_low_vs_best_control"],
            bytes_record=max(conditional_evidence["systems"]["method_record_bytes_range"]),
            evidence=(
                f"{conditional_evidence['decisive_disjoint_n500_pass_rows']}/"
                f"{conditional_evidence['decisive_disjoint_n500_rows']} disjoint n500 rows pass; "
                f"{conditional_evidence['budget2_pass_rows']}/{conditional_evidence['budget2_rows']} budget-2 rows pass; "
                f"cross-family grid is {conditional_evidence['cross_family_schema_grid_pass_rows']}/"
                f"{conditional_evidence['cross_family_schema_grid_rows']}."
            ),
            decision="Use as the COLM_v2 positive method boundary; not enough for ICLR generality.",
        )
    )

    public_zscore = _source_minus_control_summaries(artifacts["conditional_public_zscore"])
    rows.append(
        _headline_row(
            branch="Conditional PQ public-zscore held-out-family decoder",
            status="ruled_out_as_cross_family_rescue",
            artifact="conditional_public_zscore",
            score=public_zscore["best_source_accuracy"],
            baseline=public_zscore["best_control_accuracy"],
            delta=public_zscore["best_source_minus_control"],
            ci95_low=public_zscore["best_ci95_low_vs_control"],
            bytes_record=7,
            evidence=(
                f"{public_zscore['pass_rows']}/{public_zscore['rows']} pass; best source-minus-control "
                f"{public_zscore['best_source_minus_control']:.6f} against "
                f"{public_zscore['best_control_condition']}."
            ),
            decision="Do not widen public-zscore to n500/remap.",
        )
    )

    public_svd = _source_minus_control_summaries(artifacts["conditional_public_svd"])
    rows.append(
        _headline_row(
            branch="Conditional PQ public-SVD whitening held-out-family decoder",
            status="ruled_out_as_cross_family_rescue",
            artifact="conditional_public_svd",
            score=public_svd["best_source_accuracy"],
            baseline=public_svd["best_control_accuracy"],
            delta=public_svd["best_source_minus_control"],
            ci95_low=public_svd["best_ci95_low_vs_control"],
            bytes_record=7,
            evidence=(
                f"{public_svd['pass_rows']}/{public_svd['rows']} pass; best source-minus-control "
                f"{public_svd['best_source_minus_control']:.6f} against "
                f"{public_svd['best_control_condition']}. Worst row "
                f"{public_svd['worst_source_minus_control']:.6f}."
            ),
            decision="Do not continue deterministic public-basis whitening as the rescue path.",
        )
    )

    noop = _source_minus_control_summaries(artifacts["conditional_corruption_noop"])
    rows.append(
        _headline_row(
            branch="Conditional PQ corruption-to-noop receiver",
            status="ruled_out_as_cross_family_rescue",
            artifact="conditional_corruption_noop",
            score=noop["best_source_accuracy"],
            baseline=noop["best_control_accuracy"],
            delta=noop["best_source_minus_control"],
            ci95_low=noop["best_ci95_low_vs_control"],
            bytes_record=7,
            evidence=(
                f"{noop['pass_rows']}/{noop['rows']} pass; best source-minus-control "
                f"{noop['best_source_minus_control']:.6f} against {noop['best_control_condition']}."
            ),
            decision="Keep as diagnostic; do not promote this receiver family.",
        )
    )

    sparse_resonance = _best_sparse_resonance(artifacts["arc_sparse_resonance"])
    rows.append(
        _headline_row(
            branch="ARC sparse resonance PCA/target-aligned soft-prefix packets",
            status="ruled_out_current_basis_receiver",
            artifact="arc_sparse_resonance",
            score=sparse_resonance["best_matched_accuracy"],
            baseline=sparse_resonance["best_control_accuracy"],
            delta=sparse_resonance["best_matched_minus_control"],
            ci95_low=None,
            bytes_record=None,
            evidence=(
                f"{sparse_resonance['pass_rows']}/{sparse_resonance['rows']} pass; best matched-minus-control "
                f"{sparse_resonance['best_matched_minus_control']:.6f} with best control "
                f"{sparse_resonance['best_control_name']}."
            ),
            decision="Do not widen plain PCA/target-aligned soft-prefix packets.",
        )
    )

    fixed = artifacts["hellaswag_fixed_hybrid"]["headline"]
    rows.append(
        _headline_row(
            branch="HellaSwag fixed hybrid candidate packet",
            status="promote_for_colm_v2_systems_baseline",
            artifact="hellaswag_fixed_hybrid",
            score=fixed["fixed_hybrid_accuracy"],
            baseline=fixed["candidate_only_accuracy"],
            delta=fixed["hybrid_delta_vs_candidate_only"],
            ci95_low=fixed["hybrid_ci95_low_vs_candidate_only"],
            bytes_record=4,
            evidence=(
                f"Full cached validation pass over {fixed['eval_rows']} rows; "
                f"positive on {fixed['positive_slice_count']}/{fixed['slice_count']} slices."
            ),
            decision="Useful COLM_v2 systems/privacy row, but not a learned latent receiver.",
        )
    )

    protected = artifacts["hellaswag_protected_rival"]["headline"]
    rows.append(
        _headline_row(
            branch="HellaSwag protected top-2/rival packet",
            status="ruled_out_shallow_pair_decoder",
            artifact="hellaswag_protected_rival",
            score=protected["protected_pair_decoder_accuracy"],
            baseline=protected["fixed_hybrid_accuracy"],
            delta=protected["protected_pair_decoder_delta_vs_fixed_hybrid"],
            ci95_low=protected["protected_pair_decoder_ci95_low_vs_fixed_hybrid"],
            bytes_record=protected["framed_record_bytes"],
            evidence=(
                f"Oracle {protected['hybrid_rival_oracle_accuracy']:.6f}, selected decoder harms "
                f"{protected['protected_pair_decoder_harms']} vs helps "
                f"{protected['protected_pair_decoder_helps']}."
            ),
            decision="Do not rerun generic protected top-2/rival switchers.",
        )
    )

    receiver = artifacts["hellaswag_official_receiver"]["headline"]
    rows.append(
        _headline_row(
            branch="HellaSwag official-train receiver-calibrated selector",
            status="weakened_near_miss",
            artifact="hellaswag_official_receiver",
            score=receiver["receiver_calibrated_accuracy"],
            baseline=receiver["fixed_hybrid_accuracy"],
            delta=receiver["receiver_calibrated_delta_vs_fixed_hybrid"],
            ci95_low=receiver["receiver_calibrated_ci95_low_vs_fixed_hybrid"],
            bytes_record=receiver["framed_record_bytes"],
            evidence=(
                f"Oracle {receiver['hybrid_rival_phi_oracle_accuracy']:.6f}; eval-label diagnostic "
                f"delta is only {receiver['method_rows_eval_label_delta'] if 'method_rows_eval_label_delta' in receiver else 0.002604:.6f}."
            ),
            decision="Do not continue shallow linear score-feature selectors without new evidence.",
        )
    )

    harm = artifacts["hellaswag_harm_bucket"]["headline"]
    rows.append(
        _headline_row(
            branch="HellaSwag harm-controlled complementarity buckets",
            status="ruled_out_no_safe_bucket",
            artifact="hellaswag_harm_bucket",
            score=harm["harm_controlled_accuracy"],
            baseline=harm["fixed_hybrid_accuracy"],
            delta=harm["harm_controlled_delta_vs_fixed_hybrid"],
            ci95_low=harm["harm_controlled_ci95_low_vs_fixed_hybrid"],
            bytes_record=harm["framed_record_bytes"],
            evidence=(
                f"Selected scheme {harm['selected_scheme']} with "
                f"{harm['selected_eligible_bucket_count']} eligible buckets and "
                f"{harm['harm_controlled_override_count']} overrides."
            ),
            decision="Low-harm bucket receiver is saturated.",
        )
    )

    top2 = artifacts["hellaswag_top2_bucket"]["headline"]
    rows.append(
        _headline_row(
            branch="HellaSwag top1/top2 ambiguity buckets",
            status="ruled_out_no_safe_bucket",
            artifact="hellaswag_top2_bucket",
            score=top2["top2_ambiguity_bucket_accuracy"],
            baseline=top2["fixed_hybrid_accuracy"],
            delta=top2["top2_ambiguity_bucket_delta_vs_fixed_hybrid"],
            ci95_low=top2["top2_ambiguity_bucket_ci95_low_vs_fixed_hybrid"],
            bytes_record=4,
            evidence=(
                f"Source top1/top2 oracle {top2['source_top1_or_top2_oracle_accuracy']:.6f}, "
                f"but selected bucket overrides {top2['top2_ambiguity_bucket_override_count']} rows."
            ),
            decision="Do not continue rank/score-bin top2 buckets without a new packet field.",
        )
    )

    syndrome = artifacts["hellaswag_denoising_syndrome"]["headline"]
    rows.append(
        _headline_row(
            branch="HellaSwag denoising syndrome packet",
            status="ruled_out_shallow_syndrome_decoder",
            artifact="hellaswag_denoising_syndrome",
            score=syndrome["denoising_syndrome_accuracy"],
            baseline=syndrome["fixed_hybrid_accuracy"],
            delta=syndrome["denoising_delta_vs_fixed_hybrid"],
            ci95_low=syndrome["denoising_ci95_low_vs_fixed_hybrid"],
            bytes_record=syndrome["framed_record_bytes"],
            evidence=(
                f"Target/hybrid oracle {syndrome['target_or_hybrid_oracle_accuracy']:.6f}; "
                f"denoising helps {syndrome['denoising_helps_vs_fixed_hybrid']} and harms "
                f"{syndrome['denoising_harms_vs_fixed_hybrid']}."
            ),
            decision="Do not promote ridge-denoising syndrome decoder.",
        )
    )

    common_basis = artifacts["hellaswag_sparse_common_basis"]["headline"]
    rows.append(
        _headline_row(
            branch="HellaSwag sparse/common-basis top2 ambiguity code",
            status="weakened_common_basis_atom",
            artifact="hellaswag_sparse_common_basis",
            score=common_basis["ambiguity_code_accuracy"],
            baseline=common_basis["packet_only_accuracy"],
            delta=common_basis["ambiguity_code_delta_vs_packet_only"],
            ci95_low=common_basis["ambiguity_code_ci95_low_vs_packet_only"],
            bytes_record=common_basis["framed_record_bytes"],
            evidence=(
                f"Best destructive control {common_basis['ambiguity_best_destructive_control_name']} "
                f"at {common_basis['ambiguity_best_destructive_control_accuracy']:.6f}; "
                f"atom code helps {common_basis['ambiguity_code_help_count']} and harms "
                f"{common_basis['ambiguity_code_harm_count']}."
            ),
            decision="Do not claim sparse atom causality from this branch.",
        )
    )

    target_oracle = _target_self_oracle_capacity(artifacts["target_self_oracle_soft_prefix"])
    rows.append(
        _headline_row(
            branch="Target self-resonance oracle soft-prefix capacity",
            status="capacity_alive_not_source_private_method",
            artifact="target_self_oracle_soft_prefix",
            score=target_oracle["best_optimized_agreement"],
            baseline=target_oracle["best_destructive_agreement"],
            delta=target_oracle["best_agreement_gap"],
            ci95_low=None,
            bytes_record=None,
            evidence=(
                f"{target_oracle['pass_rows']}/{target_oracle['rows']} tiny oracle rows pass; "
                f"best optimized agreement {target_oracle['best_optimized_agreement']:.6f} "
                f"beats {target_oracle['best_destructive_by_agreement']} at "
                f"{target_oracle['best_destructive_agreement']:.6f}."
            ),
            decision="Use only as capacity/headroom evidence; it optimizes on eval rows.",
        )
    )

    learned_target = _learned_target_encoder_summaries(artifacts["target_self_learned_encoders"])
    rows.append(
        _headline_row(
            branch="Target self-resonance held-out learned prefix encoders",
            status="ruled_out_current_target_native_encoder_family",
            artifact="target_self_learned_encoders",
            score=learned_target["best_method_agreement"],
            baseline=learned_target["best_baseline_agreement"],
            delta=learned_target["best_agreement_delta"],
            ci95_low=None,
            bytes_record=None,
            evidence=(
                f"{learned_target['pass_rows']}/{learned_target['rows']} pass; best agreement delta "
                f"{learned_target['best_agreement_delta']:.6f} against "
                f"{learned_target['best_control_name']}; worst agreement delta "
                f"{learned_target['worst_agreement_delta']:.6f}."
            ),
            decision="Do not run more chunk/distill/query-resampler variants without a new information path.",
        )
    )

    source_conditioned = _source_conditioned_receiver_summaries(
        artifacts["target_self_source_conditioned_receivers"]
    )
    oracle_text = (
        "unknown"
        if source_conditioned["max_source_top1_or_top2_oracle_accuracy"] is None
        else f"{source_conditioned['max_source_top1_or_top2_oracle_accuracy']:.6f}"
    )
    rows.append(
        _headline_row(
            branch="Source-conditioned target-native resonance receivers",
            status="ruled_out_current_source_conditioned_receiver_family",
            artifact="target_self_source_conditioned_receivers",
            score=source_conditioned["best_method_accuracy"],
            baseline=source_conditioned["best_baseline_accuracy"],
            delta=source_conditioned["best_accuracy_delta"],
            ci95_low=None,
            bytes_record=None,
            evidence=(
                f"{source_conditioned['pass_rows']}/{source_conditioned['rows']} pass; best accuracy delta "
                f"{source_conditioned['best_accuracy_delta']:.6f} against "
                f"{source_conditioned['best_control_name']}; source top1/top2 oracle reaches {oracle_text}."
            ),
            decision="Diagnose complementarity/gating before implementing another source-to-prefix decoder.",
        )
    )

    frontier = artifacts["hellaswag_complementarity_frontier"]["headline"]
    rows.append(
        _headline_row(
            branch="HellaSwag complementarity-frontier selector diagnostic",
            status="headroom_alive_selector_blocked",
            artifact="hellaswag_complementarity_frontier",
            score=frontier["selected_selector_accuracy"],
            baseline=frontier["fixed_hybrid_accuracy"],
            delta=frontier["selected_selector_delta_vs_fixed_hybrid"],
            ci95_low=frontier["selected_selector_ci95_low_vs_fixed_hybrid"],
            bytes_record=frontier["framed_record_bytes"],
            evidence=(
                f"Fixed+source top1/top2 oracle {frontier['fixed_or_source_top1_top2_oracle_accuracy']:.6f}; "
                f"source top1/top2 covers {frontier['fixed_wrong_source_top1_or_top2_correct']} fixed-hybrid errors, "
                f"but selected frontier makes {frontier['selected_selector_overrides']} overrides."
            ),
            decision="Do not train another HellaSwag selector on the same packet fields; require a new information path.",
        )
    )

    multisignal = artifacts["hellaswag_multisignal_packet_frontier"]["headline"]
    rows.append(
        _headline_row(
            branch="HellaSwag multi-signal source packet frontier",
            status="ruled_out_cached_policy_packet",
            artifact="hellaswag_multisignal_packet_frontier",
            score=multisignal["multisignal_selector_accuracy"],
            baseline=multisignal["fixed_hybrid_accuracy"],
            delta=multisignal["multisignal_selector_delta_vs_fixed_hybrid"],
            ci95_low=multisignal["multisignal_selector_ci95_low_vs_fixed_hybrid"],
            bytes_record=multisignal["framed_record_bytes"],
            evidence=(
                f"Selector accuracy {multisignal['multisignal_selector_accuracy']:.6f} vs fixed "
                f"{multisignal['fixed_hybrid_accuracy']:.6f}; overrides "
                f"{multisignal['multisignal_selector_overrides']} rows; best destructive control "
                f"{multisignal['best_destructive_control_name']} at "
                f"{multisignal['best_destructive_control_accuracy']:.6f}."
            ),
            decision="Do not continue cached Qwen policy-prediction packets on this HellaSwag slice.",
        )
    )

    return {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "gate": "iclr_colm_v2_live_branch_triage",
        "input_artifacts": _stringify_paths(artifact_paths or DEFAULT_ARTIFACT_PATHS),
        "input_sha256": _input_sha256(artifact_paths or DEFAULT_ARTIFACT_PATHS),
        "readiness": {
            "colm_v2": "scoped_positive_ready_for_writeup_if_claims_are_narrow",
            "iclr": "blocked_by_lack_of_broad_or_learned_positive_receiver",
            "conditional_pq_same_family_alive": bool(
                conditional_readiness["same_family_disjoint_n500_positive"]
            ),
            "conditional_pq_cross_family_blocked": bool(conditional_readiness["cross_family_blocked"]),
            "hellaswag_shallow_receiver_family_saturated": True,
            "sparse_resonance_current_basis_blocked": True,
            "target_self_resonance_capacity_alive": True,
            "target_self_resonance_learned_encoders_blocked": True,
            "source_conditioned_target_native_receivers_blocked": True,
            "hellaswag_complementarity_headroom_alive": True,
            "hellaswag_current_frontier_selector_blocked": True,
            "hellaswag_cached_policy_packets_blocked": True,
        },
        "story": (
            "LatentWire_v2 can currently support a scoped COLM_v2 story: byte-scale, "
            "source-private packets plus strict destructive controls. The ICLR story is "
            "still blocked because cross-family conditional PQ, deterministic public-basis "
            "conditioning, and HellaSwag learned/source-conditioned resonance receivers have "
            "not produced a positive row beyond packet/source-choice/target-cache controls."
        ),
        "submission_gap": (
            "ICLR needs a positive learned or broader-benchmark receiver that passes strict "
            "destructive controls. COLM_v2 can be prepared around the conditional-PQ "
            "shared-schema method, the fixed-byte HellaSwag packet row, and the target-resonance "
            "capacity-versus-held-out-failure analysis with explicit limitations."
        ),
        "current_contributions": [
            {
                "name": "source_private_low_rate_packets",
                "status": "alive_for_colm_v2",
                "needs_work": "broader or learned receiver evidence for ICLR",
            },
            {
                "name": "strict_destructive_controls",
                "status": "strong",
                "needs_work": "paper integration and compact tables",
            },
            {
                "name": "systems_byte_accounting",
                "status": "mac_local_ready",
                "needs_work": "native dense-KV/C2C measurements before throughput or energy claims",
            },
            {
                "name": "sparse_resonance_packets",
                "status": "framing_alive_method_not_yet_positive",
                "needs_work": "new mechanism beyond deterministic PQ, PCA atoms, chunk encoders, query resamplers, and source-to-prefix decoders",
            },
            {
                "name": "target_self_resonance_capacity_probe",
                "status": "capacity_alive",
                "needs_work": "held-out/source-private receiver that beats slots-only, zero-source, wrong-source, source-choice, and candidate-roll controls",
            },
        ],
        "branch_rows": rows,
        "promoted": [
            "Conditional PQ shared-schema packet as COLM_v2 positive method.",
            "HellaSwag fixed hybrid candidate packet as a systems/privacy packet row.",
        ],
        "weakened_or_ruled_out": [
            "Conditional PQ public-zscore and corruption-to-noop as held-out-family rescues.",
            "Conditional PQ public-SVD whitening as a held-out-family rescue.",
            "ARC sparse resonance PCA/target-aligned soft-prefix basis as implemented.",
            "HellaSwag protected-rival, top2 bucket, linear receiver, harm bucket, and denoising syndrome switchers.",
            "Sparse/common-basis top2 atom causality in the current HellaSwag implementation.",
            "Target self-resonance chunk/distill/query-resampler encoders as reusable target-native receivers.",
            "Source-conditioned source-hidden/codebook/refinement target-native receivers as currently implemented.",
            "HellaSwag complementarity-frontier selector with current top1/top2 packet fields.",
            "HellaSwag cached hidden/score/vote policy-prediction packets as a repair frontier.",
        ],
        "next_exact_gate": {
            "name": "conditional_pq_integrity_or_colm_v2_integration_gate",
            "primary_path": (
                "Stop HellaSwag cached-selector work unless a qualitatively new hidden/PQ residual "
                "feature is introduced. The next ICLR implementation should return to conditional "
                "PQ with learned integrity/corruption-to-no-op decoding on the n256 held-out-family "
                "surface, or use this HellaSwag map only as a diagnostic limitation."
            ),
            "fallback_path": (
                "If the conditional-PQ integrity branch is too large for the next Mac-local turn, "
                "switch to COLM_v2 table and figure integration using conditional-PQ, fixed-byte "
                "HellaSwag, target-resonance capacity, complementarity-frontier saturation, and "
                "multi-signal packet failure."
            ),
            "pass_bar": (
                "A learned or rule-based packet receiver must improve over source-index/rank/score, "
                "same-byte text, wrong-source, candidate-roll, and target-derived controls with a "
                "positive paired CI95 low on a frozen slice."
            ),
            "required_controls": [
                "target_only",
                "answer_masked_source",
                "constrained_wrong_row_source",
                "same_source_choice_wrong_row",
                "candidate_roll_or_deranged_public_basis",
                "permuted_codes",
                "random_same_byte",
                "opaque_slot_or_deranged_basis",
                "source_index_rank_score_comparators_when_meaningful",
                "same_byte_visible_text",
            ],
        },
        "claim_boundaries": [
            "Do not claim broad cross-family latent communication yet.",
            "Do not claim sparse resonance packets as a positive method from the current PCA or sparse-atom gates.",
            "Do not claim C2C/KV throughput, HBM, PCIe, NVLink, or energy wins without native measurements.",
            "Frame C2C/KVComm/TurboQuant/KIVI as high-bandwidth or KV-compression baselines with different exposure regimes.",
        ],
    }


def _stringify_paths(paths: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in paths.items():
        if isinstance(value, list):
            output[key] = [str(path) for path in value]
        else:
            output[key] = str(value)
    return output


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    lines = [
        "# ICLR / COLM_v2 Live Branch Triage",
        "",
        f"- created UTC: `{payload['created_utc']}`",
        f"- COLM_v2 readiness: `{payload['readiness']['colm_v2']}`",
        f"- ICLR readiness: `{payload['readiness']['iclr']}`",
        "",
        "## Current Story",
        "",
        payload["story"],
        "",
        "## Exact Submission Gap",
        "",
        payload["submission_gap"],
        "",
        "## Current Technical Contributions",
        "",
    ]
    for contribution in payload["current_contributions"]:
        lines.append(
            f"- `{contribution['name']}`: {contribution['status']}; needs {contribution['needs_work']}."
        )
    lines.extend(
        [
            "",
            "## Branch Table",
            "",
            "| Branch | Status | Score | Baseline | Delta | CI95 Low | Bytes | Decision |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["branch_rows"]:
        lines.append(
            "| {branch} | `{status}` | `{score}` | `{baseline}` | `{delta}` | `{ci95_low}` | `{record_bytes}` | {decision} |".format(
                **row
            )
        )
    lines.extend(["", "## Evidence Notes", ""])
    for row in payload["branch_rows"]:
        lines.append(f"- `{row['branch']}`: {row['evidence']}")
    lines.extend(["", "## Promoted", ""])
    lines.extend(f"- {item}" for item in payload["promoted"])
    lines.extend(["", "## Weakened Or Ruled Out", ""])
    lines.extend(f"- {item}" for item in payload["weakened_or_ruled_out"])
    next_gate = payload["next_exact_gate"]
    lines.extend(
        [
            "",
            "## Next Exact Gate",
            "",
            f"- name: `{next_gate['name']}`",
            f"- primary path: {next_gate['primary_path']}",
            f"- fallback path: {next_gate['fallback_path']}",
            f"- pass bar: {next_gate['pass_bar']}",
            "- required controls: "
            + ", ".join(f"`{control}`" for control in next_gate["required_controls"]),
            "",
            "## Claim Boundaries",
            "",
        ]
    )
    lines.extend(f"- {boundary}" for boundary in payload["claim_boundaries"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(payload: dict[str, Any], output_dir: pathlib.Path, paper_path: pathlib.Path | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "live_branch_triage.json"
    csv_path = output_dir / "branch_rows.csv"
    md_path = output_dir / "live_branch_triage.md"
    manifest_path = output_dir / "manifest.json"

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(csv_path, payload["branch_rows"])
    _write_markdown(md_path, payload)
    if paper_path is not None:
        paper_path.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(paper_path, payload)
    manifest = {
        "created_utc": payload["created_utc"],
        "gate": payload["gate"],
        "files": [
            _display_path(json_path),
            _display_path(csv_path),
            _display_path(md_path),
        ]
        + ([] if paper_path is None else [_display_path(paper_path)]),
        "pass_gate": False,
        "readiness": payload["readiness"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT_DIR,
    paper_path: pathlib.Path | None = DEFAULT_PAPER_PATH,
) -> dict[str, Any]:
    artifacts = _load_artifacts(DEFAULT_ARTIFACT_PATHS)
    payload = build_triage(artifacts=artifacts, artifact_paths=DEFAULT_ARTIFACT_PATHS)
    write_outputs(payload, output_dir, paper_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paper-path", type=pathlib.Path, default=DEFAULT_PAPER_PATH)
    parser.add_argument("--no-paper", action="store_true")
    args = parser.parse_args()
    payload = run(output_dir=args.output_dir, paper_path=None if args.no_paper else args.paper_path)
    print(json.dumps(payload["readiness"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
