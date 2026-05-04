from __future__ import annotations

import json

from scripts.build_iclr_colm_v2_live_branch_triage import build_triage, write_outputs


def _summary(
    *,
    source_accuracy: float,
    best_control_accuracy: float,
    source_minus_best_control: float,
    pass_gate: bool,
) -> dict:
    return {
        "summary": {
            "best_control_accuracy": best_control_accuracy,
            "best_control_condition": "random_same_byte",
            "n": 16,
            "pass_gate": pass_gate,
            "source_accuracy": source_accuracy,
            "source_minus_best_control": source_minus_best_control,
            "target_only_accuracy": 0.25,
        }
    }


def _hellaswag_headline(**overrides: float | int | str | bool) -> dict:
    headline = {
        "candidate_only_accuracy": 0.45,
        "eval_rows": 16,
        "fixed_hybrid_accuracy": 0.47,
        "framed_record_bytes": 5,
        "native_systems_claim_allowed": False,
        "phi_target_accuracy": 0.25,
        "raw_payload_bytes": 2,
    }
    headline.update(overrides)
    return {"headline": headline}


def _target_oracle_headline(**overrides: float | int | str | bool) -> dict:
    headline = {
        "agreement_gap_vs_best_destructive": 0.25,
        "best_destructive_agreement": 0.75,
        "best_destructive_by_agreement": "zero_prefix",
        "optimized_accuracy": 0.31,
        "optimized_agreement": 1.0,
    }
    headline.update(overrides)
    return {"headline": headline, "pass_gate": True}


def _chunk_encoder_headline(**overrides: float | int | str | bool) -> dict:
    headline = {
        "best_destructive_agreement": 0.75,
        "best_destructive_by_agreement": "zero_prefix",
        "learned_accuracy": 0.25,
        "learned_agreement": 0.75,
    }
    headline.update(overrides)
    return {"headline": headline, "pass_gate": False}


def _oracle_distill_headline(**overrides: float | int | str | bool) -> dict:
    headline = {
        "best_target_only_agreement": 0.5,
        "best_target_only_by_agreement": "slots_only_oracle_distill",
        "distill_accuracy": 0.25,
        "distill_agreement": 0.375,
    }
    headline.update(overrides)
    return {"headline": headline, "pass_gate": False}


def _query_resampler_headline(**overrides: float | int | str | bool) -> dict:
    headline = {
        "best_control_agreement": 0.625,
        "best_control_by_agreement": "chunk_mean_prefix",
        "query_accuracy": 0.375,
        "query_agreement": 0.5,
    }
    headline.update(overrides)
    return {"headline": headline, "pass_gate": False}


def _source_conditioned_headline(method_field: str, **overrides: float | int | str | bool) -> dict:
    headline = {
        "best_destructive_accuracy": 0.375,
        "best_destructive_by_accuracy": "zero_source",
        method_field: 0.375,
        "source_top1_or_top2_oracle_accuracy": 0.625,
    }
    headline.update(overrides)
    return {"headline": headline, "pass_gate": False}


def test_live_branch_triage_summarizes_current_decision(tmp_path) -> None:
    synthetic_input = tmp_path / "synthetic.json"
    synthetic_input.write_text("{}", encoding="utf-8")
    artifacts = {
        "conditional_pq_status": {
            "readiness": {
                "same_family_disjoint_n500_positive": True,
                "cross_family_blocked": True,
            },
            "evidence": {
                "budget2_pass_rows": 4,
                "budget2_rows": 4,
                "cross_family_schema_grid_pass_rows": 0,
                "cross_family_schema_grid_rows": 28,
                "decisive_disjoint_n500_pass_rows": 16,
                "decisive_disjoint_n500_rows": 16,
                "min_decisive_ci95_low_vs_best_control": 0.5,
                "systems": {"method_record_bytes_range": [5, 7]},
            },
        },
        "conditional_public_zscore": [
            _summary(source_accuracy=0.3, best_control_accuracy=0.35, source_minus_best_control=-0.05, pass_gate=False),
            _summary(source_accuracy=0.46, best_control_accuracy=0.44, source_minus_best_control=0.02, pass_gate=False),
        ],
        "conditional_public_svd": [
            _summary(source_accuracy=0.375, best_control_accuracy=0.613, source_minus_best_control=-0.238, pass_gate=False),
            _summary(source_accuracy=0.25, best_control_accuracy=0.75, source_minus_best_control=-0.5, pass_gate=False),
        ],
        "conditional_corruption_noop": [
            _summary(source_accuracy=0.25, best_control_accuracy=0.25, source_minus_best_control=0.0, pass_gate=False)
        ],
        "conditional_pq_integrity_threshold": {
            "headline": {
                "best_control_accuracy": 0.457,
                "best_control_condition": "label_shuffled_encoder",
                "ci95_low_vs_best_control": -0.098,
                "framed_record_bytes": 7,
                "max_corrupt_accept_rate": 1.0,
                "selected_score_name": "negative_min_l2",
                "source_accept_rate": 0.773,
                "source_accuracy": 0.426,
                "source_minus_best_control": -0.031,
            }
        },
        "arc_sparse_resonance": [
            {
                "headline": {
                    "best_control_accuracy": 0.625,
                    "best_control_by_accuracy": "target_derived_prefix",
                    "matched_accuracy": 0.25,
                    "matched_minus_best_control_accuracy": -0.375,
                },
                "pass_gate": False,
            }
        ],
        "arc_behavior_atom_decoder": [
            {
                "strict_headline": {
                    "best_required_control": "top_atom_knockout",
                    "best_required_control_accuracy": 0.4375,
                    "matched_accuracy": 0.375,
                    "matched_packet_fired": 8,
                    "matched_packet_harmed": 0,
                    "matched_packet_helped": 2,
                    "target_only_accuracy": 0.25,
                    "worst_required_ci95_low": -0.375,
                },
                "systems_packet_sideband": {"framed_packet_bytes_per_row": 7},
                "pass_gate": False,
            }
        ],
        "openbookqa_receiver_headroom": {
            "budget_bytes": 3,
            "headline": {
                "aggregate_seed_row_ci_vs_packet": {"ci95_low": 0.0004},
                "all_seed_deltas_positive": True,
                "default_best_receiver_control": "same_byte_structured_text",
                "default_best_receiver_control_accuracy": 0.378,
                "default_seed_matched": {
                    "base_accuracy": 0.378,
                    "harm_count": 35,
                    "help_count": 58,
                    "paired_ci95_vs_base": {"ci95_low": 0.008},
                    "receiver_accuracy": 0.424,
                    "receiver_minus_base": 0.046,
                    "receiver_minus_target_public": 0.052,
                    "target_public_accuracy": 0.372,
                },
                "seed_count": 5,
                "strict_per_seed_ci_pass_count": 2,
            },
            "pass_gate": True,
            "receiver_candidate_pass": True,
            "test_rows": 500,
        },
        "hellaswag_fixed_hybrid": _hellaswag_headline(
            best_control_accuracy=0.48,
            candidate_only_accuracy=0.52,
            eval_rows=100,
            fixed_hybrid_accuracy=0.53,
            hybrid_ci95_low_vs_candidate_only=0.002,
            hybrid_delta_vs_candidate_only=0.01,
            positive_slice_count=10,
            slice_count=10,
        ),
        "hellaswag_protected_rival": _hellaswag_headline(
            fixed_hybrid_accuracy=0.47,
            framed_record_bytes=5,
            hybrid_rival_oracle_accuracy=0.68,
            protected_pair_decoder_accuracy=0.46,
            protected_pair_decoder_ci95_low_vs_fixed_hybrid=-0.01,
            protected_pair_decoder_delta_vs_fixed_hybrid=-0.01,
            protected_pair_decoder_harms=6,
            protected_pair_decoder_helps=2,
        ),
        "hellaswag_official_receiver": _hellaswag_headline(
            fixed_hybrid_accuracy=0.47,
            framed_record_bytes=5,
            hybrid_rival_phi_oracle_accuracy=0.77,
            receiver_calibrated_accuracy=0.466,
            receiver_calibrated_ci95_low_vs_fixed_hybrid=-0.008,
            receiver_calibrated_delta_vs_fixed_hybrid=-0.004,
        ),
        "hellaswag_harm_bucket": _hellaswag_headline(
            fixed_hybrid_accuracy=0.47,
            framed_record_bytes=6,
            harm_controlled_accuracy=0.47,
            harm_controlled_ci95_low_vs_fixed_hybrid=0.0,
            harm_controlled_delta_vs_fixed_hybrid=0.0,
            harm_controlled_override_count=0,
            selected_eligible_bucket_count=0,
            selected_scheme="no_op",
        ),
        "hellaswag_top2_bucket": _hellaswag_headline(
            fixed_hybrid_accuracy=0.47,
            source_top1_or_top2_oracle_accuracy=0.67,
            top2_ambiguity_bucket_accuracy=0.47,
            top2_ambiguity_bucket_ci95_low_vs_fixed_hybrid=0.0,
            top2_ambiguity_bucket_delta_vs_fixed_hybrid=0.0,
            top2_ambiguity_bucket_override_count=0,
        ),
        "hellaswag_denoising_syndrome": _hellaswag_headline(
            fixed_hybrid_accuracy=0.47,
            framed_record_bytes=4,
            target_or_hybrid_oracle_accuracy=0.60,
            denoising_syndrome_accuracy=0.46,
            denoising_ci95_low_vs_fixed_hybrid=-0.01,
            denoising_delta_vs_fixed_hybrid=-0.01,
            denoising_harms_vs_fixed_hybrid=5,
            denoising_helps_vs_fixed_hybrid=2,
        ),
        "hellaswag_sparse_common_basis": _hellaswag_headline(
            ambiguity_best_destructive_control_accuracy=0.50,
            ambiguity_best_destructive_control_name="target_derived_source_pair",
            ambiguity_code_accuracy=0.50,
            ambiguity_code_ci95_low_vs_packet_only=-0.003,
            ambiguity_code_delta_vs_packet_only=-0.001,
            ambiguity_code_harm_count=1,
            ambiguity_code_help_count=0,
            framed_record_bytes=4,
            packet_only_accuracy=0.502,
        ),
        "target_self_oracle_soft_prefix": [
            _target_oracle_headline(),
            _target_oracle_headline(agreement_gap_vs_best_destructive=0.125, optimized_agreement=0.875),
        ],
        "target_self_learned_encoders": [
            _chunk_encoder_headline(),
            _oracle_distill_headline(),
            _query_resampler_headline(),
        ],
        "target_self_source_conditioned_receivers": [
            _source_conditioned_headline("source_oracle_accuracy", source_oracle_accuracy=0.25),
            _source_conditioned_headline("source_hidden_residual_accuracy"),
            _source_conditioned_headline(
                "source_codebook_accuracy",
                best_destructive_accuracy=0.75,
                best_destructive_by_accuracy="source_top1_label_control",
                source_codebook_accuracy=0.5,
                source_top1_or_top2_oracle_accuracy=1.0,
            ),
            _source_conditioned_headline("consistency_refined_accuracy"),
        ],
        "hellaswag_complementarity_frontier": {
            "headline": {
                "fixed_hybrid_accuracy": 0.47,
                "fixed_or_source_top1_top2_oracle_accuracy": 0.69,
                "fixed_wrong_source_top1_or_top2_correct": 17,
                "framed_record_bytes": 4,
                "selected_selector_accuracy": 0.47,
                "selected_selector_ci95_low_vs_fixed_hybrid": 0.0,
                "selected_selector_delta_vs_fixed_hybrid": 0.0,
                "selected_selector_overrides": 0,
            }
        },
        "hellaswag_multisignal_packet_frontier": {
            "headline": {
                "best_destructive_control_accuracy": 0.431,
                "best_destructive_control_name": "field_shuffle_multisignal_control",
                "fixed_hybrid_accuracy": 0.47,
                "framed_record_bytes": 5,
                "multisignal_selector_accuracy": 0.456,
                "multisignal_selector_ci95_low_vs_fixed_hybrid": -0.024,
                "multisignal_selector_delta_vs_fixed_hybrid": -0.014,
                "multisignal_selector_overrides": 30,
            }
        },
    }

    payload = build_triage(artifacts=artifacts, artifact_paths={"synthetic": synthetic_input})
    assert payload["readiness"]["colm_v2"] == "scoped_positive_ready_for_writeup_if_claims_are_narrow"
    assert payload["readiness"]["iclr"] == "blocked_by_lack_of_broad_or_learned_positive_receiver"
    assert payload["branch_rows"][0]["status"] == "promote_for_colm_v2_only"
    assert any(row["status"] == "ruled_out_shallow_pair_decoder" for row in payload["branch_rows"])
    assert any(
        row["status"] == "capacity_alive_not_source_private_method" for row in payload["branch_rows"]
    )
    assert any(
        row["status"] == "ruled_out_current_source_conditioned_receiver_family"
        for row in payload["branch_rows"]
    )
    assert any(
        row["status"] == "headroom_alive_selector_blocked"
        for row in payload["branch_rows"]
    )
    assert payload["readiness"]["hellaswag_complementarity_headroom_alive"] is True
    assert payload["readiness"]["hellaswag_current_frontier_selector_blocked"] is True
    assert payload["readiness"]["hellaswag_cached_policy_packets_blocked"] is True
    assert payload["readiness"]["conditional_pq_simple_integrity_threshold_blocked"] is True
    assert payload["readiness"]["arc_behavior_atom_basis_blocked"] is True
    assert payload["readiness"]["openbookqa_receiver_second_benchmark_alive"] is True
    assert any(
        row["status"] == "ruled_out_cached_policy_packet"
        for row in payload["branch_rows"]
    )
    assert any(
        row["status"] == "ruled_out_simple_integrity_threshold"
        for row in payload["branch_rows"]
    )
    assert any(
        row["status"] == "promote_for_colm_v2_second_benchmark_caveated"
        for row in payload["branch_rows"]
    )
    assert any(
        row["status"] == "ruled_out_current_behavior_atom_basis"
        for row in payload["branch_rows"]
    )
    assert payload["next_exact_gate"]["name"] == "arc_n32_tokenwise_source_evidence_preflight"

    out_dir = tmp_path / "out"
    paper_path = tmp_path / "paper.md"
    write_outputs(payload, out_dir, paper_path)
    assert json.loads((out_dir / "live_branch_triage.json").read_text())["gate"] == "iclr_colm_v2_live_branch_triage"
    assert (out_dir / "branch_rows.csv").exists()
    assert "ICLR / COLM_v2 Live Branch Triage" in paper_path.read_text()
