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
        "conditional_corruption_noop": [
            _summary(source_accuracy=0.25, best_control_accuracy=0.25, source_minus_best_control=0.0, pass_gate=False)
        ],
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
    }

    payload = build_triage(artifacts=artifacts, artifact_paths={"synthetic": synthetic_input})
    assert payload["readiness"]["colm_v2"] == "scoped_positive_ready_for_writeup_if_claims_are_narrow"
    assert payload["readiness"]["iclr"] == "blocked_by_lack_of_broad_or_learned_positive_receiver"
    assert payload["branch_rows"][0]["status"] == "promote_for_colm_v2_only"
    assert any(row["status"] == "ruled_out_shallow_pair_decoder" for row in payload["branch_rows"])
    assert payload["next_exact_gate"]["name"] == "source_private_conditional_codebook_or_target_resonance_gate"

    out_dir = tmp_path / "out"
    paper_path = tmp_path / "paper.md"
    write_outputs(payload, out_dir, paper_path)
    assert json.loads((out_dir / "live_branch_triage.json").read_text())["gate"] == "iclr_colm_v2_live_branch_triage"
    assert (out_dir / "branch_rows.csv").exists()
    assert "ICLR / COLM_v2 Live Branch Triage" in paper_path.read_text()
