from __future__ import annotations

import json

from scripts.build_source_private_candidate_local_common_basis_falsification import (
    build_common_basis_falsification,
)


def _metric(accuracy: float) -> dict[str, float | int]:
    return {
        "accuracy": accuracy,
        "correct": int(accuracy * 16),
        "mean_payload_bytes": 8.0,
        "mean_payload_tokens": 4.0,
        "n": 16,
        "p50_latency_ms": 0.1,
        "p95_latency_ms": 0.2,
        "strict_accuracy": accuracy,
    }


def _write_run(
    root,
    *,
    name: str,
    decoder_score_mode: str,
    matched: float,
    best_control: float,
    best_control_name: str,
    controls_ok: bool,
    pass_gate: bool,
) -> None:
    run_dir = root / name
    direction_dir = run_dir / "core_to_holdout"
    direction_dir.mkdir(parents=True)
    metrics = {
        "target_only": _metric(0.25),
        "learned_synonym_dictionary_packet": _metric(matched),
        "zero_source": _metric(0.25),
        "shuffled_source": _metric(0.25),
        "atom_id_derangement": _metric(best_control if best_control_name == "atom_id_derangement" else 0.25),
        "private_random_source_atoms": _metric(0.25),
        "permuted_teacher_receiver": _metric(
            best_control if best_control_name == "permuted_teacher_receiver" else 0.25
        ),
        "random_same_byte": _metric(0.25),
        "answer_only_text": _metric(0.25),
        "structured_text_matched": _metric(0.25),
        "top_atom_knockout": _metric(0.25),
        "private_random_knockout": _metric(0.25),
        "oracle_learned_candidate_atoms": _metric(0.875),
    }
    summary = {
        "direction": "core_to_holdout",
        "surface_overlap_audit": {
            "calibration_eval_exact_id_overlap_count": 0,
            "exact_transformed_eval_surface_overlap_count": 0,
        },
        "budget_summaries": [
            {
                "budget_bytes": 8,
                "n": 16,
                "target_accuracy": 0.25,
                "learned_synonym_dictionary_accuracy": matched,
                "best_control_accuracy": best_control,
                "best_control_name": best_control_name,
                "learned_minus_target": matched - 0.25,
                "learned_minus_best_control": matched - best_control,
                "oracle_learned_candidate_atoms_accuracy": 0.875,
                "paired_bootstrap_vs_target": {"ci95_low": 0.2, "ci95_high": 0.4, "mean": matched - 0.25},
                "controls_ok": controls_ok,
                "pass_gate": pass_gate,
                "metrics": metrics,
            }
        ],
    }
    (direction_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    gate = {
        "decoder_score_mode": decoder_score_mode,
        "rows": [
            {
                "direction": "core_to_holdout",
                "budget_bytes": 8,
                "pass_gate": pass_gate,
            }
        ],
    }
    (run_dir / "learned_synonym_dictionary_packet_gate.json").write_text(json.dumps(gate), encoding="utf-8")


def test_common_basis_falsification_marks_global_dot_control_leak(tmp_path) -> None:
    _write_run(
        tmp_path,
        name="live_seed47",
        decoder_score_mode="candidate_local_residual_norm",
        matched=0.625,
        best_control=0.25,
        best_control_name="zero_source",
        controls_ok=True,
        pass_gate=True,
    )
    _write_run(
        tmp_path,
        name="global_seed47",
        decoder_score_mode="global_dot",
        matched=0.875,
        best_control=0.625,
        best_control_name="permuted_teacher_receiver",
        controls_ok=False,
        pass_gate=False,
    )
    _write_run(
        tmp_path,
        name="relative_seed47",
        decoder_score_mode="relative_anchor_dot",
        matched=0.75,
        best_control=0.375,
        best_control_name="permuted_teacher_receiver",
        controls_ok=False,
        pass_gate=False,
    )
    _write_run(
        tmp_path,
        name="relative_innovation_seed47",
        decoder_score_mode="relative_anchor_innovation_residual_norm",
        matched=0.75,
        best_control=0.375,
        best_control_name="permuted_teacher_receiver",
        controls_ok=False,
        pass_gate=False,
    )
    _write_run(
        tmp_path,
        name="relative_rank_innovation_seed47",
        decoder_score_mode="relative_anchor_rank_innovation_residual_norm",
        matched=0.625,
        best_control=0.375,
        best_control_name="permuted_teacher_receiver",
        controls_ok=False,
        pass_gate=False,
    )
    _write_run(
        tmp_path,
        name="procrustes_seed47",
        decoder_score_mode="procrustes_dot",
        matched=0.875,
        best_control=0.875,
        best_control_name="permuted_teacher_receiver",
        controls_ok=False,
        pass_gate=False,
    )
    _write_run(
        tmp_path,
        name="ridge_cca_seed47",
        decoder_score_mode="ridge_cca_dot",
        matched=0.75,
        best_control=0.5,
        best_control_name="permuted_teacher_receiver",
        controls_ok=False,
        pass_gate=False,
    )
    _write_run(
        tmp_path,
        name="ridge_cca_stack_seed47",
        decoder_score_mode="ridge_cca_residual_norm",
        matched=0.625,
        best_control=0.375,
        best_control_name="permuted_teacher_receiver",
        controls_ok=False,
        pass_gate=False,
    )
    _write_run(
        tmp_path,
        name="lstirp_seed47",
        decoder_score_mode="inverse_relative_dot",
        matched=0.75,
        best_control=0.5,
        best_control_name="permuted_teacher_receiver",
        controls_ok=False,
        pass_gate=False,
    )
    _write_run(
        tmp_path,
        name="lstirp_stack_seed47",
        decoder_score_mode="inverse_relative_residual_norm",
        matched=0.625,
        best_control=0.375,
        best_control_name="permuted_teacher_receiver",
        controls_ok=False,
        pass_gate=False,
    )
    _write_run(
        tmp_path,
        name="sinkhorn_seed47",
        decoder_score_mode="sinkhorn_ot_dot",
        matched=0.75,
        best_control=0.5,
        best_control_name="permuted_teacher_receiver",
        controls_ok=False,
        pass_gate=False,
    )
    _write_run(
        tmp_path,
        name="sinkhorn_stack_seed47",
        decoder_score_mode="sinkhorn_ot_residual_norm",
        matched=0.625,
        best_control=0.375,
        best_control_name="permuted_teacher_receiver",
        controls_ok=False,
        pass_gate=False,
    )
    _write_run(
        tmp_path,
        name="gw_seed47",
        decoder_score_mode="gromov_wasserstein_dot",
        matched=0.75,
        best_control=0.5,
        best_control_name="permuted_teacher_receiver",
        controls_ok=False,
        pass_gate=False,
    )
    _write_run(
        tmp_path,
        name="gw_stack_seed47",
        decoder_score_mode="gromov_wasserstein_residual_norm",
        matched=0.625,
        best_control=0.375,
        best_control_name="permuted_teacher_receiver",
        controls_ok=False,
        pass_gate=False,
    )

    payload = build_common_basis_falsification(
        output_dir=tmp_path / "out",
        live_run_dirs=[tmp_path / "live_seed47"],
        global_dot_run_dirs=[tmp_path / "global_seed47"],
        procrustes_run_dirs=[tmp_path / "procrustes_seed47"],
        ridge_cca_run_dirs=[tmp_path / "ridge_cca_seed47"],
        ridge_cca_stack_run_dirs=[tmp_path / "ridge_cca_stack_seed47"],
        lstirp_run_dirs=[tmp_path / "lstirp_seed47"],
        lstirp_stack_run_dirs=[tmp_path / "lstirp_stack_seed47"],
        sinkhorn_ot_run_dirs=[tmp_path / "sinkhorn_seed47"],
        sinkhorn_ot_stack_run_dirs=[tmp_path / "sinkhorn_stack_seed47"],
        gw_run_dirs=[tmp_path / "gw_seed47"],
        gw_stack_run_dirs=[tmp_path / "gw_stack_seed47"],
        relative_anchor_run_dirs=[tmp_path / "relative_seed47"],
        relative_anchor_stack_run_dirs=[],
        diagnostic_run_dirs=[],
        relative_anchor_innovation_run_dirs=[tmp_path / "relative_innovation_seed47"],
        relative_anchor_rank_innovation_run_dirs=[tmp_path / "relative_rank_innovation_seed47"],
    )

    assert payload["headline"]["pass_gate"] is True
    assert payload["headline"]["live_pass_rows"] == 1
    assert payload["headline"]["global_common_basis_pass_rows"] == 0
    assert payload["headline"]["global_common_basis_control_leak_rows"] == 1
    assert payload["headline"]["procrustes_common_basis_pass_rows"] == 0
    assert payload["headline"]["procrustes_common_basis_control_leak_rows"] == 1
    assert payload["headline"]["ridge_cca_common_basis_pass_rows"] == 0
    assert payload["headline"]["ridge_cca_common_basis_control_leak_rows"] == 1
    assert payload["headline"]["ridge_cca_stack_pass_rows"] == 0
    assert payload["headline"]["lstirp_pass_rows"] == 0
    assert payload["headline"]["lstirp_control_leak_rows"] == 1
    assert payload["headline"]["lstirp_stack_pass_rows"] == 0
    assert payload["headline"]["sinkhorn_ot_pass_rows"] == 0
    assert payload["headline"]["sinkhorn_ot_control_leak_rows"] == 1
    assert payload["headline"]["sinkhorn_ot_stack_pass_rows"] == 0
    assert payload["headline"]["gw_pass_rows"] == 0
    assert payload["headline"]["gw_control_leak_rows"] == 1
    assert payload["headline"]["gw_stack_pass_rows"] == 0
    assert payload["headline"]["relative_anchor_pass_rows"] == 0
    assert payload["headline"]["relative_anchor_control_leak_rows"] == 1
    assert payload["headline"]["relative_anchor_innovation_pass_rows"] == 0
    assert payload["headline"]["relative_anchor_innovation_control_leak_rows"] == 1
    assert payload["headline"]["relative_anchor_rank_innovation_pass_rows"] == 0
    assert payload["headline"]["relative_anchor_rank_innovation_control_leak_rows"] == 1
    comparison = payload["comparisons"][0]
    assert comparison["baseline_invalidated_by_controls"] is True
    assert comparison["baseline_matched_accuracy"] > comparison["live_matched_accuracy"]
    assert (tmp_path / "out" / "candidate_local_common_basis_falsification.json").exists()
    assert (tmp_path / "out" / "manifest.json").exists()
