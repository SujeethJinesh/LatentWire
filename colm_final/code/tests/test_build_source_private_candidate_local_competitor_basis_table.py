from __future__ import annotations

import json

from scripts.build_source_private_candidate_local_competitor_basis_table import build_competitor_basis_table


def _common_row(row_group: str, *, pass_gate: bool, matched: float, best_control: float) -> dict[str, object]:
    return {
        "row_group": row_group,
        "method": row_group,
        "seed": 47,
        "direction": "core_to_holdout",
        "budget_bytes": 8,
        "n": 16,
        "pass_gate": pass_gate,
        "controls_ok": pass_gate,
        "matched_accuracy": matched,
        "target_accuracy": 0.25,
        "best_control_accuracy": best_control,
        "best_control_name": "zero_source",
        "control_leak_over_target": best_control - 0.25,
        "delta_vs_target": matched - 0.25,
        "delta_vs_best_control": matched - best_control,
        "paired_ci95_low_vs_target": 0.2,
        "oracle_accuracy": 0.875,
        "random_same_byte_accuracy": 0.25,
        "structured_text_matched_accuracy": 0.25,
    }


def test_competitor_basis_table_keeps_pending_iclr_rows_explicit(tmp_path) -> None:
    common = {
        "headline": {
            "pass_gate": True,
            "live_rows": 1,
            "live_pass_rows": 1,
            "global_common_basis_rows": 1,
            "global_common_basis_pass_rows": 0,
            "procrustes_common_basis_rows": 1,
            "procrustes_common_basis_pass_rows": 0,
            "ridge_cca_common_basis_rows": 1,
            "ridge_cca_common_basis_pass_rows": 0,
            "ridge_cca_stack_rows": 1,
            "ridge_cca_stack_pass_rows": 0,
            "lstirp_rows": 1,
            "lstirp_pass_rows": 0,
            "lstirp_stack_rows": 1,
            "lstirp_stack_pass_rows": 0,
            "sinkhorn_ot_rows": 1,
            "sinkhorn_ot_pass_rows": 0,
            "sinkhorn_ot_stack_rows": 1,
            "sinkhorn_ot_stack_pass_rows": 0,
            "gw_rows": 1,
            "gw_pass_rows": 0,
            "gw_stack_rows": 1,
            "gw_stack_pass_rows": 0,
            "relative_anchor_innovation_rows": 1,
            "relative_anchor_innovation_pass_rows": 0,
            "relative_anchor_rank_innovation_rows": 1,
            "relative_anchor_rank_innovation_pass_rows": 0,
            "diagnostic_rows": 1,
            "diagnostic_pass_rows": 0,
        },
        "rows": [
            _common_row("live", pass_gate=True, matched=0.625, best_control=0.25),
            _common_row("global_common_basis", pass_gate=False, matched=0.875, best_control=0.625),
            _common_row("procrustes_common_basis", pass_gate=False, matched=0.875, best_control=0.875),
            _common_row("ridge_cca_common_basis", pass_gate=False, matched=0.75, best_control=0.5),
            _common_row("ridge_cca_local_stack", pass_gate=False, matched=0.625, best_control=0.375),
            _common_row("lstirp_relative_translation", pass_gate=False, matched=0.75, best_control=0.5),
            _common_row("lstirp_relative_local_stack", pass_gate=False, matched=0.625, best_control=0.375),
            _common_row("sinkhorn_ot_transport", pass_gate=False, matched=0.75, best_control=0.5),
            _common_row("sinkhorn_ot_local_stack", pass_gate=False, matched=0.625, best_control=0.375),
            _common_row("gw_transport", pass_gate=False, matched=0.75, best_control=0.5),
            _common_row("gw_local_stack", pass_gate=False, matched=0.625, best_control=0.375),
            _common_row("relative_anchor_common_basis", pass_gate=False, matched=0.75, best_control=0.375),
            _common_row("relative_anchor_innovation_stack", pass_gate=False, matched=0.75, best_control=0.375),
            _common_row("relative_anchor_rank_innovation_stack", pass_gate=False, matched=0.625, best_control=0.375),
            _common_row("diagnostic_ablation", pass_gate=False, matched=0.875, best_control=0.5),
        ],
    }
    systems = {
        "headline": {
            "packet_record_bytes": 11,
            "source_text_exposed": False,
            "source_kv_exposed": False,
            "max_resident_sparse_decode_p50_us": 5.2,
        }
    }
    common_path = tmp_path / "common.json"
    systems_path = tmp_path / "systems.json"
    common_path.write_text(json.dumps(common), encoding="utf-8")
    systems_path.write_text(json.dumps(systems), encoding="utf-8")

    payload = build_competitor_basis_table(
        common_basis_path=common_path,
        systems_path=systems_path,
        output_dir=tmp_path / "out",
    )

    assert payload["headline"]["common_basis_pass_gate"] is True
    assert payload["headline"]["iclr_competitor_complete"] is False
    assert payload["headline"]["pending_required_rows"] >= 3
    by_id = {row["method_id"]: row for row in payload["rows"]}
    assert by_id["candidate_local_residual_norm"]["status"] == "passes_strict_controls"
    assert by_id["global_public_anchor_dot"]["status"] == "fails_controls"
    assert by_id["orthogonal_procrustes_dot"]["status"] == "fails_controls"
    assert by_id["ridge_cca_dot"]["status"] == "fails_controls"
    assert by_id["ridge_cca_residual_norm_stack"]["status"] == "fails_controls"
    assert by_id["lstirp_inverse_relative_dot"]["status"] == "fails_controls"
    assert by_id["lstirp_inverse_relative_residual_norm_stack"]["status"] == "fails_controls"
    assert by_id["sinkhorn_ot_dot"]["status"] == "fails_controls"
    assert by_id["sinkhorn_ot_residual_norm_stack"]["status"] == "fails_controls"
    assert by_id["gromov_wasserstein_dot"]["status"] == "fails_controls"
    assert by_id["gromov_wasserstein_residual_norm_stack"]["status"] == "fails_controls"
    assert by_id["relative_representations_anchor_dot"]["status"] == "fails_controls"
    assert by_id["relative_anchor_innovation_residual_norm_stack"]["status"] == "fails_controls"
    assert by_id["relative_anchor_rank_innovation_residual_norm_stack"]["status"] == "fails_controls"
    assert "ridge_cca_svcca_model_stitching" not in by_id
    assert "lstirp_inverse_relative" not in by_id
    assert "ot_lstirp_gw" not in by_id
    assert "relative_representations" not in by_id
    assert by_id["structured_text_8b"]["source_text_exposed"] is True
    assert by_id["c2c_cache_fuser"]["source_kv_exposed"] is True
    assert (tmp_path / "out" / "candidate_local_competitor_basis_table.json").exists()
    assert (tmp_path / "out" / "manifest.json").exists()
