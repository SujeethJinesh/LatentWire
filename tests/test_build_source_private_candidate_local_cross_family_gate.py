from __future__ import annotations

import json

from scripts.build_source_private_candidate_local_cross_family_gate import build_cross_family_gate


def _row(row_group: str, direction: str, *, pass_gate: bool, controls_ok: bool, matched: float) -> dict[str, object]:
    return {
        "row_group": row_group,
        "method": row_group,
        "seed": 47,
        "direction": direction,
        "budget_bytes": 8,
        "n": 16,
        "pass_gate": pass_gate,
        "controls_ok": controls_ok,
        "matched_accuracy": matched,
        "target_accuracy": 0.25,
        "best_control_accuracy": 0.25 if controls_ok else matched,
        "best_control_name": "zero_source" if controls_ok else "permuted_teacher_receiver",
        "control_leak_over_target": 0.0 if controls_ok else matched - 0.25,
        "delta_vs_target": matched - 0.25,
        "delta_vs_best_control": matched - (0.25 if controls_ok else matched),
        "paired_ci95_low_vs_target": 0.2,
        "oracle_accuracy": 0.875,
    }


def test_cross_family_gate_separates_live_from_rr_partial(tmp_path) -> None:
    rows = [
        _row("live", "core_to_holdout", pass_gate=True, controls_ok=True, matched=0.625),
        _row("live", "holdout_to_core", pass_gate=True, controls_ok=True, matched=0.5),
        _row("live", "same_family_all", pass_gate=True, controls_ok=True, matched=0.5625),
        _row("relative_anchor_common_basis", "core_to_holdout", pass_gate=True, controls_ok=True, matched=1.0),
        _row("relative_anchor_common_basis", "holdout_to_core", pass_gate=False, controls_ok=True, matched=0.25),
        _row("relative_anchor_common_basis", "same_family_all", pass_gate=True, controls_ok=True, matched=0.625),
        _row("global_common_basis", "core_to_holdout", pass_gate=False, controls_ok=False, matched=0.875),
        _row("global_common_basis", "holdout_to_core", pass_gate=False, controls_ok=False, matched=0.625),
        _row("global_common_basis", "same_family_all", pass_gate=False, controls_ok=False, matched=0.75),
    ]
    common_path = tmp_path / "common.json"
    common_path.write_text(json.dumps({"rows": rows}), encoding="utf-8")

    payload = build_cross_family_gate(common_basis_path=common_path, output_dir=tmp_path / "out")

    assert payload["headline"]["pass_gate"] is True
    assert payload["headline"]["live_cross_family_pass_rows"] == 2
    assert payload["headline"]["live_same_family_pass_rows"] == 1
    assert payload["headline"]["rr_cross_family_pass_rows"] == 1
    assert payload["headline"]["rr_holdout_to_core_pass_rows"] == 0
    assert payload["headline"]["control_leaky_cross_family_groups"] == ["global_common_basis"]
    assert (tmp_path / "out" / "candidate_local_cross_family_gate.json").exists()
    assert (tmp_path / "out" / "manifest.json").exists()
