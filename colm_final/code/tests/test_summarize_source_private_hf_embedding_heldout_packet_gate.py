from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_source_private_hf_embedding_heldout_packet_gate import summarize


def _row(direction: str, budget: int, *, passed: bool, oracle: float = 0.75) -> dict:
    return {
        "run_dir": "results/frozen_embedding_probe",
        "feature_model": "fake/model",
        "text_feature_mode": "hf_last_mean",
        "receiver_mode": "atom_ridge",
        "min_decision_score": 0.7,
        "top_k": 12,
        "min_score": 0.01,
        "ridge": 0.25,
        "direction": direction,
        "budget_bytes": budget,
        "n": 32,
        "target_accuracy": 0.25,
        "learned_synonym_dictionary_accuracy": 0.5,
        "best_control_accuracy": 0.25,
        "learned_minus_target": 0.25,
        "learned_minus_best_control": 0.25,
        "oracle_learned_candidate_atoms_accuracy": oracle,
        "top_atom_knockout_lift_reduction": 1.0,
        "paired_ci95_low_vs_target": 0.16,
        "paired_ci95_high_vs_target": 0.31,
        "controls_ok": True,
        "pass_gate": passed,
    }


def test_summarize_frozen_embedding_rows_counts_near_misses() -> None:
    rows = [
        _row("core_to_holdout", 4, passed=False, oracle=0.75),
        _row("holdout_to_core", 4, passed=True, oracle=0.875),
        _row("same_family_all", 4, passed=True, oracle=0.875),
    ]
    summary = summarize(rows, semantic_anchor={"pass_gate": True, "rows": 18, "pass_rows": 18})

    assert summary["pass_gate"] is False
    assert summary["pass_rows"] == 2
    assert summary["near_miss_rows"] == 3
    assert summary["semantic_anchor_reference"]["pass_rows"] == 18


def test_summary_payload_is_json_serializable(tmp_path: Path) -> None:
    rows = [_row("core_to_holdout", 4, passed=True, oracle=0.875), _row("holdout_to_core", 4, passed=True)]
    summary = summarize(rows, semantic_anchor=None)
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    assert json.loads(path.read_text(encoding="utf-8"))["pass_gate"] is True
