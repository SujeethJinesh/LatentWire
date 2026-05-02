from __future__ import annotations

import json

from scripts.summarize_source_private_pq_control_regularized_receiver import summarize_runs


def _write_run(path, *, pass_gate: bool, disjoint: bool, learned: float, l2: float) -> None:
    path.mkdir(parents=True)
    payload = {
        "pass_gate": pass_gate,
        "variant": "utility_protected_hadamard",
        "remap_slot_seed": 101,
        "eval_examples": 16,
        "train_eval_id_intersection_count": 0 if disjoint else 16,
        "matched_weight": 12.0,
        "control_weight": 0.25,
        "target_weight": 0.5,
        "deranged_weight": 0.0,
        "random_rounds": 0,
        "summary": {
            "learned_source_accuracy": learned,
            "l2_source_accuracy": l2,
            "target_only_accuracy": 0.25,
            "best_control_condition": "random_same_byte",
            "best_control_accuracy": 0.27,
            "learned_minus_best_control": learned - 0.27,
            "learned_minus_target": learned - 0.25,
            "learned_minus_l2": learned - l2,
            "learned_metrics": {"deranged_public_table": {"accuracy": 0.25}},
            "paired_bootstrap": {
                "learned_source_vs_best_control": {"ci95_low": 0.12},
                "learned_source_vs_target": {"ci95_low": 0.14},
            },
        },
    }
    (path / "run_summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_pq_control_regularized_receiver_summary_marks_overlap_positive_disjoint_blocker(tmp_path) -> None:
    overlap = tmp_path / "overlap"
    disjoint = tmp_path / "disjoint"
    _write_run(overlap, pass_gate=True, disjoint=False, learned=0.50, l2=0.50)
    _write_run(disjoint, pass_gate=False, disjoint=True, learned=0.27, l2=0.27)

    payload = summarize_runs(run_dirs=[overlap, disjoint], output_dir=tmp_path / "summary")

    assert payload["pass_gate"] is True
    assert payload["headline"]["overlap_pass_rows"] == 1
    assert payload["headline"]["disjoint_pass_rows"] == 0
    assert payload["headline"]["max_disjoint_l2_accuracy"] == 0.27
    assert (tmp_path / "summary" / "pq_control_regularized_receiver_summary.md").exists()
