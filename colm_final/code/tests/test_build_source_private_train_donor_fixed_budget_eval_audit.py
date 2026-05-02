from __future__ import annotations

import json

from scripts.build_source_private_train_donor_fixed_budget_eval_audit import (
    build_fixed_budget_eval_audit,
)


def _write_run(path, *, seed: int, pass_gate: bool = True) -> None:
    path.write_text(
        json.dumps(
            {
                "seed": seed,
                "rows": [
                    {
                        "direction": "core_to_holdout",
                        "budget_bytes": 12,
                        "n": 512,
                        "candidate_conditioned_packet_accuracy": 0.75,
                        "base_matched_accuracy": 0.625,
                        "target_accuracy": 0.25,
                        "best_control_name": "random_same_byte",
                        "best_control_accuracy": 0.26,
                        "candidate_minus_base": 0.125,
                        "paired_ci95_low_vs_base": 0.09,
                        "controls_ok": True,
                        "pass_gate": pass_gate,
                    },
                    {
                        "direction": "holdout_to_core",
                        "budget_bytes": 12,
                        "n": 512,
                        "candidate_conditioned_packet_accuracy": 0.65,
                        "base_matched_accuracy": 0.50,
                        "target_accuracy": 0.25,
                        "best_control_name": "random_same_byte",
                        "best_control_accuracy": 0.25,
                        "candidate_minus_base": 0.15,
                        "paired_ci95_low_vs_base": 0.11,
                        "controls_ok": True,
                        "pass_gate": pass_gate,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_fixed_budget_eval_audit_passes_all_expected_rows(tmp_path) -> None:
    run_a = tmp_path / "seed1.json"
    run_b = tmp_path / "seed2.json"
    _write_run(run_a, seed=1)
    _write_run(run_b, seed=2)

    payload = build_fixed_budget_eval_audit(
        eval_runs=[run_a, run_b],
        budget=12,
        directions=("core_to_holdout", "holdout_to_core"),
        output_dir=tmp_path / "out",
    )

    assert payload["pass_gate"] is True
    assert payload["headline"]["rows"] == 4
    assert payload["headline"]["pass_rows"] == 4
    assert payload["headline"]["min_candidate_minus_base"] == 0.125
    assert (tmp_path / "out" / "fixed_budget_eval_audit.json").exists()
    assert (tmp_path / "out" / "manifest.json").exists()
