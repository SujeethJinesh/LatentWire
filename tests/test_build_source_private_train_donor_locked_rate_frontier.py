from __future__ import annotations

import json

from scripts import build_source_private_train_donor_locked_rate_frontier as frontier


def _row(
    *,
    direction: str,
    budget: int,
    pass_gate: bool,
    candidate: float = 0.75,
    base: float = 0.625,
    target: float = 0.25,
    control: float = 0.26,
    ci_low: float = 0.08,
) -> dict[str, object]:
    return {
        "direction": direction,
        "budget_bytes": budget,
        "n": 128,
        "target_accuracy": target,
        "base_matched_accuracy": base,
        "candidate_conditioned_packet_accuracy": candidate,
        "best_control_accuracy": control,
        "best_control_name": "private_random_source_atoms",
        "candidate_minus_base": candidate - base,
        "paired_ci95_low_vs_base": ci_low,
        "controls_ok": pass_gate,
        "pass_gate": pass_gate,
        "oracle_candidate_conditioned_packet_accuracy": 1.0,
        "paired_ci95_low_vs_target": 0.20,
        "source_private_best_control_accuracy": control,
        "source_private_best_control_name": "private_random_source_atoms",
        "source_private_candidate_minus_best_control": candidate - control,
        "source_private_controls_ok": pass_gate,
        "source_private_selection_pass_gate": pass_gate,
    }


def _write_run(path, *, seed: int, rows: list[dict[str, object]]) -> None:
    path.mkdir(parents=True)
    payload = {
        "gate": "source_private_candidate_conditioned_packet_builder_smoke",
        "seed": seed,
        "rows": rows,
        "headline": {"cross_family_pass": any(row["pass_gate"] for row in rows)},
        "pass_gate": any(row["pass_gate"] for row in rows),
    }
    (path / "candidate_conditioned_packet_builder_smoke.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_global_locked_frontier_selects_smallest_validation_budget_and_scores_eval(tmp_path) -> None:
    validation_paths = []
    eval_paths = []
    for seed in [47, 53]:
        validation_path = tmp_path / f"validation_{seed}"
        eval_path = tmp_path / f"eval_{seed}"
        validation_paths.append(validation_path)
        eval_paths.append(eval_path)
        _write_run(
            validation_path,
            seed=seed,
            rows=[
                _row(direction="core_to_holdout", budget=10, pass_gate=True),
                _row(direction="holdout_to_core", budget=10, pass_gate=False, ci_low=0.0),
                _row(direction="core_to_holdout", budget=12, pass_gate=True),
                _row(direction="holdout_to_core", budget=12, pass_gate=True),
            ],
        )
        _write_run(
            eval_path,
            seed=seed,
            rows=[
                _row(direction="core_to_holdout", budget=12, pass_gate=True, ci_low=0.09),
                _row(direction="holdout_to_core", budget=12, pass_gate=True, ci_low=0.11),
            ],
        )

    payload = frontier.build_locked_rate_frontier(
        output_dir=tmp_path / "frontier",
        validation_run_paths=validation_paths,
        eval_run_paths=eval_paths,
        budgets=[10, 12, 14],
        directions=["core_to_holdout", "holdout_to_core"],
        selection_scope="global",
    )

    assert payload["pass_gate"] is True
    assert payload["policies"]["global"]["selected_budget_by_seed"] == {"47": 12, "53": 12}
    assert payload["policies"]["global"]["pass_rows"] == 4
    assert (tmp_path / "frontier" / "train_donor_locked_rate_frontier.json").exists()
    assert (tmp_path / "frontier" / "train_donor_locked_rate_frontier.csv").exists()
    assert (tmp_path / "frontier" / "manifest.json").exists()


def test_locked_frontier_fails_when_selected_eval_row_is_missing(tmp_path) -> None:
    validation_path = tmp_path / "validation_47"
    eval_path = tmp_path / "eval_47"
    _write_run(
        validation_path,
        seed=47,
        rows=[
            _row(direction="core_to_holdout", budget=12, pass_gate=True),
            _row(direction="holdout_to_core", budget=12, pass_gate=True),
        ],
    )
    _write_run(
        eval_path,
        seed=47,
        rows=[_row(direction="core_to_holdout", budget=12, pass_gate=True)],
    )

    payload = frontier.build_locked_rate_frontier(
        output_dir=tmp_path / "frontier",
        validation_run_paths=[validation_path],
        eval_run_paths=[eval_path],
        budgets=[12],
        directions=["core_to_holdout", "holdout_to_core"],
        selection_scope="global",
    )

    assert payload["pass_gate"] is False
    assert payload["policies"]["global"]["present_eval_rows"] == 1
    assert payload["policies"]["global"]["row_count"] == 2


def test_stable_interior_selector_skips_brittle_minimum_budget(tmp_path) -> None:
    validation_path = tmp_path / "validation_47"
    eval_path = tmp_path / "eval_47"
    _write_run(
        validation_path,
        seed=47,
        rows=[
            _row(direction="core_to_holdout", budget=10, pass_gate=True),
            _row(direction="holdout_to_core", budget=10, pass_gate=True),
            _row(direction="core_to_holdout", budget=12, pass_gate=True),
            _row(direction="holdout_to_core", budget=12, pass_gate=True),
            _row(direction="core_to_holdout", budget=14, pass_gate=True),
            _row(direction="holdout_to_core", budget=14, pass_gate=True),
            _row(direction="core_to_holdout", budget=16, pass_gate=False),
            _row(direction="holdout_to_core", budget=16, pass_gate=False),
        ],
    )
    _write_run(
        eval_path,
        seed=47,
        rows=[
            _row(direction="core_to_holdout", budget=10, pass_gate=False, ci_low=0.01),
            _row(direction="holdout_to_core", budget=10, pass_gate=False, ci_low=0.01),
            _row(direction="core_to_holdout", budget=12, pass_gate=True, ci_low=0.09),
            _row(direction="holdout_to_core", budget=12, pass_gate=True, ci_low=0.11),
        ],
    )

    payload = frontier.build_locked_rate_frontier(
        output_dir=tmp_path / "frontier",
        validation_run_paths=[validation_path],
        eval_run_paths=[eval_path],
        budgets=[10, 12, 14, 16],
        directions=["core_to_holdout", "holdout_to_core"],
        selection_scope="global",
        validation_selector="stable_interior",
    )

    assert payload["pass_gate"] is True
    assert payload["validation_selector"] == "stable_interior"
    assert payload["policies"]["global"]["selected_budget_by_seed"] == {"47": 12}
    assert payload["policies"]["global"]["pass_rows"] == 2


def test_source_private_gap_selector_uses_gap_without_validation_control_target_band(tmp_path) -> None:
    validation_path = tmp_path / "validation_47"
    eval_path = tmp_path / "eval_47"
    _write_run(
        validation_path,
        seed=47,
        rows=[
            _row(direction="core_to_holdout", budget=10, pass_gate=False, control=0.31),
            _row(direction="holdout_to_core", budget=10, pass_gate=False, control=0.30),
            _row(direction="core_to_holdout", budget=12, pass_gate=False, control=0.31),
            _row(direction="holdout_to_core", budget=12, pass_gate=False, control=0.30),
            _row(direction="core_to_holdout", budget=14, pass_gate=False, control=0.31),
            _row(direction="holdout_to_core", budget=14, pass_gate=False, control=0.30),
        ],
    )
    _write_run(
        eval_path,
        seed=47,
        rows=[
            _row(direction="core_to_holdout", budget=12, pass_gate=True, ci_low=0.09),
            _row(direction="holdout_to_core", budget=12, pass_gate=True, ci_low=0.11),
        ],
    )

    payload = frontier.build_locked_rate_frontier(
        output_dir=tmp_path / "frontier",
        validation_run_paths=[validation_path],
        eval_run_paths=[eval_path],
        budgets=[10, 12, 14],
        directions=["core_to_holdout", "holdout_to_core"],
        selection_scope="global",
        validation_control_scope="source_private_gap",
        validation_selector="stable_interior",
    )

    assert payload["pass_gate"] is True
    assert payload["validation_control_scope"] == "source_private_gap"
    assert payload["policies"]["global"]["selected_budget_by_seed"] == {"47": 12}
