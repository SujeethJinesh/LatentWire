from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT = pathlib.Path("results/source_private_train_donor_antishuffle_locked_rate_frontier_20260501")
DEFAULT_VALIDATION_RUNS = (
    pathlib.Path("results/source_private_train_donor_antishuffle_locked_frontier_seed47_n128"),
    pathlib.Path("results/source_private_train_donor_antishuffle_seed53_n128_budget10"),
    pathlib.Path("results/source_private_train_donor_antishuffle_locked_frontier_seed53_n128"),
    pathlib.Path("results/source_private_train_donor_antishuffle_seed59_n128_budget10"),
    pathlib.Path("results/source_private_train_donor_antishuffle_locked_frontier_seed59_n128"),
)
DEFAULT_EVAL_RUNS = (
    pathlib.Path("results/source_private_train_donor_antishuffle_seed47_n512_budget14"),
    pathlib.Path("results/source_private_train_donor_antishuffle_seed53_n512_budget12_14"),
    pathlib.Path("results/source_private_train_donor_antishuffle_seed59_n512_budget12_14"),
)
DEFAULT_BUDGETS = (10, 12, 14, 16)
DEFAULT_DIRECTIONS = ("core_to_holdout", "holdout_to_core")
TEXT_ACCESS_CONTROLS = {
    "answer_only_text",
    "structured_text_matched",
    "public_only_sidecar",
    "target_derived_sidecar",
}
MIN_LIFT_OVER_TARGET = 0.15
MIN_GAP_OVER_CONTROL = 0.10
MIN_IMPROVEMENT_OVER_BASE = 0.03


def _resolve(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else ROOT / path


def _rel(path: pathlib.Path) -> str:
    resolved = _resolve(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary_path(path: pathlib.Path) -> pathlib.Path:
    resolved = _resolve(path)
    if resolved.is_dir():
        return resolved / "candidate_conditioned_packet_builder_smoke.json"
    return resolved


def _source_private_control_fields(summary: dict[str, Any], direction_payload: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics", {})
    target = float(summary.get("target_accuracy", 0.0))
    candidate = float(summary.get("candidate_conditioned_packet_accuracy", 0.0))
    base = float(summary.get("base_matched_accuracy", 0.0))
    controls = [
        control
        for control in direction_payload.get("source_destroying_controls", [])
        if control not in TEXT_ACCESS_CONTROLS and control in metrics
    ]
    if not controls:
        return {}
    best_name = max(controls, key=lambda control: float(metrics[control]["accuracy"]))
    best_accuracy = float(metrics[best_name]["accuracy"])
    controls_ok = all(float(metrics[control]["accuracy"]) <= target + 0.03 for control in controls)
    ci_target = float(summary.get("paired_bootstrap_vs_target", {}).get("ci95_low", 0.0))
    ci_base = float(summary.get("paired_bootstrap_vs_base", {}).get("ci95_low", 0.0))
    oracle = float(summary.get("oracle_candidate_conditioned_packet_accuracy", 0.0))
    return {
        "source_private_best_control_name": best_name,
        "source_private_best_control_accuracy": best_accuracy,
        "source_private_controls_ok": controls_ok,
        "source_private_candidate_minus_best_control": candidate - best_accuracy,
        "source_private_selection_pass_gate": (
            candidate >= target + MIN_LIFT_OVER_TARGET
            and candidate >= best_accuracy + MIN_GAP_OVER_CONTROL
            and candidate >= base + MIN_IMPROVEMENT_OVER_BASE
            and controls_ok
            and ci_target > 0.05
            and ci_base >= 0.0
            and oracle >= 0.80
        ),
    }


def _direction_rows_from_summary(path: pathlib.Path) -> list[dict[str, Any]]:
    direction_payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for summary in direction_payload.get("budget_summaries", []):
        rows.append(
            {
                "direction": direction_payload["direction"],
                "budget_bytes": summary["budget_bytes"],
                "n": summary["n"],
                "target_accuracy": summary["target_accuracy"],
                "base_matched_accuracy": summary["base_matched_accuracy"],
                "candidate_conditioned_packet_accuracy": summary[
                    "candidate_conditioned_packet_accuracy"
                ],
                "best_control_accuracy": summary["best_control_accuracy"],
                "best_control_name": summary["best_control_name"],
                "candidate_minus_target": summary["candidate_minus_target"],
                "candidate_minus_base": summary["candidate_minus_base"],
                "candidate_minus_best_control": summary["candidate_minus_best_control"],
                "paired_ci95_low_vs_target": summary["paired_bootstrap_vs_target"]["ci95_low"],
                "paired_ci95_low_vs_base": summary["paired_bootstrap_vs_base"]["ci95_low"],
                "oracle_candidate_conditioned_packet_accuracy": summary[
                    "oracle_candidate_conditioned_packet_accuracy"
                ],
                "controls_ok": summary["controls_ok"],
                "pass_gate": summary["pass_gate"],
                **_source_private_control_fields(summary, direction_payload),
            }
        )
    return rows


def _augment_payload_from_direction_summaries(payload: dict[str, Any], resolved: pathlib.Path) -> dict[str, Any]:
    direction_summaries = sorted(resolved.glob("*/summary.json"))
    if not direction_summaries:
        return payload
    detailed: dict[tuple[int, str], dict[str, Any]] = {}
    for direction_summary in direction_summaries:
        for row in _direction_rows_from_summary(direction_summary):
            detailed[(int(row["budget_bytes"]), str(row["direction"]))] = row
    if not detailed:
        return payload
    rows: list[dict[str, Any]] = []
    for row in payload.get("rows", []):
        key = (int(row["budget_bytes"]), str(row["direction"]))
        rows.append({**row, **detailed.get(key, {})})
    payload["rows"] = rows
    return payload


def _read_run(path: pathlib.Path) -> dict[str, Any]:
    summary_path = _summary_path(path)
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload = _augment_payload_from_direction_summaries(payload, _resolve(path))
        payload["_summary_path"] = summary_path
        payload["_input_path"] = _resolve(path)
        return payload

    resolved = _resolve(path)
    direction_summaries = sorted(resolved.glob("*/summary.json"))
    if not direction_summaries:
        raise FileNotFoundError(f"no top-level or per-direction summary found under {resolved}")
    rows: list[dict[str, Any]] = []
    seeds: list[int] = []
    for direction_path in direction_summaries:
        direction_payload = json.loads(direction_path.read_text(encoding="utf-8"))
        direction = direction_payload["direction"]
        train_seed = int(direction_payload["train_seed"])
        eval_seed = int(direction_payload["eval_seed"])
        seeds.append(min(train_seed, eval_seed))
        rows.extend(_direction_rows_from_summary(direction_path))
    if len(set(seeds)) != 1:
        raise ValueError(f"per-direction summaries under {resolved} imply multiple seeds: {sorted(set(seeds))}")
    return {
        "gate": "source_private_candidate_conditioned_packet_builder_smoke_partial",
        "seed": seeds[0],
        "rows": rows,
        "_summary_path": resolved,
        "_input_path": resolved,
    }


def _merge_runs_by_seed(paths: list[pathlib.Path]) -> dict[int, dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for path in paths:
        run = _read_run(path)
        seed = _run_seed(run)
        if seed not in merged:
            merged[seed] = {
                **run,
                "rows": list(run.get("rows", [])),
                "_source_paths": [run["_summary_path"]],
            }
            continue
        merged[seed]["rows"].extend(run.get("rows", []))
        merged[seed].setdefault("_source_paths", []).append(run["_summary_path"])
    return merged


def _run_seed(run: dict[str, Any]) -> int:
    seed = run.get("seed")
    if not isinstance(seed, int):
        raise ValueError(f"run at {run.get('_summary_path')} does not expose an integer seed")
    return seed


def _rows_by_budget_direction(run: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    for row in run.get("rows", []):
        budget = row.get("budget_bytes")
        direction = row.get("direction")
        if isinstance(budget, int) and isinstance(direction, str):
            rows[(budget, direction)] = row
    return rows


def _row_passes(
    row: dict[str, Any] | None,
    *,
    min_ci95_low_vs_base: float,
    control_scope: str,
) -> bool:
    if row is None:
        return False
    if control_scope == "source_private_controls":
        return (
            bool(row.get("source_private_selection_pass_gate"))
            and float(row.get("paired_ci95_low_vs_base", 0.0)) >= min_ci95_low_vs_base
        )
    if control_scope == "source_private_gap":
        return (
            float(row.get("candidate_conditioned_packet_accuracy", 0.0))
            >= float(row.get("target_accuracy", 0.0)) + MIN_LIFT_OVER_TARGET
            and float(row.get("source_private_candidate_minus_best_control", 0.0)) >= MIN_GAP_OVER_CONTROL
            and float(row.get("candidate_minus_base", 0.0)) >= MIN_IMPROVEMENT_OVER_BASE
            and float(row.get("paired_ci95_low_vs_base", 0.0)) >= min_ci95_low_vs_base
            and float(row.get("paired_ci95_low_vs_target", 0.0)) > 0.05
            and float(row.get("oracle_candidate_conditioned_packet_accuracy", 0.0)) >= 0.80
        )
    if control_scope != "all_controls":
        raise ValueError(f"unknown control scope {control_scope!r}")
    return (
        bool(row.get("pass_gate"))
        and bool(row.get("controls_ok", True))
        and float(row.get("paired_ci95_low_vs_base", 0.0)) >= min_ci95_low_vs_base
    )


def _metric(row: dict[str, Any] | None, key: str) -> float | None:
    if row is None or row.get(key) is None:
        return None
    return float(row[key])


def _compact_row(
    *,
    phase: str,
    policy: str,
    seed: int,
    direction: str,
    budget: int,
    row: dict[str, Any] | None,
    pass_gate: bool,
    source_path: pathlib.Path | None,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "policy": policy,
        "seed": seed,
        "direction": direction,
        "budget_bytes": budget,
        "n": None if row is None else row.get("n"),
        "candidate_accuracy": _metric(row, "candidate_conditioned_packet_accuracy"),
        "base_accuracy": _metric(row, "base_matched_accuracy"),
        "target_accuracy": _metric(row, "target_accuracy"),
        "best_control_name": None if row is None else row.get("best_control_name"),
        "best_control_accuracy": _metric(row, "best_control_accuracy"),
        "source_private_best_control_name": None if row is None else row.get("source_private_best_control_name"),
        "source_private_best_control_accuracy": _metric(row, "source_private_best_control_accuracy"),
        "source_private_controls_ok": None if row is None else row.get("source_private_controls_ok"),
        "candidate_minus_base": _metric(row, "candidate_minus_base"),
        "paired_ci95_low_vs_base": _metric(row, "paired_ci95_low_vs_base"),
        "controls_ok": None if row is None else bool(row.get("controls_ok", False)),
        "row_pass_gate": None if row is None else bool(row.get("pass_gate", False)),
        "selected_pass_gate": pass_gate,
        "source_path": None if source_path is None else _rel(source_path),
    }


def _select_global_budget(
    *,
    validation_rows: dict[int, dict[tuple[int, str], dict[str, Any]]],
    budgets: list[int],
    directions: list[str],
    min_validation_ci95_low_vs_base: float,
    validation_control_scope: str,
    validation_selector: str,
) -> int | None:
    for index, budget in enumerate(budgets):
        if all(
            _row_passes(
                validation_rows[seed].get((budget, direction)),
                min_ci95_low_vs_base=min_validation_ci95_low_vs_base,
                control_scope=validation_control_scope,
            )
            for seed in sorted(validation_rows)
            for direction in directions
        ):
            if validation_selector == "smallest_pass":
                return budget
            if validation_selector != "stable_interior":
                raise ValueError(f"unknown validation selector {validation_selector!r}")
            if index == 0 or index == len(budgets) - 1:
                continue
            lower_budget = budgets[index - 1]
            upper_budget = budgets[index + 1]
            if all(
                _row_passes(
                    validation_rows[seed].get((neighbor_budget, direction)),
                    min_ci95_low_vs_base=min_validation_ci95_low_vs_base,
                    control_scope=validation_control_scope,
                )
                for seed in sorted(validation_rows)
                for direction in directions
                for neighbor_budget in (lower_budget, upper_budget)
            ):
                return budget
    return None


def _select_per_seed_budget(
    *,
    validation_rows: dict[int, dict[tuple[int, str], dict[str, Any]]],
    budgets: list[int],
    directions: list[str],
    min_validation_ci95_low_vs_base: float,
    validation_control_scope: str,
    validation_selector: str,
) -> dict[int, int | None]:
    selected: dict[int, int | None] = {}
    for seed in sorted(validation_rows):
        selected[seed] = None
        for index, budget in enumerate(budgets):
            if all(
                _row_passes(
                    validation_rows[seed].get((budget, direction)),
                    min_ci95_low_vs_base=min_validation_ci95_low_vs_base,
                    control_scope=validation_control_scope,
                )
                for direction in directions
            ):
                if validation_selector == "smallest_pass":
                    selected[seed] = budget
                    break
                if validation_selector != "stable_interior":
                    raise ValueError(f"unknown validation selector {validation_selector!r}")
                if index == 0 or index == len(budgets) - 1:
                    continue
                lower_budget = budgets[index - 1]
                upper_budget = budgets[index + 1]
                if not all(
                    _row_passes(
                        validation_rows[seed].get((neighbor_budget, direction)),
                        min_ci95_low_vs_base=min_validation_ci95_low_vs_base,
                        control_scope=validation_control_scope,
                    )
                    for direction in directions
                    for neighbor_budget in (lower_budget, upper_budget)
                ):
                    continue
                selected[seed] = budget
                break
    return selected


def _policy_rows(
    *,
    policy: str,
    selected: dict[int, int | None],
    validation_runs: dict[int, dict[str, Any]],
    eval_runs: dict[int, dict[str, Any]],
    validation_rows: dict[int, dict[tuple[int, str], dict[str, Any]]],
    eval_rows: dict[int, dict[tuple[int, str], dict[str, Any]]],
    directions: list[str],
    min_validation_ci95_low_vs_base: float,
    min_eval_ci95_low_vs_base: float,
    validation_control_scope: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for seed in sorted(validation_runs):
        budget = selected.get(seed)
        for direction in directions:
            if budget is None:
                validation_row = None
                eval_row = None
                validation_pass = False
                eval_pass = False
                budget_for_row = -1
            else:
                budget_for_row = budget
                validation_row = validation_rows[seed].get((budget, direction))
                eval_row = eval_rows.get(seed, {}).get((budget, direction))
                validation_pass = _row_passes(
                    validation_row,
                    min_ci95_low_vs_base=min_validation_ci95_low_vs_base,
                    control_scope=validation_control_scope,
                )
                eval_pass = _row_passes(
                    eval_row,
                    min_ci95_low_vs_base=min_eval_ci95_low_vs_base,
                    control_scope="all_controls",
                )
            rows.append(
                _compact_row(
                    phase="validation",
                    policy=policy,
                    seed=seed,
                    direction=direction,
                    budget=budget_for_row,
                    row=validation_row,
                    pass_gate=validation_pass,
                    source_path=validation_runs[seed]["_summary_path"],
                )
            )
            eval_compact = _compact_row(
                phase="eval",
                policy=policy,
                seed=seed,
                direction=direction,
                budget=budget_for_row,
                row=eval_row,
                pass_gate=eval_pass,
                source_path=eval_runs.get(seed, {}).get("_summary_path"),
            )
            rows.append(eval_compact)
            selected_rows.append(eval_compact)
    return rows, selected_rows


def _policy_summary(
    *,
    policy: str,
    selected: dict[int, int | None],
    selected_eval_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    pass_rows = [row for row in selected_eval_rows if row["selected_pass_gate"]]
    present_rows = [row for row in selected_eval_rows if row["n"] is not None]
    return {
        "policy": policy,
        "selected_budget_by_seed": {str(seed): budget for seed, budget in sorted(selected.items())},
        "row_count": len(selected_eval_rows),
        "present_eval_rows": len(present_rows),
        "pass_rows": len(pass_rows),
        "pass_gate": len(pass_rows) == len(selected_eval_rows) and all(budget is not None for budget in selected.values()),
        "min_selected_candidate_accuracy": min(
            (row["candidate_accuracy"] for row in present_rows if row["candidate_accuracy"] is not None),
            default=None,
        ),
        "min_selected_candidate_minus_base": min(
            (row["candidate_minus_base"] for row in present_rows if row["candidate_minus_base"] is not None),
            default=None,
        ),
        "max_selected_best_control_accuracy": max(
            (row["best_control_accuracy"] for row in present_rows if row["best_control_accuracy"] is not None),
            default=None,
        ),
        "min_selected_ci95_low_vs_base": min(
            (row["paired_ci95_low_vs_base"] for row in pass_rows if row["paired_ci95_low_vs_base"] is not None),
            default=None,
        ),
    }


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "phase",
        "policy",
        "seed",
        "direction",
        "budget_bytes",
        "n",
        "candidate_accuracy",
        "base_accuracy",
        "target_accuracy",
        "best_control_name",
        "best_control_accuracy",
        "source_private_best_control_name",
        "source_private_best_control_accuracy",
        "source_private_controls_ok",
        "candidate_minus_base",
        "paired_ci95_low_vs_base",
        "controls_ok",
        "row_pass_gate",
        "selected_pass_gate",
        "source_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _validation_rule_text(*, validation_selector: str, min_validation_ci95_low_vs_base: float) -> str:
    suffix = (
        "Validation rows must pass every declared direction under the declared control scope and paired "
        f"CI95 lower bound vs base >= {min_validation_ci95_low_vs_base:.3f}."
    )
    if validation_selector == "smallest_pass":
        return "Choose the smallest byte budget in the declared budget list. " + suffix
    if validation_selector == "stable_interior":
        return (
            "Choose the smallest byte budget that is an interior point of a clean validation band: "
            "the selected budget, the previous declared budget, and the next declared budget must all pass. "
            + suffix
        )
    raise ValueError(f"unknown validation selector {validation_selector!r}")


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    primary = payload["policies"][payload["selection_scope"]]
    lines = [
        "# Train-Donor Anti-Shuffle Locked Rate Frontier",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- selection scope: `{payload['selection_scope']}`",
        f"- validation rule: `{payload['validation_rule']}`",
        f"- eval rule: `{payload['eval_rule']}`",
        f"- primary selected budget by seed: `{primary['selected_budget_by_seed']}`",
        "",
        "## Policy Summary",
        "",
        "| Policy | Selected budgets | Eval rows | Passing eval rows | Min candidate | Max best control | Min CI95 low vs base | Pass |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for summary in payload["policies"].values():
        lines.append(
            "| "
            f"{summary['policy']} | "
            f"{summary['selected_budget_by_seed']} | "
            f"{summary['row_count']} | "
            f"{summary['pass_rows']} | "
            f"{_fmt(summary['min_selected_candidate_accuracy'])} | "
            f"{_fmt(summary['max_selected_best_control_accuracy'])} | "
            f"{_fmt(summary['min_selected_ci95_low_vs_base'])} | "
            f"{summary['pass_gate']} |"
        )
    lines.extend(
        [
            "",
            "## Selected Eval Rows",
            "",
            "| Policy | Seed | Direction | Budget | Candidate | Base | Target | Best control | CI95 low vs base | Pass |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["selected_eval_rows"]:
        lines.append(
            "| "
            f"{row['policy']} | "
            f"{row['seed']} | "
            f"{row['direction']} | "
            f"{row['budget_bytes']} | "
            f"{_fmt(row['candidate_accuracy'])} | "
            f"{_fmt(row['base_accuracy'])} | "
            f"{_fmt(row['target_accuracy'])} | "
            f"{_fmt(row['best_control_accuracy'])} | "
            f"{_fmt(row['paired_ci95_low_vs_base'])} | "
            f"{row['selected_pass_gate']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This artifact separates byte-budget selection from final eval scoring. "
            "The validation frontier chooses the smallest budget that clears both cross-family directions, "
            "then the n512 eval surface is read only at that selected budget. "
            "It is a model-selection audit, not a new method or a public-benchmark result.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_locked_rate_frontier(
    *,
    output_dir: pathlib.Path,
    validation_run_paths: list[pathlib.Path],
    eval_run_paths: list[pathlib.Path],
    budgets: list[int],
    directions: list[str],
    selection_scope: str = "global",
    validation_control_scope: str = "all_controls",
    validation_selector: str = "smallest_pass",
    min_validation_ci95_low_vs_base: float = 0.05,
    min_eval_ci95_low_vs_base: float = 0.05,
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_runs = _merge_runs_by_seed(validation_run_paths)
    eval_runs = _merge_runs_by_seed(eval_run_paths)
    validation_rows = {seed: _rows_by_budget_direction(run) for seed, run in validation_runs.items()}
    eval_rows = {seed: _rows_by_budget_direction(run) for seed, run in eval_runs.items()}

    global_budget = _select_global_budget(
        validation_rows=validation_rows,
        budgets=budgets,
        directions=directions,
        min_validation_ci95_low_vs_base=min_validation_ci95_low_vs_base,
        validation_control_scope=validation_control_scope,
        validation_selector=validation_selector,
    )
    global_selected = {seed: global_budget for seed in validation_runs}
    per_seed_selected = _select_per_seed_budget(
        validation_rows=validation_rows,
        budgets=budgets,
        directions=directions,
        min_validation_ci95_low_vs_base=min_validation_ci95_low_vs_base,
        validation_control_scope=validation_control_scope,
        validation_selector=validation_selector,
    )

    all_rows: list[dict[str, Any]] = []
    all_selected_eval_rows: list[dict[str, Any]] = []
    policies: dict[str, dict[str, Any]] = {}
    for policy, selected in [("global", global_selected), ("per_seed", per_seed_selected)]:
        policy_rows, selected_eval_rows = _policy_rows(
            policy=policy,
            selected=selected,
            validation_runs=validation_runs,
            eval_runs=eval_runs,
            validation_rows=validation_rows,
            eval_rows=eval_rows,
            directions=directions,
            min_validation_ci95_low_vs_base=min_validation_ci95_low_vs_base,
            min_eval_ci95_low_vs_base=min_eval_ci95_low_vs_base,
            validation_control_scope=validation_control_scope,
        )
        all_rows.extend(policy_rows)
        all_selected_eval_rows.extend(selected_eval_rows)
        policies[policy] = _policy_summary(
            policy=policy,
            selected=selected,
            selected_eval_rows=selected_eval_rows,
        )

    if selection_scope not in policies:
        raise ValueError(f"unknown selection scope {selection_scope!r}; expected one of {sorted(policies)}")
    primary = policies[selection_scope]
    payload = {
        "gate": "source_private_train_donor_antishuffle_locked_rate_frontier",
        "selection_scope": selection_scope,
        "validation_control_scope": validation_control_scope,
        "validation_selector": validation_selector,
        "budgets": budgets,
        "directions": directions,
        "validation_runs": [_rel(path) for path in validation_run_paths],
        "eval_runs": [_rel(path) for path in eval_run_paths],
        "validation_rule": _validation_rule_text(
            validation_selector=validation_selector,
            min_validation_ci95_low_vs_base=min_validation_ci95_low_vs_base,
        ),
        "eval_rule": (
            "Report only rows at the preselected byte budget; each selected eval row must pass its original gate, "
            "keep controls clean, and have paired CI95 lower bound vs base "
            f">= {min_eval_ci95_low_vs_base:.3f}."
        ),
        "policies": policies,
        "rows": all_rows,
        "selected_eval_rows": [
            row for row in all_selected_eval_rows if row["policy"] == selection_scope
        ],
        "pass_gate": bool(primary["pass_gate"]),
        "pass_rule": "Primary selected policy must have all selected n512 eval rows present and passing.",
        "caveat": (
            "The current default validation surface is a small frozen n128 frontier. "
            "For the final ICLR paper, rerun the same artifact with a larger train-only validation split "
            "and then bridge to public benchmarks."
        ),
    }

    json_path = output_dir / "train_donor_locked_rate_frontier.json"
    csv_path = output_dir / "train_donor_locked_rate_frontier.csv"
    md_path = output_dir / "train_donor_locked_rate_frontier.md"
    manifest_path = output_dir / "manifest.json"
    manifest_md_path = output_dir / "manifest.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(csv_path, all_rows)
    _write_markdown(md_path, payload)
    manifest = {
        "artifacts": [json_path.name, csv_path.name, md_path.name, manifest_path.name, manifest_md_path.name],
        "artifact_sha256": {
            json_path.name: _sha256_file(json_path),
            csv_path.name: _sha256_file(csv_path),
            md_path.name: _sha256_file(md_path),
        },
        "pass_gate": payload["pass_gate"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest_md_path.write_text(
        "\n".join(
            [
                "# Train-Donor Anti-Shuffle Locked Rate Frontier Manifest",
                "",
                f"- pass gate: `{payload['pass_gate']}`",
                f"- json: `{json_path.name}`",
                f"- csv: `{csv_path.name}`",
                f"- markdown: `{md_path.name}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def _parse_run_paths(values: list[str] | None, defaults: tuple[pathlib.Path, ...]) -> list[pathlib.Path]:
    if not values:
        return list(defaults)
    paths: list[pathlib.Path] = []
    for value in values:
        if "=" in value:
            _, raw_path = value.split("=", 1)
            paths.append(pathlib.Path(raw_path))
        else:
            paths.append(pathlib.Path(value))
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-run", action="append", help="Validation run path, optionally seed=PATH.")
    parser.add_argument("--eval-run", action="append", help="Eval run path, optionally seed=PATH.")
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--directions", nargs="+", default=list(DEFAULT_DIRECTIONS))
    parser.add_argument("--selection-scope", choices=("global", "per_seed"), default="global")
    parser.add_argument(
        "--validation-control-scope",
        choices=("all_controls", "source_private_controls", "source_private_gap"),
        default="all_controls",
    )
    parser.add_argument(
        "--validation-selector",
        choices=("smallest_pass", "stable_interior"),
        default="smallest_pass",
    )
    parser.add_argument("--min-validation-ci95-low-vs-base", type=float, default=0.05)
    parser.add_argument("--min-eval-ci95-low-vs-base", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_locked_rate_frontier(
        output_dir=args.output_dir,
        validation_run_paths=_parse_run_paths(args.validation_run, DEFAULT_VALIDATION_RUNS),
        eval_run_paths=_parse_run_paths(args.eval_run, DEFAULT_EVAL_RUNS),
        budgets=args.budgets,
        directions=args.directions,
        selection_scope=args.selection_scope,
        validation_control_scope=args.validation_control_scope,
        validation_selector=args.validation_selector,
        min_validation_ci95_low_vs_base=args.min_validation_ci95_low_vs_base,
        min_eval_ci95_low_vs_base=args.min_eval_ci95_low_vs_base,
    )


if __name__ == "__main__":
    main()
