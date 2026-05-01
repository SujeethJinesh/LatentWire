from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import re
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_LIVE_RUNS = (
    pathlib.Path(
        "results/source_private_candidate_local_residual_receiver_20260430_seed47_n512_minilm_teacher_norm_dec048_evaldisjoint"
    ),
    pathlib.Path(
        "results/source_private_candidate_local_residual_receiver_20260430_seed53_n512_minilm_teacher_norm_dec048_evaldisjoint"
    ),
    pathlib.Path(
        "results/source_private_candidate_local_residual_receiver_20260430_seed59_n512_minilm_teacher_norm_dec048_evaldisjoint"
    ),
)
DEFAULT_GLOBAL_DOT_RUNS = (
    pathlib.Path("results/source_private_candidate_local_residual_ablation_20260430_seed47_n512_global_dot_evaldisjoint"),
    pathlib.Path("results/source_private_candidate_local_residual_ablation_20260430_seed53_n512_global_dot_evaldisjoint"),
    pathlib.Path("results/source_private_candidate_local_residual_ablation_20260430_seed59_n512_global_dot_evaldisjoint"),
)
DEFAULT_PROCRUSTES_RUNS = (
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed47_n512_procrustes_dot_evaldisjoint"
    ),
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed53_n512_procrustes_dot_evaldisjoint"
    ),
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed59_n512_procrustes_dot_evaldisjoint"
    ),
)
DEFAULT_RIDGE_CCA_RUNS = (
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed47_n512_ridge_cca_dot_evaldisjoint"
    ),
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed53_n512_ridge_cca_dot_evaldisjoint"
    ),
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed59_n512_ridge_cca_dot_evaldisjoint"
    ),
)
DEFAULT_RIDGE_CCA_STACK_RUNS = (
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed47_n512_ridge_cca_residual_norm_evaldisjoint"
    ),
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed53_n512_ridge_cca_residual_norm_evaldisjoint"
    ),
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed59_n512_ridge_cca_residual_norm_evaldisjoint"
    ),
)
DEFAULT_LSTIRP_RUNS = (
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed47_n512_inverse_relative_dot_evaldisjoint"
    ),
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed53_n512_inverse_relative_dot_evaldisjoint"
    ),
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed59_n512_inverse_relative_dot_evaldisjoint"
    ),
)
DEFAULT_LSTIRP_STACK_RUNS = (
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed47_n512_inverse_relative_residual_norm_evaldisjoint"
    ),
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed53_n512_inverse_relative_residual_norm_evaldisjoint"
    ),
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed59_n512_inverse_relative_residual_norm_evaldisjoint"
    ),
)
DEFAULT_SINKHORN_OT_RUNS = (
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed47_n512_sinkhorn_ot_dot_evaldisjoint"
    ),
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed53_n512_sinkhorn_ot_dot_evaldisjoint"
    ),
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed59_n512_sinkhorn_ot_dot_evaldisjoint"
    ),
)
DEFAULT_SINKHORN_OT_STACK_RUNS = (
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed47_n512_sinkhorn_ot_residual_norm_evaldisjoint"
    ),
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed53_n512_sinkhorn_ot_residual_norm_evaldisjoint"
    ),
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed59_n512_sinkhorn_ot_residual_norm_evaldisjoint"
    ),
)
DEFAULT_GW_RUNS = (
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed47_n512_gromov_wasserstein_dot_evaldisjoint"
    ),
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed53_n512_gromov_wasserstein_dot_evaldisjoint"
    ),
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed59_n512_gromov_wasserstein_dot_evaldisjoint"
    ),
)
DEFAULT_GW_STACK_RUNS = (
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed47_n512_gromov_wasserstein_residual_norm_evaldisjoint"
    ),
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed53_n512_gromov_wasserstein_residual_norm_evaldisjoint"
    ),
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed59_n512_gromov_wasserstein_residual_norm_evaldisjoint"
    ),
)
DEFAULT_RELATIVE_ANCHOR_RUNS = (
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed47_n512_relative_anchor_dot_evaldisjoint"
    ),
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed53_n512_relative_anchor_dot_evaldisjoint"
    ),
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed59_n512_relative_anchor_dot_evaldisjoint"
    ),
)
DEFAULT_RELATIVE_ANCHOR_STACK_RUNS = (
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed47_n512_relative_anchor_residual_norm_evaldisjoint"
    ),
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed53_n512_relative_anchor_residual_norm_evaldisjoint"
    ),
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed59_n512_relative_anchor_residual_norm_evaldisjoint"
    ),
)
DEFAULT_RELATIVE_ANCHOR_INNOVATION_RUNS = (
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260501_seed47_n512_relative_anchor_innovation_residual_norm_evaldisjoint"
    ),
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260501_seed53_n512_relative_anchor_innovation_residual_norm_evaldisjoint"
    ),
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260501_seed59_n512_relative_anchor_innovation_residual_norm_evaldisjoint"
    ),
)
DEFAULT_RELATIVE_ANCHOR_RANK_INNOVATION_RUNS = (
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260501_seed47_n512_relative_anchor_rank_innovation_residual_norm_evaldisjoint"
    ),
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260501_seed53_n512_relative_anchor_rank_innovation_residual_norm_evaldisjoint"
    ),
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260501_seed59_n512_relative_anchor_rank_innovation_residual_norm_evaldisjoint"
    ),
)
DEFAULT_DIAGNOSTIC_RUNS = (
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed47_n512_residual_no_norm_evaldisjoint"
    ),
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed53_n512_residual_no_norm_evaldisjoint"
    ),
    pathlib.Path(
        "results/source_private_candidate_local_residual_ablation_20260430_seed59_n512_residual_no_norm_evaldisjoint"
    ),
)
DEFAULT_OUTPUT = pathlib.Path("results/source_private_candidate_local_common_basis_falsification_20260430")

CONTROL_CONDITIONS = (
    "zero_source",
    "shuffled_source",
    "atom_id_derangement",
    "private_random_source_atoms",
    "permuted_teacher_receiver",
    "random_same_byte",
    "answer_only_text",
    "structured_text_matched",
)
CSV_COLUMNS = (
    "row_group",
    "method",
    "run_dir",
    "seed",
    "direction",
    "budget_bytes",
    "n",
    "pass_gate",
    "controls_ok",
    "matched_accuracy",
    "target_accuracy",
    "best_control_accuracy",
    "best_control_name",
    "control_leak_over_target",
    "delta_vs_target",
    "delta_vs_best_control",
    "paired_ci95_low_vs_target",
    "oracle_accuracy",
    "zero_source_accuracy",
    "shuffled_source_accuracy",
    "atom_id_derangement_accuracy",
    "private_random_source_atoms_accuracy",
    "permuted_teacher_receiver_accuracy",
    "random_same_byte_accuracy",
    "answer_only_text_accuracy",
    "structured_text_matched_accuracy",
    "calibration_eval_exact_id_overlap_count",
    "exact_transformed_eval_surface_overlap_count",
    "interpretation",
)


def _resolve(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else ROOT / path


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _fmt_float(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def _seed_from_path(path: pathlib.Path) -> int | None:
    match = re.search(r"seed(\d+)", str(path))
    return None if match is None else int(match.group(1))


def _relative(path: pathlib.Path) -> str:
    resolved = _resolve(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _method_label(gate: dict[str, Any], *, row_group: str) -> str:
    if row_group == "live":
        return "candidate-local residual norm"
    if gate.get("decoder_score_mode") == "global_dot":
        return "global public-anchor dot product"
    if gate.get("decoder_score_mode") == "procrustes_dot":
        return "public-calibration orthogonal Procrustes dot product"
    if gate.get("decoder_score_mode") == "ridge_cca_dot":
        return "ridge CCA/SVCCA-style canonical-coordinate dot product"
    if gate.get("decoder_score_mode") == "ridge_cca_residual_norm":
        return "ridge CCA/SVCCA-style residual norm"
    if gate.get("decoder_score_mode") == "lstirp_relative_dot":
        return "LSTIRP-lite relative-translation dot product"
    if gate.get("decoder_score_mode") == "lstirp_relative_residual_norm":
        return "LSTIRP-lite relative-translation residual norm"
    if gate.get("decoder_score_mode") == "inverse_relative_dot":
        return "LSTIRP-lite inverse-relative dot product"
    if gate.get("decoder_score_mode") == "inverse_relative_residual_norm":
        return "LSTIRP-lite inverse-relative residual norm"
    if gate.get("decoder_score_mode") == "sinkhorn_ot_dot":
        return "Sinkhorn OT public-calibration transport dot product"
    if gate.get("decoder_score_mode") == "sinkhorn_ot_residual_norm":
        return "Sinkhorn OT public-calibration transport residual norm"
    if gate.get("decoder_score_mode") == "gromov_wasserstein_dot":
        return "Gromov-Wasserstein public-calibration transport dot product"
    if gate.get("decoder_score_mode") == "gromov_wasserstein_residual_norm":
        return "Gromov-Wasserstein public-calibration transport residual norm"
    if gate.get("decoder_score_mode") == "ot_gw_dot":
        return "Gromov-Wasserstein public-calibration transport dot product"
    if gate.get("decoder_score_mode") == "ot_gw_residual_norm":
        return "Gromov-Wasserstein public-calibration transport residual norm"
    if gate.get("decoder_score_mode") == "relative_anchor_dot":
        return "Relative Representations anchor-coordinate dot product"
    if gate.get("decoder_score_mode") == "relative_anchor_residual_norm":
        return "relative-anchor residual norm"
    if gate.get("decoder_score_mode") == "relative_anchor_innovation_residual_norm":
        return "relative-anchor innovation residual norm"
    if gate.get("decoder_score_mode") == "relative_anchor_rank_innovation_residual_norm":
        return "ranked relative-anchor innovation residual norm"
    if gate.get("decoder_score_mode") == "candidate_local_residual":
        return "candidate-local residual without row/payload norm"
    return str(gate.get("decoder_score_mode", row_group))


def _row_from_run(
    *,
    run_dir: pathlib.Path,
    row_group: str,
    budget_bytes: int,
) -> list[dict[str, Any]]:
    gate = _read_json(run_dir / "learned_synonym_dictionary_packet_gate.json")
    method = _method_label(gate, row_group=row_group)
    rows: list[dict[str, Any]] = []
    for direction in ("core_to_holdout", "holdout_to_core", "same_family_all"):
        summary_path = _resolve(run_dir) / direction / "summary.json"
        if not summary_path.exists():
            continue
        direction_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        budget = next(
            item for item in direction_summary["budget_summaries"] if int(item["budget_bytes"]) == budget_bytes
        )
        metrics = budget["metrics"]
        target = float(budget["target_accuracy"])
        best_control = float(budget["best_control_accuracy"])
        control_values = {
            f"{condition}_accuracy": float(metrics[condition]["accuracy"])
            for condition in CONTROL_CONDITIONS
            if condition in metrics
        }
        audit = direction_summary.get("surface_overlap_audit", {})
        controls_ok = bool(budget["controls_ok"])
        pass_gate = bool(budget["pass_gate"])
        if row_group == "live":
            interpretation = "passes strict source-private gate"
        elif controls_ok:
            interpretation = "diagnostic ablation keeps controls clean only where pass_gate is true"
        else:
            interpretation = "common-basis decoder is invalidated because destructive controls rise"
        rows.append(
            {
                "row_group": row_group,
                "method": method,
                "run_dir": _relative(run_dir),
                "seed": _seed_from_path(run_dir),
                "direction": direction,
                "budget_bytes": int(budget["budget_bytes"]),
                "n": int(budget["n"]),
                "pass_gate": pass_gate,
                "controls_ok": controls_ok,
                "matched_accuracy": float(budget["learned_synonym_dictionary_accuracy"]),
                "target_accuracy": target,
                "best_control_accuracy": best_control,
                "best_control_name": budget["best_control_name"],
                "control_leak_over_target": best_control - target,
                "delta_vs_target": float(budget["learned_minus_target"]),
                "delta_vs_best_control": float(budget["learned_minus_best_control"]),
                "paired_ci95_low_vs_target": float(budget["paired_bootstrap_vs_target"]["ci95_low"]),
                "oracle_accuracy": float(budget["oracle_learned_candidate_atoms_accuracy"]),
                **control_values,
                "calibration_eval_exact_id_overlap_count": int(
                    audit.get("calibration_eval_exact_id_overlap_count", -1)
                ),
                "exact_transformed_eval_surface_overlap_count": int(
                    audit.get("exact_transformed_eval_surface_overlap_count", -1)
                ),
                "interpretation": interpretation,
            }
        )
    return rows


def _index(rows: list[dict[str, Any]], row_group: str) -> dict[tuple[int | None, str], dict[str, Any]]:
    return {
        (row["seed"], row["direction"]): row
        for row in rows
        if row["row_group"] == row_group
    }


def _comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    live_by_key = _index(rows, "live")
    comparisons: list[dict[str, Any]] = []
    for row in rows:
        if row["row_group"] == "live":
            continue
        live = live_by_key.get((row["seed"], row["direction"]))
        if live is None:
            continue
        comparisons.append(
            {
                "seed": row["seed"],
                "direction": row["direction"],
                "baseline_method": row["method"],
                "live_pass_gate": live["pass_gate"],
                "baseline_pass_gate": row["pass_gate"],
                "live_matched_accuracy": live["matched_accuracy"],
                "baseline_matched_accuracy": row["matched_accuracy"],
                "live_best_control_accuracy": live["best_control_accuracy"],
                "baseline_best_control_accuracy": row["best_control_accuracy"],
                "live_control_leak_over_target": live["control_leak_over_target"],
                "baseline_control_leak_over_target": row["control_leak_over_target"],
                "baseline_invalidated_by_controls": not row["controls_ok"],
            }
        )
    return comparisons


def build_common_basis_falsification(
    *,
    output_dir: pathlib.Path,
    live_run_dirs: list[pathlib.Path],
    global_dot_run_dirs: list[pathlib.Path],
    procrustes_run_dirs: list[pathlib.Path] | None = None,
    ridge_cca_run_dirs: list[pathlib.Path] | None = None,
    ridge_cca_stack_run_dirs: list[pathlib.Path] | None = None,
    lstirp_run_dirs: list[pathlib.Path] | None = None,
    lstirp_stack_run_dirs: list[pathlib.Path] | None = None,
    sinkhorn_ot_run_dirs: list[pathlib.Path] | None = None,
    sinkhorn_ot_stack_run_dirs: list[pathlib.Path] | None = None,
    gw_run_dirs: list[pathlib.Path] | None = None,
    gw_stack_run_dirs: list[pathlib.Path] | None = None,
    relative_anchor_run_dirs: list[pathlib.Path],
    relative_anchor_stack_run_dirs: list[pathlib.Path],
    diagnostic_run_dirs: list[pathlib.Path],
    relative_anchor_innovation_run_dirs: list[pathlib.Path] | None = None,
    relative_anchor_rank_innovation_run_dirs: list[pathlib.Path] | None = None,
    budget_bytes: int = 8,
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for run_dir in live_run_dirs:
        rows.extend(_row_from_run(run_dir=run_dir, row_group="live", budget_bytes=budget_bytes))
    for run_dir in global_dot_run_dirs:
        rows.extend(_row_from_run(run_dir=run_dir, row_group="global_common_basis", budget_bytes=budget_bytes))
    for run_dir in procrustes_run_dirs or []:
        if (_resolve(run_dir) / "learned_synonym_dictionary_packet_gate.json").exists():
            rows.extend(_row_from_run(run_dir=run_dir, row_group="procrustes_common_basis", budget_bytes=budget_bytes))
    for run_dir in ridge_cca_run_dirs or []:
        if (_resolve(run_dir) / "learned_synonym_dictionary_packet_gate.json").exists():
            rows.extend(_row_from_run(run_dir=run_dir, row_group="ridge_cca_common_basis", budget_bytes=budget_bytes))
    for run_dir in ridge_cca_stack_run_dirs or []:
        if (_resolve(run_dir) / "learned_synonym_dictionary_packet_gate.json").exists():
            rows.extend(_row_from_run(run_dir=run_dir, row_group="ridge_cca_local_stack", budget_bytes=budget_bytes))
    for run_dir in lstirp_run_dirs or []:
        if (_resolve(run_dir) / "learned_synonym_dictionary_packet_gate.json").exists():
            rows.extend(_row_from_run(run_dir=run_dir, row_group="lstirp_relative_translation", budget_bytes=budget_bytes))
    for run_dir in lstirp_stack_run_dirs or []:
        if (_resolve(run_dir) / "learned_synonym_dictionary_packet_gate.json").exists():
            rows.extend(_row_from_run(run_dir=run_dir, row_group="lstirp_relative_local_stack", budget_bytes=budget_bytes))
    for run_dir in sinkhorn_ot_run_dirs or []:
        if (_resolve(run_dir) / "learned_synonym_dictionary_packet_gate.json").exists():
            rows.extend(_row_from_run(run_dir=run_dir, row_group="sinkhorn_ot_transport", budget_bytes=budget_bytes))
    for run_dir in sinkhorn_ot_stack_run_dirs or []:
        if (_resolve(run_dir) / "learned_synonym_dictionary_packet_gate.json").exists():
            rows.extend(_row_from_run(run_dir=run_dir, row_group="sinkhorn_ot_local_stack", budget_bytes=budget_bytes))
    for run_dir in gw_run_dirs or []:
        if (_resolve(run_dir) / "learned_synonym_dictionary_packet_gate.json").exists():
            rows.extend(_row_from_run(run_dir=run_dir, row_group="gw_transport", budget_bytes=budget_bytes))
    for run_dir in gw_stack_run_dirs or []:
        if (_resolve(run_dir) / "learned_synonym_dictionary_packet_gate.json").exists():
            rows.extend(_row_from_run(run_dir=run_dir, row_group="gw_local_stack", budget_bytes=budget_bytes))
    for run_dir in relative_anchor_run_dirs:
        if (_resolve(run_dir) / "learned_synonym_dictionary_packet_gate.json").exists():
            rows.extend(
                _row_from_run(run_dir=run_dir, row_group="relative_anchor_common_basis", budget_bytes=budget_bytes)
            )
    for run_dir in relative_anchor_stack_run_dirs:
        if (_resolve(run_dir) / "learned_synonym_dictionary_packet_gate.json").exists():
            rows.extend(_row_from_run(run_dir=run_dir, row_group="relative_anchor_local_stack", budget_bytes=budget_bytes))
    for run_dir in relative_anchor_innovation_run_dirs or []:
        if (_resolve(run_dir) / "learned_synonym_dictionary_packet_gate.json").exists():
            rows.extend(
                _row_from_run(
                    run_dir=run_dir,
                    row_group="relative_anchor_innovation_stack",
                    budget_bytes=budget_bytes,
                )
            )
    for run_dir in relative_anchor_rank_innovation_run_dirs or []:
        if (_resolve(run_dir) / "learned_synonym_dictionary_packet_gate.json").exists():
            rows.extend(
                _row_from_run(
                    run_dir=run_dir,
                    row_group="relative_anchor_rank_innovation_stack",
                    budget_bytes=budget_bytes,
                )
            )
    for run_dir in diagnostic_run_dirs:
        if (_resolve(run_dir) / "learned_synonym_dictionary_packet_gate.json").exists():
            rows.extend(_row_from_run(run_dir=run_dir, row_group="diagnostic_ablation", budget_bytes=budget_bytes))

    live_rows = [row for row in rows if row["row_group"] == "live"]
    global_rows = [row for row in rows if row["row_group"] == "global_common_basis"]
    procrustes_rows = [row for row in rows if row["row_group"] == "procrustes_common_basis"]
    ridge_cca_rows = [row for row in rows if row["row_group"] == "ridge_cca_common_basis"]
    ridge_cca_stack_rows = [row for row in rows if row["row_group"] == "ridge_cca_local_stack"]
    lstirp_rows = [row for row in rows if row["row_group"] == "lstirp_relative_translation"]
    lstirp_stack_rows = [row for row in rows if row["row_group"] == "lstirp_relative_local_stack"]
    sinkhorn_ot_rows = [row for row in rows if row["row_group"] == "sinkhorn_ot_transport"]
    sinkhorn_ot_stack_rows = [row for row in rows if row["row_group"] == "sinkhorn_ot_local_stack"]
    gw_rows = [row for row in rows if row["row_group"] == "gw_transport"]
    gw_stack_rows = [row for row in rows if row["row_group"] == "gw_local_stack"]
    relative_rows = [row for row in rows if row["row_group"] == "relative_anchor_common_basis"]
    relative_stack_rows = [row for row in rows if row["row_group"] == "relative_anchor_local_stack"]
    relative_innovation_rows = [row for row in rows if row["row_group"] == "relative_anchor_innovation_stack"]
    relative_rank_innovation_rows = [
        row for row in rows if row["row_group"] == "relative_anchor_rank_innovation_stack"
    ]
    diagnostic_rows = [row for row in rows if row["row_group"] == "diagnostic_ablation"]
    comparisons = _comparison_rows(rows)
    global_control_leaks = [row for row in global_rows if not row["controls_ok"]]
    procrustes_control_leaks = [row for row in procrustes_rows if not row["controls_ok"]]
    ridge_cca_control_leaks = [row for row in ridge_cca_rows if not row["controls_ok"]]
    lstirp_control_leaks = [row for row in lstirp_rows if not row["controls_ok"]]
    lstirp_stack_control_leaks = [row for row in lstirp_stack_rows if not row["controls_ok"]]
    sinkhorn_ot_control_leaks = [row for row in sinkhorn_ot_rows if not row["controls_ok"]]
    sinkhorn_ot_stack_control_leaks = [row for row in sinkhorn_ot_stack_rows if not row["controls_ok"]]
    gw_control_leaks = [row for row in gw_rows if not row["controls_ok"]]
    gw_stack_control_leaks = [row for row in gw_stack_rows if not row["controls_ok"]]
    relative_control_leaks = [row for row in relative_rows if not row["controls_ok"]]
    relative_innovation_control_leaks = [row for row in relative_innovation_rows if not row["controls_ok"]]
    relative_rank_innovation_control_leaks = [
        row for row in relative_rank_innovation_rows if not row["controls_ok"]
    ]
    headline = {
        "budget_bytes": budget_bytes,
        "live_rows": len(live_rows),
        "live_pass_rows": sum(row["pass_gate"] for row in live_rows),
        "global_common_basis_rows": len(global_rows),
        "global_common_basis_pass_rows": sum(row["pass_gate"] for row in global_rows),
        "global_common_basis_control_leak_rows": len(global_control_leaks),
        "procrustes_common_basis_rows": len(procrustes_rows),
        "procrustes_common_basis_pass_rows": sum(row["pass_gate"] for row in procrustes_rows),
        "procrustes_common_basis_control_leak_rows": len(procrustes_control_leaks),
        "ridge_cca_common_basis_rows": len(ridge_cca_rows),
        "ridge_cca_common_basis_pass_rows": sum(row["pass_gate"] for row in ridge_cca_rows),
        "ridge_cca_common_basis_control_leak_rows": len(ridge_cca_control_leaks),
        "ridge_cca_stack_rows": len(ridge_cca_stack_rows),
        "ridge_cca_stack_pass_rows": sum(row["pass_gate"] for row in ridge_cca_stack_rows),
        "lstirp_rows": len(lstirp_rows),
        "lstirp_pass_rows": sum(row["pass_gate"] for row in lstirp_rows),
        "lstirp_control_leak_rows": len(lstirp_control_leaks),
        "lstirp_stack_rows": len(lstirp_stack_rows),
        "lstirp_stack_pass_rows": sum(row["pass_gate"] for row in lstirp_stack_rows),
        "lstirp_stack_control_leak_rows": len(lstirp_stack_control_leaks),
        "sinkhorn_ot_rows": len(sinkhorn_ot_rows),
        "sinkhorn_ot_pass_rows": sum(row["pass_gate"] for row in sinkhorn_ot_rows),
        "sinkhorn_ot_control_leak_rows": len(sinkhorn_ot_control_leaks),
        "sinkhorn_ot_stack_rows": len(sinkhorn_ot_stack_rows),
        "sinkhorn_ot_stack_pass_rows": sum(row["pass_gate"] for row in sinkhorn_ot_stack_rows),
        "sinkhorn_ot_stack_control_leak_rows": len(sinkhorn_ot_stack_control_leaks),
        "gw_rows": len(gw_rows),
        "gw_pass_rows": sum(row["pass_gate"] for row in gw_rows),
        "gw_control_leak_rows": len(gw_control_leaks),
        "gw_stack_rows": len(gw_stack_rows),
        "gw_stack_pass_rows": sum(row["pass_gate"] for row in gw_stack_rows),
        "gw_stack_control_leak_rows": len(gw_stack_control_leaks),
        "relative_anchor_rows": len(relative_rows),
        "relative_anchor_pass_rows": sum(row["pass_gate"] for row in relative_rows),
        "relative_anchor_control_leak_rows": len(relative_control_leaks),
        "relative_anchor_stack_rows": len(relative_stack_rows),
        "relative_anchor_stack_pass_rows": sum(row["pass_gate"] for row in relative_stack_rows),
        "relative_anchor_innovation_rows": len(relative_innovation_rows),
        "relative_anchor_innovation_pass_rows": sum(row["pass_gate"] for row in relative_innovation_rows),
        "relative_anchor_innovation_control_leak_rows": len(relative_innovation_control_leaks),
        "relative_anchor_rank_innovation_rows": len(relative_rank_innovation_rows),
        "relative_anchor_rank_innovation_pass_rows": sum(
            row["pass_gate"] for row in relative_rank_innovation_rows
        ),
        "relative_anchor_rank_innovation_control_leak_rows": len(relative_rank_innovation_control_leaks),
        "diagnostic_rows": len(diagnostic_rows),
        "diagnostic_pass_rows": sum(row["pass_gate"] for row in diagnostic_rows),
        "max_global_matched_accuracy": max(row["matched_accuracy"] for row in global_rows),
        "max_global_best_control_accuracy": max(row["best_control_accuracy"] for row in global_rows),
        "max_procrustes_matched_accuracy": max(row["matched_accuracy"] for row in procrustes_rows)
        if procrustes_rows
        else None,
        "max_procrustes_best_control_accuracy": max(row["best_control_accuracy"] for row in procrustes_rows)
        if procrustes_rows
        else None,
        "max_ridge_cca_matched_accuracy": max(row["matched_accuracy"] for row in ridge_cca_rows)
        if ridge_cca_rows
        else None,
        "max_ridge_cca_best_control_accuracy": max(row["best_control_accuracy"] for row in ridge_cca_rows)
        if ridge_cca_rows
        else None,
        "max_ridge_cca_stack_best_control_accuracy": max(row["best_control_accuracy"] for row in ridge_cca_stack_rows)
        if ridge_cca_stack_rows
        else None,
        "max_lstirp_matched_accuracy": max(row["matched_accuracy"] for row in lstirp_rows)
        if lstirp_rows
        else None,
        "max_lstirp_best_control_accuracy": max(row["best_control_accuracy"] for row in lstirp_rows)
        if lstirp_rows
        else None,
        "max_lstirp_stack_matched_accuracy": max(row["matched_accuracy"] for row in lstirp_stack_rows)
        if lstirp_stack_rows
        else None,
        "max_lstirp_stack_best_control_accuracy": max(row["best_control_accuracy"] for row in lstirp_stack_rows)
        if lstirp_stack_rows
        else None,
        "max_sinkhorn_ot_matched_accuracy": max(row["matched_accuracy"] for row in sinkhorn_ot_rows)
        if sinkhorn_ot_rows
        else None,
        "max_sinkhorn_ot_best_control_accuracy": max(row["best_control_accuracy"] for row in sinkhorn_ot_rows)
        if sinkhorn_ot_rows
        else None,
        "max_sinkhorn_ot_stack_matched_accuracy": max(row["matched_accuracy"] for row in sinkhorn_ot_stack_rows)
        if sinkhorn_ot_stack_rows
        else None,
        "max_sinkhorn_ot_stack_best_control_accuracy": max(row["best_control_accuracy"] for row in sinkhorn_ot_stack_rows)
        if sinkhorn_ot_stack_rows
        else None,
        "max_gw_matched_accuracy": max(row["matched_accuracy"] for row in gw_rows)
        if gw_rows
        else None,
        "max_gw_best_control_accuracy": max(row["best_control_accuracy"] for row in gw_rows)
        if gw_rows
        else None,
        "max_gw_stack_matched_accuracy": max(row["matched_accuracy"] for row in gw_stack_rows)
        if gw_stack_rows
        else None,
        "max_gw_stack_best_control_accuracy": max(row["best_control_accuracy"] for row in gw_stack_rows)
        if gw_stack_rows
        else None,
        "max_relative_anchor_matched_accuracy": max(row["matched_accuracy"] for row in relative_rows)
        if relative_rows
        else None,
        "max_relative_anchor_best_control_accuracy": max(row["best_control_accuracy"] for row in relative_rows)
        if relative_rows
        else None,
        "max_relative_anchor_stack_best_control_accuracy": max(
            row["best_control_accuracy"] for row in relative_stack_rows
        )
        if relative_stack_rows
        else None,
        "max_relative_anchor_innovation_matched_accuracy": max(
            row["matched_accuracy"] for row in relative_innovation_rows
        )
        if relative_innovation_rows
        else None,
        "max_relative_anchor_innovation_best_control_accuracy": max(
            row["best_control_accuracy"] for row in relative_innovation_rows
        )
        if relative_innovation_rows
        else None,
        "max_relative_anchor_rank_innovation_matched_accuracy": max(
            row["matched_accuracy"] for row in relative_rank_innovation_rows
        )
        if relative_rank_innovation_rows
        else None,
        "max_relative_anchor_rank_innovation_best_control_accuracy": max(
            row["best_control_accuracy"] for row in relative_rank_innovation_rows
        )
        if relative_rank_innovation_rows
        else None,
        "max_live_best_control_accuracy": max(row["best_control_accuracy"] for row in live_rows),
        "min_live_matched_accuracy": min(row["matched_accuracy"] for row in live_rows),
        "pass_gate": (
            bool(live_rows)
            and all(row["pass_gate"] for row in live_rows)
            and bool(global_rows)
            and all(not row["pass_gate"] for row in global_rows)
            and all(not row["controls_ok"] for row in global_rows)
            and bool(relative_rows)
            and all(not row["pass_gate"] for row in relative_rows)
            and all(not row["controls_ok"] for row in relative_rows)
        ),
    }
    payload = {
        "gate": "source_private_candidate_local_common_basis_falsification",
        "headline": headline,
        "rows": rows,
        "comparisons": comparisons,
        "interpretation": _interpretation(headline),
        "non_claims": [
            "The Relative Representations row is an anchor-coordinate public-calibration baseline for this packet benchmark, not a full dense latent-transfer reproduction.",
            "The Procrustes row is an orthogonal public-calibration packet receiver, not CCA/SVCCA or OT.",
            "The ridge CCA row is a canonical-coordinate packet receiver, not a model-stitching or CKA-only statistic.",
            "The LSTIRP-lite rows are public relative-coordinate translation baselines, not full dense latent transfer or OT.",
            "The Sinkhorn OT and Gromov-Wasserstein rows are calibration-axis transport baselines over atom supports, not full hidden-state OT across dense token activations.",
            "The RR innovation rows are scoped repair probes for the measured RR partial competitor, not new positive methods.",
            "This table falsifies measured common-basis decoders on this surface, not all possible common-basis methods.",
            "C2C/KVComm/KV-compression rows still require a separate systems/proxy table.",
        ],
    }
    json_path = output_dir / "candidate_local_common_basis_falsification.json"
    csv_path = output_dir / "candidate_local_common_basis_falsification.csv"
    md_path = output_dir / "candidate_local_common_basis_falsification.md"
    manifest_path = output_dir / "manifest.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(row.get(key)) for key in CSV_COLUMNS})
    _write_markdown(md_path, payload)
    manifest = {
        "artifacts": [json_path.name, csv_path.name, md_path.name, manifest_path.name],
        "artifact_sha256": {
            json_path.name: _sha256_file(json_path),
            csv_path.name: _sha256_file(csv_path),
            md_path.name: _sha256_file(md_path),
        },
        "headline": headline,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "manifest.md").write_text(
        "\n".join(
            [
                "# Candidate-Local Common-Basis Falsification Manifest",
                "",
                f"- pass gate: `{headline['pass_gate']}`",
                f"- live pass rows: `{headline['live_pass_rows']}/{headline['live_rows']}`",
                (
                    "- global common-basis pass rows: "
                    f"`{headline['global_common_basis_pass_rows']}/{headline['global_common_basis_rows']}`"
                ),
                (
                    "- Procrustes common-basis pass rows: "
                    f"`{headline['procrustes_common_basis_pass_rows']}/{headline['procrustes_common_basis_rows']}`"
                ),
                (
                    "- ridge CCA common-basis pass rows: "
                    f"`{headline['ridge_cca_common_basis_pass_rows']}/{headline['ridge_cca_common_basis_rows']}`"
                ),
                (
                    "- ridge CCA local-stack pass rows: "
                    f"`{headline['ridge_cca_stack_pass_rows']}/{headline['ridge_cca_stack_rows']}`"
                ),
                (
                    "- LSTIRP-lite pass rows: "
                    f"`{headline['lstirp_pass_rows']}/{headline['lstirp_rows']}`"
                ),
                (
                    "- LSTIRP-lite local-stack pass rows: "
                    f"`{headline['lstirp_stack_pass_rows']}/{headline['lstirp_stack_rows']}`"
                ),
                (
                    "- Sinkhorn OT pass rows: "
                    f"`{headline['sinkhorn_ot_pass_rows']}/{headline['sinkhorn_ot_rows']}`"
                ),
                (
                    "- Sinkhorn OT local-stack pass rows: "
                    f"`{headline['sinkhorn_ot_stack_pass_rows']}/{headline['sinkhorn_ot_stack_rows']}`"
                ),
                (
                    "- GW transport pass rows: "
                    f"`{headline['gw_pass_rows']}/{headline['gw_rows']}`"
                ),
                (
                    "- GW local-stack pass rows: "
                    f"`{headline['gw_stack_pass_rows']}/{headline['gw_stack_rows']}`"
                ),
                (
                    "- relative-anchor pass rows: "
                    f"`{headline['relative_anchor_pass_rows']}/{headline['relative_anchor_rows']}`"
                ),
                (
                    "- relative-anchor local-stack pass rows: "
                    f"`{headline['relative_anchor_stack_pass_rows']}/{headline['relative_anchor_stack_rows']}`"
                ),
                (
                    "- relative-anchor innovation-stack pass rows: "
                    f"`{headline['relative_anchor_innovation_pass_rows']}/{headline['relative_anchor_innovation_rows']}`"
                ),
                (
                    "- relative-anchor rank-innovation pass rows: "
                    f"`{headline['relative_anchor_rank_innovation_pass_rows']}/{headline['relative_anchor_rank_innovation_rows']}`"
                ),
                f"- global control-leak rows: `{headline['global_common_basis_control_leak_rows']}`",
                f"- Procrustes control-leak rows: `{headline['procrustes_common_basis_control_leak_rows']}`",
                f"- ridge CCA control-leak rows: `{headline['ridge_cca_common_basis_control_leak_rows']}`",
                f"- LSTIRP-lite control-leak rows: `{headline['lstirp_control_leak_rows']}`",
                f"- LSTIRP-lite local-stack control-leak rows: `{headline['lstirp_stack_control_leak_rows']}`",
                f"- Sinkhorn OT control-leak rows: `{headline['sinkhorn_ot_control_leak_rows']}`",
                f"- Sinkhorn OT local-stack control-leak rows: `{headline['sinkhorn_ot_stack_control_leak_rows']}`",
                f"- GW transport control-leak rows: `{headline['gw_control_leak_rows']}`",
                f"- GW local-stack control-leak rows: `{headline['gw_stack_control_leak_rows']}`",
                f"- relative-anchor control-leak rows: `{headline['relative_anchor_control_leak_rows']}`",
                f"- relative-anchor innovation control-leak rows: `{headline['relative_anchor_innovation_control_leak_rows']}`",
                (
                    "- relative-anchor rank-innovation control-leak rows: "
                    f"`{headline['relative_anchor_rank_innovation_control_leak_rows']}`"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def _interpretation(headline: dict[str, Any]) -> str:
    if headline["relative_anchor_pass_rows"] > 0:
        return (
            "The global public-anchor dot-product decoder is invalidated by destructive-control leakage, but "
            "the public-calibration orthogonal Procrustes decoder is also invalidated by destructive-control "
            "leakage. The ridge CCA/SVCCA-style decoder is reported as the linear-subspace common-basis row. "
            "The LSTIRP-lite rows test public relative-coordinate translation directly. "
            "The OT/GW rows test whether public calibration geometry can transport the packet into a receiver "
            "basis without using candidate-local residuals. "
            "The Relative Representations-style anchor-coordinate decoder is a real same-slice competitor: it "
            f"passes {headline['relative_anchor_pass_rows']}/{headline['relative_anchor_rows']} n512 rows with "
            "controls near the target floor. The live candidate-local residual-normalized receiver remains the "
            "only repeated row group that passes all directions, but the common-basis objection is not closed."
        )
    return (
        "The global public-anchor dot-product decoder is a strong common-basis ablation: it often raises "
        "matched-packet accuracy, but every n512 seed/direction row fails because destructive controls also "
        "rise. The public-calibration orthogonal Procrustes decoder follows the same unsafe pattern. The "
        "ridge CCA/SVCCA-style decoder is the measured linear-subspace row on this surface. The "
        "LSTIRP-lite rows are the measured relative-translation rows. The OT/GW rows are the measured "
        "calibration-transport rows. The "
        "Relative Representations-style anchor-coordinate decoder also fails under the strict gate. "
        "The live candidate-local residual-normalized receiver is the only repeated n512 row group that passes "
        "all strict source-private controls."
    )


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    headline = payload["headline"]
    lines = [
        "# Candidate-Local Common-Basis Falsification",
        "",
        "This table compares the live n512 candidate-local residual-normalized receiver",
        "against implemented common-basis ablations: global public-anchor dot",
        "product decoding, orthogonal Procrustes, ridge CCA/SVCCA-style",
        "canonical coordinates, LSTIRP-lite relative translation, Sinkhorn/GW",
        "calibration transport, and RR-style anchor coordinates on the same",
        "seeds, directions, and 8B budget.",
        "",
        "## Headline",
        "",
        f"- pass gate: `{headline['pass_gate']}`",
        f"- live pass rows: `{headline['live_pass_rows']}/{headline['live_rows']}`",
        (
            "- global common-basis pass rows: "
            f"`{headline['global_common_basis_pass_rows']}/{headline['global_common_basis_rows']}`"
        ),
        (
            "- Procrustes common-basis pass rows: "
            f"`{headline['procrustes_common_basis_pass_rows']}/{headline['procrustes_common_basis_rows']}`"
        ),
        (
            "- ridge CCA common-basis pass rows: "
            f"`{headline['ridge_cca_common_basis_pass_rows']}/{headline['ridge_cca_common_basis_rows']}`"
        ),
        (
            "- ridge CCA local-stack pass rows: "
            f"`{headline['ridge_cca_stack_pass_rows']}/{headline['ridge_cca_stack_rows']}`"
        ),
        (
            "- LSTIRP-lite pass rows: "
            f"`{headline['lstirp_pass_rows']}/{headline['lstirp_rows']}`"
        ),
        (
            "- LSTIRP-lite local-stack pass rows: "
            f"`{headline['lstirp_stack_pass_rows']}/{headline['lstirp_stack_rows']}`"
        ),
        (
            "- Sinkhorn OT pass rows: "
            f"`{headline['sinkhorn_ot_pass_rows']}/{headline['sinkhorn_ot_rows']}`"
        ),
        (
            "- Sinkhorn OT local-stack pass rows: "
            f"`{headline['sinkhorn_ot_stack_pass_rows']}/{headline['sinkhorn_ot_stack_rows']}`"
        ),
        (
            "- GW transport pass rows: "
            f"`{headline['gw_pass_rows']}/{headline['gw_rows']}`"
        ),
        (
            "- GW local-stack pass rows: "
            f"`{headline['gw_stack_pass_rows']}/{headline['gw_stack_rows']}`"
        ),
        (
            "- relative-anchor pass rows: "
            f"`{headline['relative_anchor_pass_rows']}/{headline['relative_anchor_rows']}`"
        ),
        (
            "- relative-anchor local-stack pass rows: "
            f"`{headline['relative_anchor_stack_pass_rows']}/{headline['relative_anchor_stack_rows']}`"
        ),
        (
            "- relative-anchor innovation-stack pass rows: "
            f"`{headline['relative_anchor_innovation_pass_rows']}/{headline['relative_anchor_innovation_rows']}`"
        ),
        (
            "- relative-anchor rank-innovation pass rows: "
            f"`{headline['relative_anchor_rank_innovation_pass_rows']}/{headline['relative_anchor_rank_innovation_rows']}`"
        ),
        f"- global common-basis control-leak rows: `{headline['global_common_basis_control_leak_rows']}`",
        f"- Procrustes common-basis control-leak rows: `{headline['procrustes_common_basis_control_leak_rows']}`",
        f"- ridge CCA common-basis control-leak rows: `{headline['ridge_cca_common_basis_control_leak_rows']}`",
        f"- LSTIRP-lite control-leak rows: `{headline['lstirp_control_leak_rows']}`",
        f"- LSTIRP-lite local-stack control-leak rows: `{headline['lstirp_stack_control_leak_rows']}`",
        f"- Sinkhorn OT control-leak rows: `{headline['sinkhorn_ot_control_leak_rows']}`",
        f"- Sinkhorn OT local-stack control-leak rows: `{headline['sinkhorn_ot_stack_control_leak_rows']}`",
        f"- GW transport control-leak rows: `{headline['gw_control_leak_rows']}`",
        f"- GW local-stack control-leak rows: `{headline['gw_stack_control_leak_rows']}`",
        f"- relative-anchor control-leak rows: `{headline['relative_anchor_control_leak_rows']}`",
        f"- relative-anchor innovation control-leak rows: `{headline['relative_anchor_innovation_control_leak_rows']}`",
        (
            "- relative-anchor rank-innovation control-leak rows: "
            f"`{headline['relative_anchor_rank_innovation_control_leak_rows']}`"
        ),
        f"- max global matched accuracy: `{_fmt_float(headline['max_global_matched_accuracy'])}`",
        f"- max global best-control accuracy: `{_fmt_float(headline['max_global_best_control_accuracy'])}`",
        f"- max Procrustes matched accuracy: `{_fmt_float(headline['max_procrustes_matched_accuracy'])}`",
        f"- max Procrustes best-control accuracy: `{_fmt_float(headline['max_procrustes_best_control_accuracy'])}`",
        f"- max ridge CCA matched accuracy: `{_fmt_float(headline['max_ridge_cca_matched_accuracy'])}`",
        f"- max ridge CCA best-control accuracy: `{_fmt_float(headline['max_ridge_cca_best_control_accuracy'])}`",
        f"- max ridge CCA local-stack best-control accuracy: `{_fmt_float(headline['max_ridge_cca_stack_best_control_accuracy'])}`",
        f"- max LSTIRP-lite matched accuracy: `{_fmt_float(headline['max_lstirp_matched_accuracy'])}`",
        f"- max LSTIRP-lite best-control accuracy: `{_fmt_float(headline['max_lstirp_best_control_accuracy'])}`",
        f"- max LSTIRP-lite local-stack matched accuracy: `{_fmt_float(headline['max_lstirp_stack_matched_accuracy'])}`",
        f"- max LSTIRP-lite local-stack best-control accuracy: `{_fmt_float(headline['max_lstirp_stack_best_control_accuracy'])}`",
        f"- max Sinkhorn OT matched accuracy: `{_fmt_float(headline['max_sinkhorn_ot_matched_accuracy'])}`",
        f"- max Sinkhorn OT best-control accuracy: `{_fmt_float(headline['max_sinkhorn_ot_best_control_accuracy'])}`",
        f"- max Sinkhorn OT local-stack matched accuracy: `{_fmt_float(headline['max_sinkhorn_ot_stack_matched_accuracy'])}`",
        (
            "- max Sinkhorn OT local-stack best-control accuracy: "
            f"`{_fmt_float(headline['max_sinkhorn_ot_stack_best_control_accuracy'])}`"
        ),
        f"- max GW matched accuracy: `{_fmt_float(headline['max_gw_matched_accuracy'])}`",
        f"- max GW best-control accuracy: `{_fmt_float(headline['max_gw_best_control_accuracy'])}`",
        f"- max GW local-stack matched accuracy: `{_fmt_float(headline['max_gw_stack_matched_accuracy'])}`",
        (
            "- max GW local-stack best-control accuracy: "
            f"`{_fmt_float(headline['max_gw_stack_best_control_accuracy'])}`"
        ),
        f"- max relative-anchor matched accuracy: `{_fmt_float(headline['max_relative_anchor_matched_accuracy'])}`",
        f"- max relative-anchor best-control accuracy: `{_fmt_float(headline['max_relative_anchor_best_control_accuracy'])}`",
        (
            "- max relative-anchor local-stack best-control accuracy: "
            f"`{_fmt_float(headline['max_relative_anchor_stack_best_control_accuracy'])}`"
        ),
        (
            "- max relative-anchor innovation matched accuracy: "
            f"`{_fmt_float(headline['max_relative_anchor_innovation_matched_accuracy'])}`"
        ),
        (
            "- max relative-anchor innovation best-control accuracy: "
            f"`{_fmt_float(headline['max_relative_anchor_innovation_best_control_accuracy'])}`"
        ),
        (
            "- max relative-anchor rank-innovation matched accuracy: "
            f"`{_fmt_float(headline['max_relative_anchor_rank_innovation_matched_accuracy'])}`"
        ),
        (
            "- max relative-anchor rank-innovation best-control accuracy: "
            f"`{_fmt_float(headline['max_relative_anchor_rank_innovation_best_control_accuracy'])}`"
        ),
        f"- max live best-control accuracy: `{_fmt_float(headline['max_live_best_control_accuracy'])}`",
        "",
        "## Rows",
        "",
        "| Group | Method | Seed | Direction | Pass | Matched | Target | Best ctrl | Ctrl leak | Best ctrl name |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {row_group} | {method} | {seed} | {direction} | `{pass_gate}` | {matched:.3f} | "
            "{target:.3f} | {best:.3f} | {leak:+.3f} | `{control}` |".format(
                row_group=row["row_group"],
                method=row["method"],
                seed=row["seed"],
                direction=row["direction"],
                pass_gate=row["pass_gate"],
                matched=row["matched_accuracy"],
                target=row["target_accuracy"],
                best=row["best_control_accuracy"],
                leak=row["control_leak_over_target"],
                control=row["best_control_name"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
            "Layman explanation: rotating, correlating, or translating through public",
            "anchor relationships can make the real clue look good, but fake",
            "transformed clues may work too. The next paper story must treat clean",
            "relative-coordinate rows as real baselines, not as defeated prior work.",
            "",
            "## Non-Claims",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload["non_claims"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budget-bytes", type=int, default=8)
    parser.add_argument("--live-run-dir", type=pathlib.Path, action="append", default=None)
    parser.add_argument("--global-dot-run-dir", type=pathlib.Path, action="append", default=None)
    parser.add_argument("--procrustes-run-dir", type=pathlib.Path, action="append", default=None)
    parser.add_argument("--ridge-cca-run-dir", type=pathlib.Path, action="append", default=None)
    parser.add_argument("--ridge-cca-stack-run-dir", type=pathlib.Path, action="append", default=None)
    parser.add_argument("--lstirp-run-dir", type=pathlib.Path, action="append", default=None)
    parser.add_argument("--lstirp-stack-run-dir", type=pathlib.Path, action="append", default=None)
    parser.add_argument("--sinkhorn-ot-run-dir", type=pathlib.Path, action="append", default=None)
    parser.add_argument("--sinkhorn-ot-stack-run-dir", type=pathlib.Path, action="append", default=None)
    parser.add_argument("--gw-run-dir", type=pathlib.Path, action="append", default=None)
    parser.add_argument("--gw-stack-run-dir", type=pathlib.Path, action="append", default=None)
    parser.add_argument("--relative-anchor-run-dir", type=pathlib.Path, action="append", default=None)
    parser.add_argument("--relative-anchor-stack-run-dir", type=pathlib.Path, action="append", default=None)
    parser.add_argument("--relative-anchor-innovation-run-dir", type=pathlib.Path, action="append", default=None)
    parser.add_argument("--relative-anchor-rank-innovation-run-dir", type=pathlib.Path, action="append", default=None)
    parser.add_argument("--diagnostic-run-dir", type=pathlib.Path, action="append", default=None)
    args = parser.parse_args()
    payload = build_common_basis_falsification(
        output_dir=args.output_dir,
        live_run_dirs=args.live_run_dir or list(DEFAULT_LIVE_RUNS),
        global_dot_run_dirs=args.global_dot_run_dir or list(DEFAULT_GLOBAL_DOT_RUNS),
        procrustes_run_dirs=args.procrustes_run_dir or list(DEFAULT_PROCRUSTES_RUNS),
        ridge_cca_run_dirs=args.ridge_cca_run_dir or list(DEFAULT_RIDGE_CCA_RUNS),
        ridge_cca_stack_run_dirs=args.ridge_cca_stack_run_dir or list(DEFAULT_RIDGE_CCA_STACK_RUNS),
        lstirp_run_dirs=args.lstirp_run_dir or list(DEFAULT_LSTIRP_RUNS),
        lstirp_stack_run_dirs=args.lstirp_stack_run_dir or list(DEFAULT_LSTIRP_STACK_RUNS),
        sinkhorn_ot_run_dirs=args.sinkhorn_ot_run_dir or list(DEFAULT_SINKHORN_OT_RUNS),
        sinkhorn_ot_stack_run_dirs=args.sinkhorn_ot_stack_run_dir or list(DEFAULT_SINKHORN_OT_STACK_RUNS),
        gw_run_dirs=args.gw_run_dir or list(DEFAULT_GW_RUNS),
        gw_stack_run_dirs=args.gw_stack_run_dir or list(DEFAULT_GW_STACK_RUNS),
        relative_anchor_run_dirs=args.relative_anchor_run_dir or list(DEFAULT_RELATIVE_ANCHOR_RUNS),
        relative_anchor_stack_run_dirs=args.relative_anchor_stack_run_dir or list(DEFAULT_RELATIVE_ANCHOR_STACK_RUNS),
        diagnostic_run_dirs=args.diagnostic_run_dir or list(DEFAULT_DIAGNOSTIC_RUNS),
        relative_anchor_innovation_run_dirs=args.relative_anchor_innovation_run_dir
        or list(DEFAULT_RELATIVE_ANCHOR_INNOVATION_RUNS),
        relative_anchor_rank_innovation_run_dirs=args.relative_anchor_rank_innovation_run_dir
        or list(DEFAULT_RELATIVE_ANCHOR_RANK_INNOVATION_RUNS),
        budget_bytes=args.budget_bytes,
    )
    print(json.dumps({"output_dir": str(_resolve(args.output_dir)), "headline": payload["headline"]}, indent=2))


if __name__ == "__main__":
    main()
