from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_source_private_learned_synonym_dictionary_packet_gate import STRICT_SOURCE_DESTROYING_CONTROLS  # noqa: E402

DEFAULT_OUTPUT = pathlib.Path("results/source_private_candidate_local_threshold_frontier_20260501")

DEFAULT_METHOD_ROOTS = {
    "live_candidate_local_residual_norm": [
        pathlib.Path("results/source_private_candidate_local_residual_receiver_20260430_seed47_n512_minilm_teacher_norm_dec048_evaldisjoint"),
        pathlib.Path("results/source_private_candidate_local_residual_receiver_20260430_seed53_n512_minilm_teacher_norm_dec048_evaldisjoint"),
        pathlib.Path("results/source_private_candidate_local_residual_receiver_20260430_seed59_n512_minilm_teacher_norm_dec048_evaldisjoint"),
    ],
    "rr_anchor_coordinate_dot": [
        pathlib.Path("results/source_private_candidate_local_residual_ablation_20260430_seed47_n512_relative_anchor_dot_evaldisjoint"),
        pathlib.Path("results/source_private_candidate_local_residual_ablation_20260430_seed53_n512_relative_anchor_dot_evaldisjoint"),
        pathlib.Path("results/source_private_candidate_local_residual_ablation_20260430_seed59_n512_relative_anchor_dot_evaldisjoint"),
    ],
    "public_random_rotation_sign": [
        pathlib.Path("results/source_private_candidate_local_random_rotation_sign_20260501_seed47_n128_evaldisjoint_tau0"),
    ],
}

THRESHOLDS = tuple(sorted({0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.48, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90}))

CSV_COLUMNS = (
    "method_id",
    "threshold",
    "rows",
    "clean_rows",
    "all_rows_clean",
    "matched_accuracy_min",
    "matched_accuracy_mean",
    "target_accuracy_max",
    "best_control_accuracy_max",
    "delta_vs_best_control_min",
    "lift_vs_target_min",
    "control_over_target_max",
)

ROW_COLUMNS = (
    "method_id",
    "threshold",
    "root",
    "direction",
    "n",
    "matched_accuracy",
    "target_accuracy",
    "best_control_name",
    "best_control_accuracy",
    "controls_ok",
    "clean",
    "lift_vs_target",
    "delta_vs_best_control",
)


def _resolve(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else ROOT / path


def _relative(path: pathlib.Path) -> str:
    resolved = _resolve(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in _resolve(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _predicted_index_at_threshold(row: dict[str, Any], threshold: float) -> int:
    prior_index = int(row.get("prior_index", 0))
    scores = row.get("metadata", {}).get("scores")
    if not scores:
        return prior_index
    values = [float(score) for score in scores]
    best_score = max(values)
    if best_score < threshold:
        return prior_index
    tied = [idx for idx, score in enumerate(values) if abs(score - best_score) <= 1e-8]
    return prior_index if prior_index in tied else tied[0]


def _accuracy_at_threshold(rows: list[dict[str, Any]], threshold: float) -> float:
    if not rows:
        return 0.0
    correct = 0
    for row in rows:
        if _predicted_index_at_threshold(row, threshold) == int(row["answer_index"]):
            correct += 1
    return correct / len(rows)


def _target_accuracy(rows: list[dict[str, Any]]) -> float:
    target = [row for row in rows if row["condition"] == "target_only"]
    if not target:
        return 0.0
    return sum(bool(row["correct"]) for row in target) / len(target)


def _threshold_direction_row(
    *,
    method_id: str,
    root: pathlib.Path,
    direction: str,
    threshold: float,
) -> dict[str, Any]:
    rows = _read_jsonl(root / direction / "predictions_budget8.jsonl")
    by_condition = {}
    for row in rows:
        by_condition.setdefault(row["condition"], []).append(row)
    matched = _accuracy_at_threshold(by_condition.get("learned_synonym_dictionary_packet", []), threshold)
    target = _target_accuracy(rows)
    control_scores = {
        condition: _accuracy_at_threshold(by_condition.get(condition, []), threshold)
        for condition in STRICT_SOURCE_DESTROYING_CONTROLS
        if condition in by_condition
    }
    best_control_name = max(control_scores, key=control_scores.get)
    best_control = control_scores[best_control_name]
    controls_ok = all(value <= target + 0.03 for value in control_scores.values())
    clean = matched >= target + 0.15 and matched >= best_control + 0.10 and controls_ok
    return {
        "method_id": method_id,
        "threshold": threshold,
        "root": _relative(root),
        "direction": direction,
        "n": len(by_condition.get("target_only", [])),
        "matched_accuracy": matched,
        "target_accuracy": target,
        "best_control_name": best_control_name,
        "best_control_accuracy": best_control,
        "controls_ok": controls_ok,
        "clean": clean,
        "lift_vs_target": matched - target,
        "delta_vs_best_control": matched - best_control,
        "control_over_target": best_control - target,
    }


def _aggregate_threshold_rows(method_id: str, threshold: float, rows: list[dict[str, Any]]) -> dict[str, Any]:
    matched = [float(row["matched_accuracy"]) for row in rows]
    targets = [float(row["target_accuracy"]) for row in rows]
    controls = [float(row["best_control_accuracy"]) for row in rows]
    deltas = [float(row["delta_vs_best_control"]) for row in rows]
    lifts = [float(row["lift_vs_target"]) for row in rows]
    control_over_target = [float(row["control_over_target"]) for row in rows]
    clean_rows = sum(bool(row["clean"]) for row in rows)
    return {
        "method_id": method_id,
        "threshold": threshold,
        "rows": len(rows),
        "clean_rows": clean_rows,
        "all_rows_clean": clean_rows == len(rows) and bool(rows),
        "matched_accuracy_min": min(matched) if matched else None,
        "matched_accuracy_mean": sum(matched) / len(matched) if matched else None,
        "target_accuracy_max": max(targets) if targets else None,
        "best_control_accuracy_max": max(controls) if controls else None,
        "delta_vs_best_control_min": min(deltas) if deltas else None,
        "lift_vs_target_min": min(lifts) if lifts else None,
        "control_over_target_max": max(control_over_target) if control_over_target else None,
    }


def _clean_threshold_range(rows: list[dict[str, Any]]) -> dict[str, Any]:
    clean = [float(row["threshold"]) for row in rows if row["all_rows_clean"]]
    if not clean:
        return {"exists": False, "min": None, "max": None, "count": 0}
    return {"exists": True, "min": min(clean), "max": max(clean), "count": len(clean)}


def build_threshold_frontier(
    *,
    method_roots: dict[str, list[pathlib.Path]],
    output_dir: pathlib.Path,
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    direction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for method_id, roots in method_roots.items():
        for threshold in thresholds:
            rows_for_threshold: list[dict[str, Any]] = []
            for root in roots:
                for direction in ("core_to_holdout", "holdout_to_core", "same_family_all"):
                    prediction_path = _resolve(root) / direction / "predictions_budget8.jsonl"
                    if not prediction_path.exists():
                        continue
                    row = _threshold_direction_row(
                        method_id=method_id,
                        root=root,
                        direction=direction,
                        threshold=threshold,
                    )
                    rows_for_threshold.append(row)
                    direction_rows.append(row)
            if rows_for_threshold:
                summary_rows.append(_aggregate_threshold_rows(method_id, threshold, rows_for_threshold))

    by_method = {
        method_id: [row for row in summary_rows if row["method_id"] == method_id]
        for method_id in method_roots
    }
    live_at_048 = next(
        row
        for row in by_method["live_candidate_local_residual_norm"]
        if abs(float(row["threshold"]) - 0.48) <= 1e-9
    )
    headline = {
        "pass_gate": bool(live_at_048["all_rows_clean"]),
        "live_threshold_0_48_clean_rows": live_at_048["clean_rows"],
        "live_threshold_0_48_rows": live_at_048["rows"],
        "live_clean_threshold_range": _clean_threshold_range(by_method["live_candidate_local_residual_norm"]),
        "rr_clean_threshold_range": _clean_threshold_range(by_method.get("rr_anchor_coordinate_dot", [])),
        "random_rotation_sign_clean_threshold_range": _clean_threshold_range(
            by_method.get("public_random_rotation_sign", [])
        ),
        "interpretation": (
            "A method has a clean threshold only when every replayed direction keeps matched accuracy at least "
            "0.15 above target, at least 0.10 above the best destructive control, and every strict destructive "
            "control remains within target+0.03. This diagnostic ignores bootstrap and knockout constraints, "
            "so it is a threshold robustness frontier rather than the original pass gate."
        ),
    }

    payload = {
        "gate": "source_private_candidate_local_threshold_frontier",
        "thresholds": list(thresholds),
        "method_roots": {method: [_relative(root) for root in roots] for method, roots in method_roots.items()},
        "headline": headline,
        "summary_rows": summary_rows,
        "direction_rows": direction_rows,
        "layman_explanation": (
            "This replays stored model scores as if we had chosen different confidence cutoffs. If a method is "
            "real and robust, there should be a band of cutoffs where the real packet helps but fake packets do "
            "not. If fake packets rise in the same band, the method is not clean source communication."
        ),
    }

    json_path = output_dir / "candidate_local_threshold_frontier.json"
    csv_path = output_dir / "candidate_local_threshold_frontier.csv"
    row_csv_path = output_dir / "candidate_local_threshold_frontier_direction_rows.csv"
    md_path = output_dir / "candidate_local_threshold_frontier.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({column: _fmt(row.get(column)) for column in CSV_COLUMNS})
    with row_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_COLUMNS)
        writer.writeheader()
        for row in direction_rows:
            writer.writerow({column: _fmt(row.get(column)) for column in ROW_COLUMNS})
    md_lines = [
        "# Candidate-Local Threshold Frontier",
        "",
        f"- pass gate: `{headline['pass_gate']}`",
        f"- live threshold 0.48 clean rows: `{headline['live_threshold_0_48_clean_rows']}/{headline['live_threshold_0_48_rows']}`",
        f"- live clean threshold range: `{headline['live_clean_threshold_range']}`",
        f"- RR clean threshold range: `{headline['rr_clean_threshold_range']}`",
        f"- random-rotation sign clean threshold range: `{headline['random_rotation_sign_clean_threshold_range']}`",
        "",
        "| method | threshold | clean rows | min matched | max best control | min matched-control |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        if row["threshold"] in {0.0, 0.30, 0.48, 0.60}:
            md_lines.append(
                "| "
                f"{row['method_id']} | {row['threshold']:.2f} | {row['clean_rows']}/{row['rows']} | "
                f"{row['matched_accuracy_min']:.3f} | {row['best_control_accuracy_max']:.3f} | "
                f"{row['delta_vs_best_control_min']:.3f} |"
            )
    md_lines.extend(["", payload["layman_explanation"], ""])
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    manifest = {
        "artifacts": [
            "candidate_local_threshold_frontier.json",
            "candidate_local_threshold_frontier.csv",
            "candidate_local_threshold_frontier_direction_rows.csv",
            "candidate_local_threshold_frontier.md",
        ],
        "sha256": {
            path.name: _sha256_file(path)
            for path in (json_path, csv_path, row_csv_path, md_path)
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _parse_method_root(value: str) -> tuple[str, pathlib.Path]:
    method, sep, root = value.partition("=")
    if not sep or not method or not root:
        raise argparse.ArgumentTypeError("method roots must use METHOD=PATH")
    return method, pathlib.Path(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--method-root",
        action="append",
        type=_parse_method_root,
        default=[],
        help="Add or override a method root as METHOD=PATH. May be repeated.",
    )
    args = parser.parse_args()
    method_roots = {method: list(roots) for method, roots in DEFAULT_METHOD_ROOTS.items()}
    for method, root in args.method_root:
        method_roots.setdefault(method, []).append(root)
    payload = build_threshold_frontier(method_roots=method_roots, output_dir=args.output_dir)
    print(json.dumps({"output_dir": str(_resolve(args.output_dir)), "pass_gate": payload["headline"]["pass_gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
