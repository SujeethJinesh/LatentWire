from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import pathlib
import statistics
import sys
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_source_private_learned_synonym_dictionary_packet_gate import STRICT_SOURCE_DESTROYING_CONTROLS  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path("results/source_private_candidate_local_margin_atlas_20260501")

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
    "procrustes_common_basis": [
        pathlib.Path("results/source_private_candidate_local_residual_ablation_20260430_seed47_n512_procrustes_dot_evaldisjoint"),
        pathlib.Path("results/source_private_candidate_local_residual_ablation_20260430_seed53_n512_procrustes_dot_evaldisjoint"),
        pathlib.Path("results/source_private_candidate_local_residual_ablation_20260430_seed59_n512_procrustes_dot_evaldisjoint"),
    ],
    "public_random_rotation_sign": [
        pathlib.Path("results/source_private_candidate_local_random_rotation_sign_20260501_seed47_n128_evaldisjoint_tau0"),
    ],
}

FOCUS_CONDITIONS = (
    "learned_synonym_dictionary_packet",
    "oracle_learned_candidate_atoms",
    "zero_source",
    "shuffled_source",
    "random_same_byte",
    "atom_id_derangement",
    "private_random_source_atoms",
    "permuted_teacher_receiver",
)
EXAMPLE_CONDITIONS = ("target_only",) + FOCUS_CONDITIONS

SUMMARY_COLUMNS = (
    "method_id",
    "condition",
    "n",
    "stored_accuracy",
    "accuracy_at_048",
    "argmax_accuracy",
    "positive_margin_rate",
    "accepted_rate_at_048",
    "correct_high_conf_rate",
    "wrong_high_conf_rate",
    "answer_rank1_rate",
    "margin_mean",
    "margin_p10",
    "margin_p50",
    "margin_p90",
    "winner_margin_p50",
    "payload_bytes_mean",
    "latency_ms_p50",
    "worst_family",
    "worst_family_accuracy",
    "worst_family_positive_margin_rate",
)

DIRECTION_COLUMNS = (
    "method_id",
    "direction",
    "condition",
    "n",
    "stored_accuracy",
    "accuracy_at_048",
    "argmax_accuracy",
    "positive_margin_rate",
    "accepted_rate_at_048",
    "margin_p10",
    "margin_p50",
    "margin_p90",
    "worst_family",
)

EXAMPLE_COLUMNS = (
    "method_id",
    "root",
    "direction",
    "condition",
    "example_id",
    "family_name",
    "answer_index",
    "prior_index",
    "stored_correct",
    "answer_score",
    "prior_score",
    "best_nonanswer_score",
    "top1_score",
    "top2_score",
    "answer_margin",
    "winner_margin",
    "prior_margin",
    "answer_rank",
    "margin_bucket",
    "accepted_at_048",
    "predicted_index_at_048",
    "correct_at_048",
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


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    resolved = _resolve(path)
    if not resolved.exists():
        return []
    return [json.loads(line) for line in resolved.read_text(encoding="utf-8").splitlines() if line.strip()]


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _gold_margin(row: dict[str, Any]) -> float | None:
    scores = row.get("metadata", {}).get("scores")
    if not scores:
        return None
    values = [float(score) for score in scores]
    answer_index = int(row["answer_index"])
    if answer_index < 0 or answer_index >= len(values) or len(values) <= 1:
        return None
    other = max(score for idx, score in enumerate(values) if idx != answer_index)
    return values[answer_index] - other


def _argmax_correct(row: dict[str, Any]) -> bool | None:
    scores = row.get("metadata", {}).get("scores")
    if not scores:
        return None
    values = [float(score) for score in scores]
    answer_index = int(row["answer_index"])
    best = max(values)
    return abs(values[answer_index] - best) <= 1e-8


def _predicted_index_at_threshold(row: dict[str, Any], threshold: float = 0.48) -> int:
    prior_index = int(row.get("prior_index", 0))
    scores = row.get("metadata", {}).get("scores")
    if not scores:
        return prior_index
    values = [float(score) for score in scores]
    best = max(values)
    if best < threshold:
        return prior_index
    tied = [idx for idx, score in enumerate(values) if abs(score - best) <= 1e-8]
    return prior_index if prior_index in tied else tied[0]


def _accepted_at_threshold(row: dict[str, Any], threshold: float = 0.48) -> bool:
    scores = row.get("metadata", {}).get("scores")
    return bool(scores) and max(float(score) for score in scores) >= threshold


def _answer_rank(row: dict[str, Any]) -> int | None:
    scores = row.get("metadata", {}).get("scores")
    if not scores:
        return None
    values = [float(score) for score in scores]
    answer_score = values[int(row["answer_index"])]
    return 1 + sum(score > answer_score for score in values)


def _winner_margin(row: dict[str, Any]) -> float | None:
    scores = row.get("metadata", {}).get("scores")
    if not scores or len(scores) <= 1:
        return None
    values = sorted((float(score) for score in scores), reverse=True)
    return values[0] - values[1]


def _margin_bucket(margin: float | None) -> str:
    if margin is None:
        return "missing"
    if margin < 0:
        return "<0"
    if margin < 0.05:
        return "0-0.05"
    if margin < 0.10:
        return "0.05-0.10"
    if margin < 0.25:
        return "0.10-0.25"
    return ">=0.25"


def _example_margin_row(
    *,
    method_id: str,
    root: pathlib.Path,
    direction: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    scores = row.get("metadata", {}).get("scores")
    values = [float(score) for score in scores] if scores else []
    answer_index = int(row["answer_index"])
    prior_index = int(row.get("prior_index", 0))
    answer_score = values[answer_index] if values else None
    prior_score = values[prior_index] if values and 0 <= prior_index < len(values) else None
    best_nonanswer = max((score for idx, score in enumerate(values) if idx != answer_index), default=None)
    top_scores = sorted(values, reverse=True)
    top1 = top_scores[0] if top_scores else None
    top2 = top_scores[1] if len(top_scores) > 1 else None
    answer_margin = _gold_margin(row)
    predicted_index = _predicted_index_at_threshold(row)
    return {
        "method_id": method_id,
        "root": _relative(root),
        "direction": direction,
        "condition": row["condition"],
        "example_id": row["example_id"],
        "family_name": row.get("family_name", "unknown"),
        "answer_index": answer_index,
        "prior_index": prior_index,
        "stored_correct": bool(row.get("correct")),
        "answer_score": answer_score,
        "prior_score": prior_score,
        "best_nonanswer_score": best_nonanswer,
        "top1_score": top1,
        "top2_score": top2,
        "answer_margin": answer_margin,
        "winner_margin": _winner_margin(row),
        "prior_margin": (answer_score - prior_score) if answer_score is not None and prior_score is not None else None,
        "answer_rank": _answer_rank(row),
        "margin_bucket": _margin_bucket(answer_margin),
        "accepted_at_048": _accepted_at_threshold(row),
        "predicted_index_at_048": predicted_index,
        "correct_at_048": predicted_index == answer_index,
    }


def _worst_family(rows: list[dict[str, Any]], margins: list[float | None]) -> dict[str, Any]:
    by_family: dict[str, list[tuple[bool, float | None]]] = {}
    for row, margin in zip(rows, margins, strict=True):
        by_family.setdefault(str(row.get("family_name", "unknown")), []).append((bool(row.get("correct")), margin))
    if not by_family:
        return {"family": None, "accuracy": None, "positive_margin_rate": None}
    summaries = []
    for family, values in by_family.items():
        accuracy = sum(correct for correct, _ in values) / len(values)
        margin_values = [margin for _, margin in values if margin is not None]
        positive = sum(margin > 0 for margin in margin_values) / len(margin_values) if margin_values else 0.0
        summaries.append((accuracy, positive, family))
    accuracy, positive, family = sorted(summaries, key=lambda item: (item[0], item[1], item[2]))[0]
    return {"family": family, "accuracy": accuracy, "positive_margin_rate": positive}


def _summarize_rows(method_id: str, condition: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    margins = [_gold_margin(row) for row in rows]
    margin_values = [margin for margin in margins if margin is not None]
    argmax_values = [_argmax_correct(row) for row in rows]
    argmax_known = [value for value in argmax_values if value is not None]
    predicted_at_048 = [_predicted_index_at_threshold(row) for row in rows]
    accepted_at_048 = [_accepted_at_threshold(row) for row in rows]
    correct_at_048 = [prediction == int(row["answer_index"]) for prediction, row in zip(predicted_at_048, rows, strict=True)]
    ranks = [_answer_rank(row) for row in rows]
    known_ranks = [rank for rank in ranks if rank is not None]
    winner_margins = [_winner_margin(row) for row in rows]
    winner_margin_values = [margin for margin in winner_margins if margin is not None]
    payload_bytes = [float(row.get("payload_bytes", 0.0)) for row in rows]
    latencies = [float(row.get("latency_ms", 0.0)) for row in rows]
    worst = _worst_family(rows, margins)
    return {
        "method_id": method_id,
        "condition": condition,
        "n": len(rows),
        "stored_accuracy": sum(bool(row.get("correct")) for row in rows) / len(rows) if rows else None,
        "accuracy_at_048": sum(correct_at_048) / len(correct_at_048) if correct_at_048 else None,
        "argmax_accuracy": sum(bool(value) for value in argmax_known) / len(argmax_known) if argmax_known else None,
        "positive_margin_rate": (
            sum(margin > 0 for margin in margin_values) / len(margin_values) if margin_values else None
        ),
        "accepted_rate_at_048": sum(accepted_at_048) / len(accepted_at_048) if accepted_at_048 else None,
        "correct_high_conf_rate": (
            sum(accepted and correct for accepted, correct in zip(accepted_at_048, correct_at_048, strict=True))
            / len(rows)
            if rows
            else None
        ),
        "wrong_high_conf_rate": (
            sum(accepted and not correct for accepted, correct in zip(accepted_at_048, correct_at_048, strict=True))
            / len(rows)
            if rows
            else None
        ),
        "answer_rank1_rate": sum(rank == 1 for rank in known_ranks) / len(known_ranks) if known_ranks else None,
        "margin_mean": statistics.fmean(margin_values) if margin_values else None,
        "margin_p10": _percentile(margin_values, 0.10),
        "margin_p50": _percentile(margin_values, 0.50),
        "margin_p90": _percentile(margin_values, 0.90),
        "winner_margin_p50": _percentile(winner_margin_values, 0.50),
        "payload_bytes_mean": statistics.fmean(payload_bytes) if payload_bytes else None,
        "latency_ms_p50": _percentile(latencies, 0.50),
        "worst_family": worst["family"],
        "worst_family_accuracy": worst["accuracy"],
        "worst_family_positive_margin_rate": worst["positive_margin_rate"],
    }


def _collect_rows(method_roots: dict[str, list[pathlib.Path]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    for method_id, roots in method_roots.items():
        by_condition: dict[str, list[dict[str, Any]]] = {condition: [] for condition in FOCUS_CONDITIONS}
        by_direction_condition: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for root in roots:
            for direction in ("core_to_holdout", "holdout_to_core", "same_family_all"):
                rows = _read_jsonl(root / direction / "predictions_budget8.jsonl")
                for row in rows:
                    condition = row["condition"]
                    if condition in EXAMPLE_CONDITIONS:
                        example_rows.append(_example_margin_row(method_id=method_id, root=root, direction=direction, row=row))
                    if condition not in FOCUS_CONDITIONS:
                        continue
                    by_condition.setdefault(condition, []).append(row)
                    by_direction_condition.setdefault((direction, condition), []).append(row)
        for condition, rows in by_condition.items():
            if rows:
                aggregate_rows.append(_summarize_rows(method_id, condition, rows))
        for (direction, condition), rows in sorted(by_direction_condition.items()):
            if rows:
                summary = _summarize_rows(method_id, condition, rows)
                summary["direction"] = direction
                direction_rows.append(summary)
    return aggregate_rows, direction_rows, example_rows


def _best_control_summary(rows: list[dict[str, Any]], method_id: str) -> dict[str, Any]:
    controls = [
        row
        for row in rows
        if row["method_id"] == method_id and row["condition"] in STRICT_SOURCE_DESTROYING_CONTROLS
    ]
    if not controls:
        return {}
    return max(
        controls,
        key=lambda row: (
            row["stored_accuracy"] if row["stored_accuracy"] is not None else -1.0,
            row["positive_margin_rate"] if row["positive_margin_rate"] is not None else -1.0,
        ),
    )


def _condition_summary(rows: list[dict[str, Any]], method_id: str, condition: str) -> dict[str, Any]:
    return next(row for row in rows if row["method_id"] == method_id and row["condition"] == condition)


def _write_plot(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    plot_rows: list[tuple[str, dict[str, Any]]] = []
    for method_id in dict.fromkeys(row["method_id"] for row in rows):
        matched = _condition_summary(rows, method_id, "learned_synonym_dictionary_packet")
        best_control = _best_control_summary(rows, method_id)
        plot_rows.append((f"{method_id} matched", matched))
        if best_control:
            plot_rows.append((f"{method_id} best control", best_control))

    width = 1120
    left = 330
    right = 50
    top = 64
    row_h = 34
    height = top + len(plot_rows) * row_h + 58
    values = [0.0]
    for _, row in plot_rows:
        values.extend(float(row[key] or 0.0) for key in ("margin_p10", "margin_p50", "margin_p90"))
    lo = min(values)
    hi = max(values)
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    pad = max((hi - lo) * 0.08, 0.05)
    lo -= pad
    hi += pad
    plot_w = width - left - right

    def x(value: float) -> float:
        return left + (value - lo) / (hi - lo) * plot_w

    zero_x = x(0.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700">Candidate-local margin separation</text>',
        '<text x="24" y="56" font-family="Arial, sans-serif" font-size="13" fill="#555555">Bars show median gold-candidate margin; whiskers show p10-p90.</text>',
        f'<line x1="{zero_x:.2f}" y1="{top - 12}" x2="{zero_x:.2f}" y2="{height - 42}" stroke="#111111" stroke-width="1"/>',
    ]
    for tick in (lo, 0.0, hi):
        tx = x(tick)
        parts.append(f'<line x1="{tx:.2f}" y1="{height - 40}" x2="{tx:.2f}" y2="{height - 34}" stroke="#333333" stroke-width="1"/>')
        parts.append(
            f'<text x="{tx:.2f}" y="{height - 18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#333333">{tick:.2f}</text>'
        )
    for idx, (label, row) in enumerate(plot_rows):
        y = top + idx * row_h
        median = float(row["margin_p50"] or 0.0)
        p10 = float(row["margin_p10"] or 0.0)
        p90 = float(row["margin_p90"] or 0.0)
        color = "#2f6f9f" if label.endswith("matched") else "#9f3f2f"
        bar_x = min(zero_x, x(median))
        bar_w = max(abs(x(median) - zero_x), 1.0)
        parts.append(
            f'<text x="24" y="{y + 19}" font-family="Arial, sans-serif" font-size="12" fill="#222222">{html.escape(label)}</text>'
        )
        parts.append(f'<line x1="{x(p10):.2f}" y1="{y + 15}" x2="{x(p90):.2f}" y2="{y + 15}" stroke="#222222" stroke-width="1.5"/>')
        parts.append(f'<rect x="{bar_x:.2f}" y="{y + 7}" width="{bar_w:.2f}" height="16" fill="{color}" fill-opacity="0.84"/>')
        parts.append(
            f'<text x="{x(median) + 6:.2f}" y="{y + 20}" font-family="Arial, sans-serif" font-size="11" fill="#222222">{median:.2f}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    headline = payload["headline"]
    lines = [
        "# Candidate-Local Margin Atlas",
        "",
        f"- pass gate: `{headline['pass_gate']}`",
        f"- live matched positive-margin rate: `{headline['live_matched_positive_margin_rate']:.3f}`",
        f"- live best-control positive-margin rate: `{headline['live_best_control_positive_margin_rate']:.3f}`",
        f"- live oracle positive-margin rate: `{headline['live_oracle_positive_margin_rate']:.3f}`",
        f"- procrustes matched/control positive-margin rates: `{headline['procrustes_matched_positive_margin_rate']:.3f}` / `{headline['procrustes_best_control_positive_margin_rate']:.3f}`",
        "",
        "| method | condition | n | stored acc | argmax acc | pos margin | p10 | p50 | p90 | worst family |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    wanted = {"learned_synonym_dictionary_packet", "oracle_learned_candidate_atoms"}
    for row in payload["summary_rows"]:
        best_control = payload["headline"]["best_control_by_method"].get(row["method_id"], {}).get("condition")
        if row["condition"] not in wanted and row["condition"] != best_control:
            continue
        lines.append(
            "| "
            f"{row['method_id']} | {row['condition']} | {row['n']} | "
            f"{row['stored_accuracy']:.3f} | {row['argmax_accuracy']:.3f} | "
            f"{row['positive_margin_rate']:.3f} | {row['margin_p10']:.3f} | "
            f"{row['margin_p50']:.3f} | {row['margin_p90']:.3f} | {row['worst_family']} |"
        )
    lines.extend(["", payload["layman_explanation"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_margin_atlas(
    *,
    method_roots: dict[str, list[pathlib.Path]],
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows, direction_rows, example_rows = _collect_rows(method_roots)

    best_control_by_method = {
        method_id: _best_control_summary(summary_rows, method_id)
        for method_id in method_roots
    }
    live_matched = _condition_summary(summary_rows, "live_candidate_local_residual_norm", "learned_synonym_dictionary_packet")
    live_oracle = _condition_summary(summary_rows, "live_candidate_local_residual_norm", "oracle_learned_candidate_atoms")
    live_best_control = best_control_by_method["live_candidate_local_residual_norm"]
    procrustes_matched = _condition_summary(summary_rows, "procrustes_common_basis", "learned_synonym_dictionary_packet")
    procrustes_best_control = best_control_by_method["procrustes_common_basis"]
    rr_matched = _condition_summary(summary_rows, "rr_anchor_coordinate_dot", "learned_synonym_dictionary_packet")
    sign_matched = _condition_summary(summary_rows, "public_random_rotation_sign", "learned_synonym_dictionary_packet")
    headline = {
        "pass_gate": (
            live_matched["positive_margin_rate"] >= 0.70
            and live_best_control["stored_accuracy"] <= 0.30
            and live_oracle["positive_margin_rate"] >= 0.85
            and procrustes_best_control["positive_margin_rate"] >= procrustes_matched["positive_margin_rate"] - 1e-9
        ),
        "live_matched_positive_margin_rate": live_matched["positive_margin_rate"],
        "live_matched_margin_p50": live_matched["margin_p50"],
        "live_best_control_condition": live_best_control["condition"],
        "live_best_control_stored_accuracy": live_best_control["stored_accuracy"],
        "live_best_control_positive_margin_rate": live_best_control["positive_margin_rate"],
        "live_oracle_positive_margin_rate": live_oracle["positive_margin_rate"],
        "rr_matched_positive_margin_rate": rr_matched["positive_margin_rate"],
        "public_random_rotation_sign_matched_positive_margin_rate": sign_matched["positive_margin_rate"],
        "procrustes_matched_positive_margin_rate": procrustes_matched["positive_margin_rate"],
        "procrustes_best_control_condition": procrustes_best_control["condition"],
        "procrustes_best_control_positive_margin_rate": procrustes_best_control["positive_margin_rate"],
        "best_control_by_method": {
            method_id: {
                "condition": row.get("condition"),
                "stored_accuracy": row.get("stored_accuracy"),
                "positive_margin_rate": row.get("positive_margin_rate"),
                "margin_p50": row.get("margin_p50"),
            }
            for method_id, row in best_control_by_method.items()
        },
    }
    payload = {
        "gate": "source_private_candidate_local_margin_atlas",
        "method_roots": {method: [_relative(root) for root in roots] for method, roots in method_roots.items()},
        "headline": headline,
        "summary_rows": summary_rows,
        "direction_rows": direction_rows,
        "example_rows": example_rows,
        "layman_explanation": (
            "This asks whether the real packet pushes the correct candidate above the wrong candidates. "
            "A positive margin means the score surface favors the answer. The live packet has many positive "
            "margins while its destructive controls stay low; the Procrustes common-basis row has the same "
            "margin profile for a destructive permuted-teacher control, so it is not source-private."
        ),
    }

    json_path = output_dir / "margin_atlas.json"
    csv_path = output_dir / "margin_atlas.csv"
    direction_csv_path = output_dir / "margin_atlas_direction_rows.csv"
    example_csv_path = output_dir / "margin_atlas_example_rows.csv"
    md_path = output_dir / "margin_atlas.md"
    svg_path = output_dir / "margin_atlas.svg"

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({column: _fmt(row.get(column)) for column in SUMMARY_COLUMNS})
    with direction_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DIRECTION_COLUMNS)
        writer.writeheader()
        for row in direction_rows:
            writer.writerow({column: _fmt(row.get(column)) for column in DIRECTION_COLUMNS})
    with example_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXAMPLE_COLUMNS)
        writer.writeheader()
        for row in example_rows:
            writer.writerow({column: _fmt(row.get(column)) for column in EXAMPLE_COLUMNS})
    _write_markdown(md_path, payload)
    _write_plot(svg_path, summary_rows)

    manifest = {
        "artifacts": [
            "margin_atlas.json",
            "margin_atlas.csv",
            "margin_atlas_direction_rows.csv",
            "margin_atlas_example_rows.csv",
            "margin_atlas.md",
            "margin_atlas.svg",
        ],
        "sha256": {
            path.name: _sha256_file(path)
            for path in (json_path, csv_path, direction_csv_path, example_csv_path, md_path, svg_path)
        },
        "pass_gate": headline["pass_gate"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _parse_method_root(value: str) -> tuple[str, pathlib.Path]:
    method, sep, root = value.partition("=")
    if not sep or not method or not root:
        raise argparse.ArgumentTypeError("method roots must use METHOD=PATH")
    return method, pathlib.Path(root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--method-root", action="append", type=_parse_method_root, default=[])
    args = parser.parse_args()
    method_roots = {method: list(roots) for method, roots in DEFAULT_METHOD_ROOTS.items()}
    for method, root in args.method_root:
        method_roots.setdefault(method, []).append(root)
    payload = build_margin_atlas(method_roots=method_roots, output_dir=args.output_dir)
    print(json.dumps({"output_dir": str(_resolve(args.output_dir)), "pass_gate": payload["headline"]["pass_gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
