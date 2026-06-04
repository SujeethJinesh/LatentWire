from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable

from pmc.cache_split import cache_unit_for, discover_cache_files, split_for_key


ALLOWED_MAC_STATUSES = {
    "KILLED",
    "AMBIGUOUS",
    "CPU_SCREENED",
    "PARKED_NEEDS_GPU",
    "PROVISIONAL_PROMOTE_TO_GPU",
}


@dataclass(frozen=True)
class ParsedRow:
    cache_family: str
    path: str
    row_id: str
    split: str
    payload: dict


def row_split(path: Path, row_id: str) -> str:
    unit = cache_unit_for(path).as_posix()
    return split_for_key(f"{unit}|{path.as_posix()}|{row_id}")


def row_id_from_payload(payload: dict, fallback: str) -> str:
    for key in ("row_id", "example_id", "prompt_id", "trace_id", "id", "content_id"):
        if key in payload:
            return str(payload[key])
    return fallback


def read_jsonl_rows(path: Path) -> list[ParsedRow]:
    rows: list[ParsedRow] = []
    family = cache_unit_for(path).as_posix()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            payload = json.loads(line)
            row_id = row_id_from_payload(payload, f"line:{index}")
            split = row_split(path, row_id)
            rows.append(ParsedRow(family, path.as_posix(), row_id, split, payload))
    return rows


def read_per_trace_rows(path: Path) -> list[ParsedRow]:
    payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    traces = payload.get("traces", [])
    rows: list[ParsedRow] = []
    family = cache_unit_for(path).as_posix()
    for index, trace in enumerate(traces):
        row_id = row_id_from_payload(trace, f"trace:{index}")
        split = row_split(path, row_id)
        rows.append(ParsedRow(family, path.as_posix(), row_id, split, trace))
    return rows


def is_stage1_output_path(path: Path) -> bool:
    parts = path.parts
    return any(
        part == "results" and idx + 1 < len(parts) and parts[idx + 1] == "stage1"
        for idx, part in enumerate(parts)
    )


def discover_stage1_cache_files(roots: Iterable[Path]) -> list[Path]:
    return [path for path in discover_cache_files(roots) if not is_stage1_output_path(path)]


def discover_parseable_files(roots: Iterable[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    files = discover_stage1_cache_files(roots)
    wz_predictions = [
        path
        for path in files
        if path.name.startswith("predictions_budget")
        and path.suffix == ".jsonl"
        and "source_private_wyner_ziv" in path.as_posix()
    ]
    fixed_packet_predictions = [
        path
        for path in files
        if path.name == "predictions.jsonl"
        and (
            "source_private_arc_challenge_fixed_packet_gate" in path.as_posix()
            or "source_private_openbookqa_fixed_packet_gate" in path.as_posix()
            or "source_private_hellaswag_fixed_packet_gate" in path.as_posix()
        )
    ]
    channel_per_trace = [
        path
        for path in files
        if path.name == "per_trace_metrics.json"
        and "experimental/outlier_migrate/phase9/results" in path.as_posix()
    ]
    return sorted(wz_predictions), sorted(fixed_packet_predictions), sorted(channel_per_trace)


def parse_coverage(roots: Iterable[Path]) -> tuple[list[dict], dict[str, list[ParsedRow]]]:
    files = discover_stage1_cache_files(roots)
    wz_files, fixed_files, channel_files = discover_parseable_files(roots)
    parseable = set(wz_files + fixed_files + channel_files)
    parsed_by_path: dict[str, list[ParsedRow]] = {}
    coverage: list[dict] = []
    for path in files:
        family = cache_unit_for(path).as_posix()
        if path in parseable:
            if path in channel_files:
                rows = read_per_trace_rows(path)
                parser = "per_trace_metrics_json"
            else:
                rows = read_jsonl_rows(path)
                parser = "predictions_jsonl"
            counts = Counter(row.split for row in rows)
            parsed_by_path[path.as_posix()] = rows
            status = "row_parsed"
            usable = True
            blocked = ""
        else:
            rows = []
            counts = Counter()
            parser = "none"
            status = "quarantined_unimplemented_parser"
            usable = False
            blocked = "no row parser for this cache family in bounded pass"
        coverage.append(
            {
                "cache_family": family,
                "path": path.as_posix(),
                "files_seen": 1,
                "rows_seen": len(rows),
                "rows_dev": counts.get("dev", 0),
                "rows_gate": counts.get("gate", 0),
                "rows_confirm": counts.get("confirm", 0),
                "parse_status": status,
                "parser": parser,
                "usable_for_methods": str(usable).lower(),
                "blocked_reason": blocked,
            }
        )
    return coverage, parsed_by_path


def condition_accuracy(rows: list[ParsedRow]) -> dict[str, dict]:
    groups: dict[str, list[ParsedRow]] = defaultdict(list)
    for row in rows:
        condition = str(row.payload.get("condition", "unknown"))
        groups[condition].append(row)
    out: dict[str, dict] = {}
    for condition, condition_rows in groups.items():
        correct = sum(1 for row in condition_rows if bool(row.payload.get("correct")))
        n = len(condition_rows)
        out[condition] = {"n": n, "correct": correct, "accuracy": correct / n if n else None}
    return out


def paired_delta_ci(
    rows: list[ParsedRow], condition_a: str, condition_b: str, samples: int = 500
) -> dict[str, float | int | None]:
    by_condition: dict[str, dict[str, int]] = defaultdict(dict)
    for row in rows:
        condition = str(row.payload.get("condition", "unknown"))
        by_condition[condition][row.row_id] = 1 if bool(row.payload.get("correct")) else 0
    common = sorted(set(by_condition[condition_a]) & set(by_condition[condition_b]))
    if not common:
        return {"n": 0, "delta": None, "ci95_low": None, "ci95_high": None}
    diffs = [by_condition[condition_a][rid] - by_condition[condition_b][rid] for rid in common]
    delta = sum(diffs) / len(diffs)
    rng = random.Random(20260604)
    boot = []
    for _ in range(samples):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        boot.append(sum(sample) / len(sample))
    boot.sort()
    low = boot[int(0.025 * (len(boot) - 1))]
    high = boot[int(0.975 * (len(boot) - 1))]
    return {"n": len(diffs), "delta": delta, "ci95_low": low, "ci95_high": high}


def summarize_latentwire(rows_by_path: dict[str, list[ParsedRow]]) -> tuple[list[dict], list[dict]]:
    raw_rows: list[dict] = []
    summaries: list[dict] = []
    for path, rows in sorted(rows_by_path.items()):
        if "source_private_wyner_ziv" not in path and "fixed_packet_gate" not in path:
            continue
        is_wz_candidate = "source_private_wyner_ziv" in path
        method_id = (
            "L_SCORECOMP_wz_bins_deployable"
            if is_wz_candidate
            else "latentwire_cached_fixed_packet_baseline"
        )
        consumed = [row for row in rows if row.split in {"dev", "gate"}]
        raw_rows.extend(
            {
                "paper": "latentwire",
                "path": row.path,
                "row_id": row.row_id,
                "split": row.split,
                "condition": row.payload.get("condition"),
                "correct": row.payload.get("correct"),
                "budget_bytes": row.payload.get("budget_bytes", row.payload.get("payload_bytes")),
                "payload_bytes": row.payload.get("payload_bytes"),
            }
            for row in consumed
        )
        for split in ("dev", "gate"):
            split_rows = [row for row in consumed if row.split == split]
            if not split_rows:
                continue
            acc = condition_accuracy(split_rows)
            if "matched_learned_syndrome" in acc:
                matched = "matched_learned_syndrome"
            elif "matched_source_private_packet" in acc:
                matched = "matched_source_private_packet"
            else:
                matched = None
            if matched is None:
                continue
            baseline_conditions = [
                c
                for c in acc
                if c != matched
                and any(
                    marker in c
                    for marker in (
                        "target_only",
                        "random_same_byte",
                        "shuffled",
                        "zero_source",
                        "score_source",
                        "scalar_quantized_source",
                        "label_shuffled",
                    )
                )
            ]
            best_baseline = max(
                baseline_conditions,
                key=lambda condition: acc[condition]["accuracy"]
                if acc[condition]["accuracy"] is not None
                else -1,
                default=None,
            )
            paired = (
                paired_delta_ci(split_rows, matched, best_baseline)
                if best_baseline is not None
                else {"n": 0, "delta": None, "ci95_low": None, "ci95_high": None}
            )
            matched_acc = acc[matched]["accuracy"]
            best_acc = acc[best_baseline]["accuracy"] if best_baseline else None
            status = "AMBIGUOUS"
            if split == "gate" and paired["ci95_low"] is not None:
                if paired["ci95_low"] > 0:
                    status = "PROVISIONAL_PROMOTE_TO_GPU"
                elif paired["ci95_high"] < 0:
                    status = "KILLED"
                else:
                    status = "CPU_SCREENED"
            if not is_wz_candidate and split == "gate":
                status = "CPU_SCREENED"
            summaries.append(
                {
                    "method_id": method_id,
                    "paper": "latentwire",
                    "source_path": path,
                    "split": split,
                    "status": status,
                    "matched_condition": matched,
                    "matched_accuracy": matched_acc,
                    "best_baseline_condition": best_baseline,
                    "best_baseline_accuracy": best_acc,
                    "delta_vs_best_baseline": paired["delta"],
                    "ci95_low_vs_best_baseline": paired["ci95_low"],
                    "ci95_high_vs_best_baseline": paired["ci95_high"],
                    "paired_n": paired["n"],
                    "note": (
                        "baseline context only; not a candidate-method promotion"
                        if not is_wz_candidate
                        else "dev/gate row-filtered cache screen; no Mac PASSED status"
                    ),
                }
            )
    return raw_rows, summaries


def cvar(values: list[float], alpha: float = 0.25) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = max(1, int(len(ordered) * alpha))
    return sum(ordered[:k]) / k


def summarize_channel_set(rows_by_path: dict[str, list[ParsedRow]]) -> tuple[list[dict], list[dict]]:
    raw_rows: list[dict] = []
    summaries: list[dict] = []
    for path, rows in sorted(rows_by_path.items()):
        if "experimental/outlier_migrate" not in path or not path.endswith("per_trace_metrics.json"):
            continue
        consumed = [row for row in rows if row.split in {"dev", "gate"}]
        raw_rows.extend(
            {
                "paper": "channel_set",
                "path": row.path,
                "row_id": row.row_id,
                "split": row.split,
                "static_gap": row.payload.get("static_gap"),
                "no_recoverable_static_gap": row.payload.get("no_recoverable_static_gap"),
                "recoveries": row.payload.get("recoveries", {}),
            }
            for row in consumed
        )
        regimes = sorted(
            {
                regime
                for row in consumed
                for regime, value in row.payload.get("recoveries", {}).items()
                if value is not None
            }
        )
        for split in ("dev", "gate"):
            split_rows = [row for row in consumed if row.split == split]
            no_gap = sum(1 for row in split_rows if row.payload.get("no_recoverable_static_gap"))
            total = len(split_rows)
            for regime in regimes:
                values = [
                    float(row.payload["recoveries"][regime])
                    for row in split_rows
                    if row.payload.get("recoveries", {}).get(regime) is not None
                ]
                if not values:
                    continue
                status = "CPU_SCREENED" if split == "gate" and len(values) >= 2 else "AMBIGUOUS"
                method_id = "C_F_survival_stable_core"
                lower_path = path.lower()
                if "clip" in lower_path or "paroquant" in lower_path:
                    method_id = "C_A1_cvar_evt_clip_grid"
                if "decdec" in lower_path:
                    method_id = "C_D1_osc_static_cluster_stress"
                summaries.append(
                    {
                        "method_id": method_id,
                        "paper": "channel_set",
                        "source_path": path,
                        "split": split,
                        "status": status,
                        "regime": regime,
                        "n": len(values),
                        "median_recovery": median(values),
                        "cvar25_recovery": cvar(values),
                        "worst_recovery": min(values),
                        "no_gap_rows": no_gap,
                        "total_rows": total,
                        "note": "offline row-filtered per-trace recovery; confirmation still needs held-out/fresh or GPU as applicable",
                    }
                )
        if total := len(consumed):
            summaries.append(
                {
                    "method_id": "CE21_no_gap_filter",
                    "paper": "channel_set",
                    "source_path": path,
                    "split": "dev_gate",
                    "status": "CPU_SCREENED",
                    "regime": "no_gap_denominator_audit",
                    "n": total,
                    "median_recovery": "",
                    "cvar25_recovery": "",
                    "worst_recovery": "",
                    "no_gap_rows": sum(
                        1 for row in consumed if row.payload.get("no_recoverable_static_gap")
                    ),
                    "total_rows": total,
                    "note": "denominator audit only; cannot promote a method",
                }
            )
    summaries.extend(
        [
            {
                "method_id": "CE13_warmup_policy_selector",
                "paper": "channel_set",
                "source_path": "",
                "split": "dev_gate",
                "status": "PARKED_NEEDS_GPU",
                "regime": "warmup_policy_selector",
                "n": 0,
                "median_recovery": "",
                "cvar25_recovery": "",
                "worst_recovery": "",
                "no_gap_rows": "",
                "total_rows": "",
                "note": "no parseable warmup-policy cache found in bounded pass",
            },
            {
                "method_id": "C_A2_horizon_rotation",
                "paper": "channel_set",
                "source_path": "",
                "split": "tests_only",
                "status": "PARKED_NEEDS_GPU",
                "regime": "tests_only",
                "n": 0,
                "median_recovery": "",
                "cvar25_recovery": "",
                "worst_recovery": "",
                "no_gap_rows": "",
                "total_rows": "",
                "note": "correctness tests only; no method screen launched",
            },
        ]
    )
    return raw_rows, summaries


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        field_set: list[str] = []
        for row in rows:
            for key in row:
                if key not in field_set:
                    field_set.append(key)
        fields = field_set
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def markdown_table(rows: list[dict], fields: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines) + "\n"
