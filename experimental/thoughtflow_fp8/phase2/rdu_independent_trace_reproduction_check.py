"""Independent-trace no-retuning check for the first-surface rdu_topk branch.

This script keeps the pre-registered rdu_topk rule fixed and changes only the
saved trace inputs. It writes a separate artifact so the first-surface 74-trace gate
remains a cached reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .frozen_sparse_cache_probe import RDU_POLICY_NAME, _run as run_frozen_probe
    from .rdu_no_retune_reproduction_check import (
        DEFAULT_CACHED_INPUT,
        FULL_POLICY,
        PROMOTION_MARGIN,
        RKV_POLICY,
        THIN_POLICY,
        _cached_vs_measured,
        _compact_result,
        _family_separation,
        _fmt_ci,
        _oracle_headroom,
        _promotion_decision,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from frozen_sparse_cache_probe import RDU_POLICY_NAME, _run as run_frozen_probe
    from rdu_no_retune_reproduction_check import (
        DEFAULT_CACHED_INPUT,
        FULL_POLICY,
        PROMOTION_MARGIN,
        RKV_POLICY,
        THIN_POLICY,
        _cached_vs_measured,
        _compact_result,
        _family_separation,
        _fmt_ci,
        _oracle_headroom,
        _promotion_decision,
    )


PHASE2_DIR = Path(__file__).resolve().parent
ROOT = PHASE2_DIR.parents[2]
DEFAULT_JSON_OUTPUT = PHASE2_DIR / "rdu_independent_trace_reproduction_check.json"
DEFAULT_MD_OUTPUT = PHASE2_DIR / "rdu_independent_trace_reproduction_check.md"
DEFAULT_TRACE_INPUTS = (
    ROOT / "results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/source_alone.jsonl",
    ROOT / "results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/text_to_text.jsonl",
    ROOT / "results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/target_alone.jsonl",
)


def _path_label(path: Path) -> str:
    if path.is_absolute() and path.is_relative_to(ROOT):
        return str(path.relative_to(ROOT))
    return str(path)


def _strict_family_pass(family: dict[str, object]) -> dict[str, object]:
    same_family = {
        str(policy): float(margin)
        for policy, margin in family["same_family_margin_nll_vs_rdu"].items()
    }
    cross_family = {
        str(policy): float(margin)
        for policy, margin in family["cross_family_margin_nll_vs_rdu"].items()
    }
    return {
        "same_family_positive": bool(same_family) and all(margin > 0.0 for margin in same_family.values()),
        "cross_family_positive": bool(cross_family) and all(margin > 0.0 for margin in cross_family.values()),
        "same_family_min_margin": min(same_family.values()) if same_family else 0.0,
        "cross_family_min_margin": min(cross_family.values()) if cross_family else 0.0,
    }


def _rows_by_trace(result: dict[str, object]) -> dict[int, dict[str, dict[str, object]]]:
    grouped: dict[int, dict[str, dict[str, object]]] = {}
    for row in result.get("rows", []):
        grouped.setdefault(int(row["trace_id"]), {})[str(row["policy"])] = row
    return grouped


def _infer_prefix_len(row: dict[str, object]) -> float:
    keep_rate = float(row.get("keep_rate", 0.0))
    retained = float(row.get("retained_prefix_tokens", 0.0))
    return retained / keep_rate if keep_rate > 0.0 else retained


def _bucket_label(values: list[float], value: float, low_name: str, mid_name: str, high_name: str) -> str:
    if not values:
        return mid_name
    ordered = sorted(values)
    low = ordered[len(ordered) // 3]
    high = ordered[(2 * len(ordered)) // 3]
    if value <= low:
        return low_name
    if value >= high:
        return high_name
    return mid_name


def _trace_failure_decomposition(result: dict[str, object]) -> dict[str, object]:
    """Summarize where independent-surface `rdu_topk` loses.

    This is intentionally coarse: the frozen probe stores per-policy losses and
    RDU telemetry, but not retained-token sets for every policy. We therefore
    decompose by observable trace properties only.
    """

    grouped = _rows_by_trace(result)
    telemetry_by_trace = {
        int(item["trace_id"]): item["telemetry"]
        for item in result.get("rdu_topk_telemetry", {}).get("per_trace", [])
    }
    trace_rows = []
    prefix_lengths = []
    continuation_lengths = []
    densities = []
    for trace_id, policies in grouped.items():
        if RDU_POLICY_NAME not in policies or RKV_POLICY not in policies:
            continue
        rdu = policies[RDU_POLICY_NAME]
        rkv = policies[RKV_POLICY]
        prefix_len = _infer_prefix_len(rdu)
        continuation_len = float(rdu.get("continuation_tokens", 0.0))
        telemetry = telemetry_by_trace.get(trace_id, {})
        rdu_summary = telemetry.get("rdu_summary", {}) if isinstance(telemetry, dict) else {}
        nonzero = float(rdu_summary.get("nonzero_tokens", 0.0))
        density = nonzero / prefix_len if prefix_len > 0.0 else 0.0
        compressed = {
            policy: float(row["nll"])
            for policy, row in policies.items()
            if policy != FULL_POLICY
        }
        oracle_policy, oracle_nll = min(compressed.items(), key=lambda item: item[1])
        trace_row = {
            "trace_id": trace_id,
            "prefix_len": prefix_len,
            "continuation_tokens": continuation_len,
            "rdu_minus_rkv": float(rdu["nll"]) - float(rkv["nll"]),
            "rdu_minus_thin": float(rdu["nll"]) - float(policies[THIN_POLICY]["nll"]),
            "rdu_oracle_gap": float(rdu["nll"]) - oracle_nll,
            "oracle_policy": oracle_policy,
            "rdu_nonzero_density": density,
        }
        trace_rows.append(trace_row)
        prefix_lengths.append(prefix_len)
        continuation_lengths.append(continuation_len)
        densities.append(density)

    def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
        if not rows:
            return {
                "n": 0,
                "mean_rdu_minus_rkv": 0.0,
                "mean_rdu_minus_thin": 0.0,
                "mean_rdu_oracle_gap": 0.0,
                "rdu_beats_rkv_rate": 0.0,
                "rdu_oracle_hit_rate": 0.0,
            }
        return {
            "n": len(rows),
            "mean_rdu_minus_rkv": sum(float(row["rdu_minus_rkv"]) for row in rows) / len(rows),
            "mean_rdu_minus_thin": sum(float(row["rdu_minus_thin"]) for row in rows) / len(rows),
            "mean_rdu_oracle_gap": sum(float(row["rdu_oracle_gap"]) for row in rows) / len(rows),
            "rdu_beats_rkv_rate": sum(float(row["rdu_minus_rkv"]) < 0.0 for row in rows) / len(rows),
            "rdu_oracle_hit_rate": sum(float(row["rdu_oracle_gap"]) == 0.0 for row in rows) / len(rows),
        }

    groups: dict[str, list[dict[str, object]]] = {
        "all": trace_rows,
    }
    for row in trace_rows:
        prefix_bucket = _bucket_label(prefix_lengths, float(row["prefix_len"]), "short_prefix", "mid_prefix", "long_prefix")
        continuation_bucket = _bucket_label(
            continuation_lengths,
            float(row["continuation_tokens"]),
            "short_continuation",
            "mid_continuation",
            "long_continuation",
        )
        density_bucket = _bucket_label(
            densities,
            float(row["rdu_nonzero_density"]),
            "low_rdu_density",
            "mid_rdu_density",
            "high_rdu_density",
        )
        groups.setdefault(prefix_bucket, []).append(row)
        groups.setdefault(continuation_bucket, []).append(row)
        groups.setdefault(density_bucket, []).append(row)
        groups.setdefault(f"oracle_{row['oracle_policy']}", []).append(row)

    return {
        "status": "coarse_trace_failure_decomposition_from_saved_rows",
        "available_fields": [
            "prefix_len",
            "continuation_tokens",
            "rdu_nonzero_density",
            "oracle_policy",
            "rdu_minus_rkv",
            "rdu_minus_thin",
            "rdu_oracle_gap",
        ],
        "group_summaries": {name: summarize(rows) for name, rows in sorted(groups.items())},
    }


def build_report(
    cached: dict[str, object],
    measured: dict[str, object],
    *,
    measured_label: str,
    trace_input_paths: tuple[str, ...],
) -> dict[str, object]:
    cached_decision = _promotion_decision(cached)
    measured_decision = _promotion_decision(measured)
    measured_family = _family_separation(measured)
    measured_oracle = _oracle_headroom(measured)
    strict_family = _strict_family_pass(measured_family)
    reproduction_pass = bool(
        measured_decision["promotion_pass"]
        and measured_decision["rdu_is_best_compressed"]
        and measured_decision["margin_vs_rkv_like"] >= PROMOTION_MARGIN
        and measured_decision["margin_vs_thin_kv_like"] >= PROMOTION_MARGIN
        and strict_family["same_family_positive"]
        and strict_family["cross_family_positive"]
    )
    return {
        "status": (
            "REPRODUCED on independent saved-trace no-retuning surface; rdu_topk remains best compressed and keeps strict same-family/cross-family separation."
            if reproduction_pass
            else "NOT REPRODUCED on independent saved-trace no-retuning surface; inspect same-family/cross-family decision details."
        ),
        "method_branch": RDU_POLICY_NAME,
        "diagnostic_type": "measured_no_retuning_independent_trace_slice_against_cached_frozen_gate",
        "cached_label": "cached_promoted_gate",
        "measured_label": measured_label,
        "trace_input_paths": list(trace_input_paths),
        "cached_baseline": _compact_result(cached),
        "measured_reproduction": _compact_result(measured),
        "cached_decision": cached_decision,
        "measured_decision": measured_decision,
        "cached_vs_measured": _cached_vs_measured(cached, measured),
        "measured_family_separation": measured_family,
        "measured_failure_decomposition": _trace_failure_decomposition(measured),
        "strict_family_pass": strict_family,
        "measured_oracle_headroom": measured_oracle,
        "reproduction_pass": reproduction_pass,
    }


def _write_markdown(report: dict[str, object], output_path: Path) -> None:
    measured = report["measured_decision"]
    cached = report["cached_decision"]
    oracle = report["measured_oracle_headroom"]
    family = report["measured_family_separation"]
    drift = report["cached_vs_measured"]["policy_nll_delta_measured_minus_cached"]
    lines = [
        "# ThoughtFlow-FP8 RDU Independent-Trace Reproduction Check",
        "",
        f"Status: **{report['status']}**",
        "",
        f"- diagnostic type: `{report['diagnostic_type']}`",
        f"- cached label: `{report['cached_label']}`",
        f"- measured label: `{report['measured_label']}`",
        f"- method branch: `{report['method_branch']}`",
        f"- scored traces: {report['measured_reproduction']['n_scored_traces']}",
        f"- keep fraction: {float(report['measured_reproduction']['keep_fraction']):.2f}",
        f"- max length: {report['measured_reproduction']['max_length']}",
        f"- continuation tokens: {report['measured_reproduction']['continuation_tokens']}",
        "",
        "This check reruns the frozen sparse-cache probe with the same `rdu_topk` rule and no policy retuning. Only the saved trace inputs change.",
        "",
        "## Trace Inputs",
        "",
    ]
    lines.extend(f"- `{path}`" for path in report["trace_input_paths"])
    lines.extend(
        [
            "",
            "## Cached vs Measured Decision",
            "",
            "| Label | RDU NLL | Best compressed | Margin vs R-KV | Paired vs R-KV | Margin vs ThinKV | Paired vs ThinKV | Promotion |",
            "|---|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for label, decision in ((report["cached_label"], cached), (report["measured_label"], measured)):
        lines.append(
            "| {label} | {rdu_nll:.3f} | {best} | {margin_rkv:+.3f} | {pair_rkv} | {margin_thin:+.3f} | {pair_thin} | {promotion} |".format(
                label=label,
                rdu_nll=decision["rdu_nll"],
                best=decision["best_compressed_policy"],
                margin_rkv=decision["margin_vs_rkv_like"],
                pair_rkv=_fmt_ci(decision["paired_delta_vs_rkv_like"], "mean_delta_nll_minus_rkv_like"),
                margin_thin=decision["margin_vs_thin_kv_like"],
                pair_thin=_fmt_ci(decision["paired_delta_vs_thin_kv_like"], "mean_delta_nll_minus_thin_kv_like"),
                promotion="pass" if decision["promotion_pass"] else "fail",
            )
        )
    lines.extend(
        [
            "",
            "## Measured Policy Table",
            "",
            "| Policy | NLL | Delta measured-cached |",
            "|---|---:|---:|",
        ]
    )
    for policy, metrics in sorted(report["measured_reproduction"]["summary"].items(), key=lambda item: item[1]["nll"]):
        lines.append(
            "| {policy} | {nll:.3f} | {drift:+.3f} |".format(
                policy=policy,
                nll=metrics["nll"],
                drift=drift.get(policy, 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Strict Separation",
            "",
            "Positive margins mean the measured row is worse than `rdu_topk`.",
            "",
            "| Family | Policy | Margin NLL vs RDU |",
            "|---|---|---:|",
        ]
    )
    for policy, margin in family["same_family_margin_nll_vs_rdu"].items():
        lines.append(f"| stopped ThoughtFlow family | {policy} | {margin:+.3f} |")
    for policy, margin in family["cross_family_margin_nll_vs_rdu"].items():
        lines.append(f"| cross-family baseline | {policy} | {margin:+.3f} |")
    strict = report["strict_family_pass"]
    lines.extend(
        [
            "",
            "## Oracle And Headroom",
            "",
            f"- measured per-trace compressed oracle NLL: {oracle['per_trace_oracle_nll']:.3f}",
            f"- measured `rdu_topk` NLL: {oracle['rdu_topk_nll']:.3f}",
            f"- measured full-cache NLL: {oracle['full_cache_nll']:.3f}",
            f"- `rdu_topk` gap to per-trace compressed oracle: {oracle['rdu_gap_to_per_trace_oracle']:.3f}",
            f"- per-trace compressed oracle gap to full cache: {oracle['oracle_gap_to_full_cache']:.3f}",
            f"- `rdu_topk` oracle hit rate: {oracle['rdu_oracle_hit_rate']:.3f}",
            "",
            "## Failure Decomposition",
            "",
            "Rows report `rdu_topk` minus the named baseline; positive is worse for `rdu_topk`. Buckets are tertiles computed on the measured independent surface.",
            "",
            "| Group | n | RDU-RKV NLL | RDU-Thin NLL | RDU-oracle gap | RDU beats RKV | RDU oracle hit |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group_name, metrics in report["measured_failure_decomposition"]["group_summaries"].items():
        lines.append(
            "| {group} | {n} | {rkv:+.3f} | {thin:+.3f} | {oracle:+.3f} | {win:.3f} | {hit:.3f} |".format(
                group=group_name,
                n=metrics["n"],
                rkv=metrics["mean_rdu_minus_rkv"],
                thin=metrics["mean_rdu_minus_thin"],
                oracle=metrics["mean_rdu_oracle_gap"],
                win=metrics["rdu_beats_rkv_rate"],
                hit=metrics["rdu_oracle_hit_rate"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- same-family positive separation: {strict['same_family_positive']} (min margin {strict['same_family_min_margin']:+.3f})",
            f"- cross-family positive separation: {strict['cross_family_positive']} (min margin {strict['cross_family_min_margin']:+.3f})",
            "",
            "This is an independent saved-trace reproduction surface, not a policy-tuning surface.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_input_paths(paths: list[Path]) -> list[Path]:
    selected = paths or list(DEFAULT_TRACE_INPUTS)
    return [path if path.is_absolute() else ROOT / path for path in selected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached-input", type=Path, default=DEFAULT_CACHED_INPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    parser.add_argument("--input-jsonl", action="append", type=Path, default=[])
    parser.add_argument("--measured-label", default="measured_independent_chat_svamp96")
    parser.add_argument("--model-name", default="distilgpt2")
    parser.add_argument("--keep-fraction", type=float, default=0.20)
    parser.add_argument("--max-traces", type=int, default=96)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--continuation-tokens", type=int, default=24)
    args = parser.parse_args()

    trace_paths = _resolve_input_paths(args.input_jsonl)
    cached = json.loads(args.cached_input.read_text(encoding="utf-8"))
    measured = run_frozen_probe(
        args.model_name,
        args.keep_fraction,
        args.max_traces,
        args.max_length,
        args.continuation_tokens,
        trace_paths=trace_paths,
    )
    report = build_report(
        cached,
        measured,
        measured_label=args.measured_label,
        trace_input_paths=tuple(_path_label(path) for path in trace_paths),
    )
    args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_markdown(report, args.md_output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "measured_decision": report["measured_decision"],
                "strict_family_pass": report["strict_family_pass"],
                "measured_oracle_headroom": report["measured_oracle_headroom"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
