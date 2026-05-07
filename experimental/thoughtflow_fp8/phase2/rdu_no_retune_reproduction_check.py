"""No-retuning reproduction check for the first-surface rdu_topk branch.

This script runs the frozen sparse-cache probe unchanged, stores the newly
measured result in a separate artifact, and compares it with the cached
first-surface gate. It does not overwrite frozen_sparse_cache_probe.json and does not change
the rdu_topk scoring rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .frozen_sparse_cache_probe import DISTILGPT2_REVISION, RDU_POLICY_NAME, _run as run_frozen_probe
except ImportError:  # pragma: no cover - supports direct script execution.
    from frozen_sparse_cache_probe import DISTILGPT2_REVISION, RDU_POLICY_NAME, _run as run_frozen_probe


PHASE2_DIR = Path(__file__).resolve().parent
DEFAULT_CACHED_INPUT = PHASE2_DIR / "frozen_sparse_cache_probe.json"
DEFAULT_JSON_OUTPUT = PHASE2_DIR / "rdu_no_retune_reproduction_check.json"
DEFAULT_MD_OUTPUT = PHASE2_DIR / "rdu_no_retune_reproduction_check.md"

RKV_POLICY = "rkv_like"
THIN_POLICY = "thin_kv_like"
FULL_POLICY = "full_cache"
LONGFLOW_POLICY = "longflow_like"
STOPPED_THOUGHTFLOW_POLICIES = (
    "thoughtflow_saliency_recent",
    "tf_sparse_r0.55_p0.05_m0.12_a2",
)
PROMOTION_MARGIN = 0.03


def _policy_nll(result: dict[str, object], policy: str) -> float:
    return float(result["summary"][policy]["nll"])


def _best_compressed(result: dict[str, object]) -> tuple[str, float]:
    compressed = {
        str(policy): float(metrics["nll"])
        for policy, metrics in result["summary"].items()
        if policy != FULL_POLICY
    }
    return min(compressed.items(), key=lambda item: item[1])


def _paired_ci(result: dict[str, object], *, policy: str, baseline: str) -> dict[str, float]:
    key = f"paired_delta_nll_vs_{baseline}"
    if baseline == RKV_POLICY:
        key = "paired_delta_nll_vs_rkv_like"
    elif baseline == THIN_POLICY:
        key = "paired_delta_nll_vs_thin_kv_like"
    return {
        metric: float(value)
        for metric, value in result[key][policy].items()
    }


def _promotion_decision(result: dict[str, object]) -> dict[str, object]:
    rdu_nll = _policy_nll(result, RDU_POLICY_NAME)
    margin_vs_rkv = _policy_nll(result, RKV_POLICY) - rdu_nll
    margin_vs_thin = _policy_nll(result, THIN_POLICY) - rdu_nll
    paired_vs_rkv = _paired_ci(result, policy=RDU_POLICY_NAME, baseline=RKV_POLICY)
    paired_vs_thin = _paired_ci(result, policy=RDU_POLICY_NAME, baseline=THIN_POLICY)
    best_name, best_nll = _best_compressed(result)
    return {
        "rdu_nll": rdu_nll,
        "best_compressed_policy": best_name,
        "best_compressed_nll": best_nll,
        "margin_vs_rkv_like": margin_vs_rkv,
        "margin_vs_thin_kv_like": margin_vs_thin,
        "paired_delta_vs_rkv_like": paired_vs_rkv,
        "paired_delta_vs_thin_kv_like": paired_vs_thin,
        "mean_margin_pass": margin_vs_rkv >= PROMOTION_MARGIN and margin_vs_thin >= PROMOTION_MARGIN,
        "paired_ci_pass": paired_vs_rkv["ci95_high"] < 0.0 and paired_vs_thin["ci95_high"] < 0.0,
        "promotion_pass": (
            margin_vs_rkv >= PROMOTION_MARGIN
            and margin_vs_thin >= PROMOTION_MARGIN
            and paired_vs_rkv["ci95_high"] < 0.0
            and paired_vs_thin["ci95_high"] < 0.0
        ),
        "rdu_is_best_compressed": best_name == RDU_POLICY_NAME,
    }


def _paired_trace_values(result: dict[str, object], policy: str) -> dict[int, float]:
    return {
        int(row["trace_id"]): float(row["nll"])
        for row in result["rows"]
        if row["policy"] == policy
    }


def _win_rate(result: dict[str, object], *, policy: str, baseline: str) -> float:
    policy_rows = _paired_trace_values(result, policy)
    baseline_rows = _paired_trace_values(result, baseline)
    trace_ids = sorted(set(policy_rows) & set(baseline_rows))
    if not trace_ids:
        return 0.0
    return sum(policy_rows[trace_id] < baseline_rows[trace_id] for trace_id in trace_ids) / len(trace_ids)


def _oracle_headroom(result: dict[str, object]) -> dict[str, object]:
    by_policy = {
        str(policy): _paired_trace_values(result, str(policy))
        for policy in result["summary"]
        if policy != FULL_POLICY
    }
    trace_ids = sorted(set.intersection(*(set(rows) for rows in by_policy.values())))
    oracle_values = [min(rows[trace_id] for rows in by_policy.values()) for trace_id in trace_ids]
    rdu_values = [_paired_trace_values(result, RDU_POLICY_NAME)[trace_id] for trace_id in trace_ids]
    full_values = _paired_trace_values(result, FULL_POLICY)
    full_mean = sum(full_values[trace_id] for trace_id in trace_ids) / len(trace_ids)
    oracle_mean = sum(oracle_values) / len(oracle_values)
    rdu_mean = sum(rdu_values) / len(rdu_values)
    rdu_oracle_hits = sum(rdu == oracle for rdu, oracle in zip(rdu_values, oracle_values))
    return {
        "oracle_policy_set": sorted(by_policy),
        "n_traces": len(trace_ids),
        "per_trace_oracle_nll": oracle_mean,
        "rdu_topk_nll": rdu_mean,
        "full_cache_nll": full_mean,
        "rdu_gap_to_per_trace_oracle": rdu_mean - oracle_mean,
        "oracle_gap_to_full_cache": oracle_mean - full_mean,
        "rdu_gap_to_full_cache": rdu_mean - full_mean,
        "rdu_oracle_hit_rate": rdu_oracle_hits / len(trace_ids) if trace_ids else 0.0,
    }


def _family_separation(result: dict[str, object]) -> dict[str, object]:
    rdu_nll = _policy_nll(result, RDU_POLICY_NAME)
    same_family = {
        policy: _policy_nll(result, policy) - rdu_nll
        for policy in STOPPED_THOUGHTFLOW_POLICIES
        if policy in result["summary"]
    }
    cross_family = {
        policy: _policy_nll(result, policy) - rdu_nll
        for policy in (RKV_POLICY, THIN_POLICY, LONGFLOW_POLICY)
        if policy in result["summary"]
    }
    return {
        "same_family_margin_nll_vs_rdu": same_family,
        "cross_family_margin_nll_vs_rdu": cross_family,
        "win_rate_vs_rkv_like": _win_rate(result, policy=RDU_POLICY_NAME, baseline=RKV_POLICY),
        "win_rate_vs_thin_kv_like": _win_rate(result, policy=RDU_POLICY_NAME, baseline=THIN_POLICY),
    }


def _cached_vs_measured(cached: dict[str, object], measured: dict[str, object]) -> dict[str, object]:
    policies = sorted(set(cached["summary"]) & set(measured["summary"]))
    return {
        "policy_nll_delta_measured_minus_cached": {
            policy: _policy_nll(measured, policy) - _policy_nll(cached, policy)
            for policy in policies
        },
        "rdu_margin_delta_vs_rkv_like": (
            _promotion_decision(measured)["margin_vs_rkv_like"]
            - _promotion_decision(cached)["margin_vs_rkv_like"]
        ),
        "rdu_margin_delta_vs_thin_kv_like": (
            _promotion_decision(measured)["margin_vs_thin_kv_like"]
            - _promotion_decision(cached)["margin_vs_thin_kv_like"]
        ),
    }


def _compact_result(result: dict[str, object]) -> dict[str, object]:
    return {
        "model_name": result["model_name"],
        "model_revision": result.get("model_revision", ""),
        "tokenizer_revision": result.get("tokenizer_revision", ""),
        "input_paths": result.get("input_paths", []),
        "keep_fraction": result["keep_fraction"],
        "max_traces": result["max_traces"],
        "max_length": result["max_length"],
        "continuation_tokens": result["continuation_tokens"],
        "n_scored_traces": result["n_scored_traces"],
        "summary": result["summary"],
        "paired_delta_nll_vs_rkv_like": result["paired_delta_nll_vs_rkv_like"],
        "paired_delta_nll_vs_thin_kv_like": result["paired_delta_nll_vs_thin_kv_like"],
        "status": result["status"],
    }


def build_report(cached: dict[str, object], measured: dict[str, object]) -> dict[str, object]:
    cached_decision = _promotion_decision(cached)
    measured_decision = _promotion_decision(measured)
    measured_oracle = _oracle_headroom(measured)
    measured_family = _family_separation(measured)
    reproduction_pass = bool(
        measured_decision["promotion_pass"]
        and measured_decision["rdu_is_best_compressed"]
        and measured["n_scored_traces"] == cached["n_scored_traces"]
    )
    return {
        "status": (
            "REPRODUCED on measured no-retuning rerun; rdu_topk remains best compressed and clears the preregistered rule."
            if reproduction_pass
            else "NOT REPRODUCED on measured no-retuning rerun; inspect measured decision details."
        ),
        "method_branch": RDU_POLICY_NAME,
        "diagnostic_type": "measured_no_retuning_rerun_against_cached_frozen_gate",
        "cached_label": "cached_promoted_gate",
        "measured_label": "measured_reproduction_rerun",
        "cached_baseline": _compact_result(cached),
        "measured_reproduction": _compact_result(measured),
        "cached_decision": cached_decision,
        "measured_decision": measured_decision,
        "cached_vs_measured": _cached_vs_measured(cached, measured),
        "measured_family_separation": measured_family,
        "measured_oracle_headroom": measured_oracle,
        "reproduction_pass": reproduction_pass,
    }


def _fmt_ci(metrics: dict[str, float], key: str) -> str:
    return "{mean:+.3f} [{low:+.3f},{high:+.3f}]".format(
        mean=metrics[key],
        low=metrics["ci95_low"],
        high=metrics["ci95_high"],
    )


def _write_markdown(report: dict[str, object], output_path: Path) -> None:
    measured = report["measured_decision"]
    cached = report["cached_decision"]
    oracle = report["measured_oracle_headroom"]
    family = report["measured_family_separation"]
    drift = report["cached_vs_measured"]["policy_nll_delta_measured_minus_cached"]
    lines = [
        "# ThoughtFlow-FP8 RDU No-Retuning Reproduction Check",
        "",
        f"Status: **{report['status']}**",
        "",
        f"- diagnostic type: `{report['diagnostic_type']}`",
        f"- cached label: `{report['cached_label']}`",
        f"- measured label: `{report['measured_label']}`",
        f"- method branch: `{report['method_branch']}`",
        f"- scored traces: {report['measured_reproduction']['n_scored_traces']}",
        f"- keep fraction: {float(report['measured_reproduction']['keep_fraction']):.2f}",
        f"- continuation tokens: {report['measured_reproduction']['continuation_tokens']}",
        "",
        "This reruns the frozen sparse-cache probe with the existing `rdu_topk` rule and writes a separate measured artifact. It does not retune policy parameters and does not overwrite the cached first-surface frozen gate.",
        "",
        "## Cached vs Measured Decision",
        "",
        "| Label | RDU NLL | Best compressed | Margin vs R-KV | Paired vs R-KV | Margin vs ThinKV | Paired vs ThinKV | Promotion |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for label, decision in (("cached_promoted_gate", cached), ("measured_reproduction_rerun", measured)):
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
    measured_summary = report["measured_reproduction"]["summary"]
    for policy, metrics in sorted(measured_summary.items(), key=lambda item: item[1]["nll"]):
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
            "## Decision",
            "",
            "This is a measured reproduction-style check, not a new tuning surface. The next gate remains a larger or independently seeded frozen slice with the same strict reporting.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached-input", type=Path, default=DEFAULT_CACHED_INPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    parser.add_argument("--model-name", default="distilgpt2")
    parser.add_argument("--model-revision", default=DISTILGPT2_REVISION)
    parser.add_argument("--keep-fraction", type=float, default=0.20)
    parser.add_argument("--max-traces", type=int, default=74)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--continuation-tokens", type=int, default=24)
    args = parser.parse_args()

    cached = json.loads(args.cached_input.read_text(encoding="utf-8"))
    measured = run_frozen_probe(
        args.model_name,
        args.keep_fraction,
        args.max_traces,
        args.max_length,
        args.continuation_tokens,
        model_revision=args.model_revision,
    )
    report = build_report(cached, measured)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_markdown(report, args.md_output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "measured_decision": report["measured_decision"],
                "measured_oracle_headroom": report["measured_oracle_headroom"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
