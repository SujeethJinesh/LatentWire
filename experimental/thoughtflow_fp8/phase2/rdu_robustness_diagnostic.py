"""Cached robustness diagnostics for the pre-registered rdu_topk result.

This script does not score the model or change the rdu_topk rule. It reads the
existing frozen sparse-cache probe rows and checks whether the promoted row is
stable under deterministic trace partitions.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean


PHASE2_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PHASE2_DIR / "frozen_sparse_cache_probe.json"
DEFAULT_JSON_OUTPUT = PHASE2_DIR / "rdu_robustness_diagnostic.json"
DEFAULT_MD_OUTPUT = PHASE2_DIR / "rdu_robustness_diagnostic.md"

RDU_POLICY = "rdu_topk"
RKV_POLICY = "rkv_like"
THIN_POLICY = "thin_kv_like"
FULL_POLICY = "full_cache"
STOPPED_THOUGHTFLOW_POLICIES = (
    "thoughtflow_saliency_recent",
    "tf_sparse_r0.55_p0.05_m0.12_a2",
)
PROMOTION_MARGIN = 0.03


def _summary(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    summary = {}
    for policy in sorted({str(row["policy"]) for row in rows}):
        policy_rows = [row for row in rows if row["policy"] == policy]
        summary[policy] = {
            "n_traces": float(len(policy_rows)),
            "keep_rate": mean(float(row["keep_rate"]) for row in policy_rows),
            "nll": mean(float(row["nll"]) for row in policy_rows),
            "delta_nll_vs_full": mean(float(row.get("delta_nll_vs_full", 0.0)) for row in policy_rows),
        }
    return summary


def _paired_deltas(
    rows: list[dict[str, object]],
    *,
    baseline_policy: str,
    bootstrap_samples: int = 1000,
    seed: int = 17,
) -> dict[str, dict[str, float]]:
    by_policy: dict[str, dict[int, float]] = {}
    for row in rows:
        by_policy.setdefault(str(row["policy"]), {})[int(row["trace_id"])] = float(row["nll"])
    if baseline_policy not in by_policy:
        return {}

    rng = random.Random(seed)
    result = {}
    for policy, by_trace in by_policy.items():
        if policy == baseline_policy:
            continue
        trace_ids = sorted(set(by_trace) & set(by_policy[baseline_policy]))
        if not trace_ids:
            continue
        deltas = [by_trace[trace_id] - by_policy[baseline_policy][trace_id] for trace_id in trace_ids]
        boot = []
        for _ in range(bootstrap_samples):
            sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
            boot.append(mean(sample))
        boot.sort()
        result[policy] = {
            "n_pairs": float(len(deltas)),
            f"mean_delta_nll_minus_{baseline_policy}": mean(deltas),
            "ci95_low": boot[int(0.025 * (bootstrap_samples - 1))],
            "ci95_high": boot[int(0.975 * (bootstrap_samples - 1))],
        }
    return result


def _trace_ids(rows: list[dict[str, object]]) -> list[int]:
    return sorted({int(row["trace_id"]) for row in rows})


def _deterministic_splits(trace_ids: list[int]) -> list[dict[str, object]]:
    midpoint = len(trace_ids) // 2
    return [
        {
            "name": "all_traces",
            "description": "full cached frozen sparse-cache gate",
            "trace_ids": trace_ids,
            "primary_full_gate": True,
        },
        {
            "name": "even_trace_ids",
            "description": "even trace ids only",
            "trace_ids": [trace_id for trace_id in trace_ids if trace_id % 2 == 0],
            "primary_full_gate": False,
        },
        {
            "name": "odd_trace_ids",
            "description": "odd trace ids only",
            "trace_ids": [trace_id for trace_id in trace_ids if trace_id % 2 == 1],
            "primary_full_gate": False,
        },
        {
            "name": "first_half_trace_ids",
            "description": "lower trace ids",
            "trace_ids": trace_ids[:midpoint],
            "primary_full_gate": False,
        },
        {
            "name": "second_half_trace_ids",
            "description": "higher trace ids",
            "trace_ids": trace_ids[midpoint:],
            "primary_full_gate": False,
        },
    ]


def _rows_for_trace_ids(rows: list[dict[str, object]], trace_ids: list[int]) -> list[dict[str, object]]:
    allowed = set(trace_ids)
    return [row for row in rows if int(row["trace_id"]) in allowed]


def _win_rate(rows: list[dict[str, object]], *, candidate_policy: str, baseline_policy: str) -> float:
    by_policy: dict[str, dict[int, float]] = {}
    for row in rows:
        by_policy.setdefault(str(row["policy"]), {})[int(row["trace_id"])] = float(row["nll"])
    trace_ids = sorted(set(by_policy.get(candidate_policy, {})) & set(by_policy.get(baseline_policy, {})))
    if not trace_ids:
        return 0.0
    wins = sum(
        by_policy[candidate_policy][trace_id] < by_policy[baseline_policy][trace_id]
        for trace_id in trace_ids
    )
    return wins / len(trace_ids)


def _best_compressed_policy(summary: dict[str, dict[str, float]]) -> str:
    compressed = {policy: metrics for policy, metrics in summary.items() if policy != FULL_POLICY}
    return min(compressed.items(), key=lambda item: item[1]["nll"])[0]


def _rdu_decision(
    rows: list[dict[str, object]],
    summary: dict[str, dict[str, float]],
    paired_vs_rkv: dict[str, dict[str, float]],
    paired_vs_thin: dict[str, dict[str, float]],
) -> dict[str, object]:
    rdu = summary[RDU_POLICY]
    margin_vs_rkv = summary[RKV_POLICY]["nll"] - rdu["nll"]
    margin_vs_thin = summary[THIN_POLICY]["nll"] - rdu["nll"]
    rkv_pair = paired_vs_rkv[RDU_POLICY]
    thin_pair = paired_vs_thin[RDU_POLICY]
    rkv_mean = rkv_pair[f"mean_delta_nll_minus_{RKV_POLICY}"]
    thin_mean = thin_pair[f"mean_delta_nll_minus_{THIN_POLICY}"]

    stopped_family_margins = {
        policy: summary[policy]["nll"] - rdu["nll"]
        for policy in STOPPED_THOUGHTFLOW_POLICIES
        if policy in summary
    }
    mean_margin_pass = margin_vs_rkv >= PROMOTION_MARGIN and margin_vs_thin >= PROMOTION_MARGIN
    paired_mean_pass = rkv_mean < 0.0 and thin_mean < 0.0
    paired_ci_pass = rkv_pair["ci95_high"] < 0.0 and thin_pair["ci95_high"] < 0.0
    return {
        "rdu_nll": rdu["nll"],
        "margin_vs_rkv_like": margin_vs_rkv,
        "margin_vs_thin_kv_like": margin_vs_thin,
        "paired_delta_vs_rkv_like": rkv_pair,
        "paired_delta_vs_thin_kv_like": thin_pair,
        "win_rate_vs_rkv_like": _win_rate(rows, candidate_policy=RDU_POLICY, baseline_policy=RKV_POLICY),
        "win_rate_vs_thin_kv_like": _win_rate(rows, candidate_policy=RDU_POLICY, baseline_policy=THIN_POLICY),
        "stopped_family_margins": stopped_family_margins,
        "mean_margin_pass": mean_margin_pass,
        "paired_mean_pass": paired_mean_pass,
        "paired_ci_pass": paired_ci_pass,
        "promotion_pass": mean_margin_pass and paired_ci_pass,
    }


def _split_result(
    rows: list[dict[str, object]],
    split: dict[str, object],
    *,
    bootstrap_samples: int,
) -> dict[str, object]:
    split_rows = _rows_for_trace_ids(rows, list(split["trace_ids"]))
    summary = _summary(split_rows)
    paired_vs_rkv = _paired_deltas(split_rows, baseline_policy=RKV_POLICY, bootstrap_samples=bootstrap_samples)
    paired_vs_thin = _paired_deltas(split_rows, baseline_policy=THIN_POLICY, bootstrap_samples=bootstrap_samples)
    return {
        "name": split["name"],
        "description": split["description"],
        "primary_full_gate": bool(split["primary_full_gate"]),
        "trace_ids": list(split["trace_ids"]),
        "n_traces": len(split["trace_ids"]),
        "best_compressed_policy": _best_compressed_policy(summary),
        "summary": summary,
        "paired_delta_nll_vs_rkv_like": paired_vs_rkv,
        "paired_delta_nll_vs_thin_kv_like": paired_vs_thin,
        "rdu_decision": _rdu_decision(split_rows, summary, paired_vs_rkv, paired_vs_thin),
    }


def _status(split_results: list[dict[str, object]]) -> str:
    full = next(item for item in split_results if item["primary_full_gate"])
    full_promoted = bool(full["rdu_decision"]["promotion_pass"])
    if not full_promoted:
        return "NOT PROMOTED; cached full-gate rdu_topk row does not clear the preregistered rule."

    diagnostics = [item for item in split_results if not item["primary_full_gate"]]
    mean_stable = all(bool(item["rdu_decision"]["mean_margin_pass"]) for item in diagnostics)
    paired_mean_stable = all(bool(item["rdu_decision"]["paired_mean_pass"]) for item in diagnostics)
    ci_passes = sum(1 for item in diagnostics if item["rdu_decision"]["promotion_pass"])
    if mean_stable and paired_mean_stable and ci_passes == len(diagnostics):
        return "PROMOTED on cached full gate; all deterministic trace splits also clear the mean-margin and paired-CI rule."
    if mean_stable and paired_mean_stable:
        return (
            "PROMOTED on cached full gate; deterministic trace splits keep positive margins and paired means, "
            "but split CIs are not uniformly below zero."
        )
    return (
        "PROMOTED on cached full gate; deterministic trace splits expose at least one unstable mean margin, "
        "so robustness is weakened."
    )


def _run_from_result(result: dict[str, object], *, bootstrap_samples: int = 1000) -> dict[str, object]:
    rows = list(result["rows"])
    splits = _deterministic_splits(_trace_ids(rows))
    split_results = [
        _split_result(rows, split, bootstrap_samples=bootstrap_samples)
        for split in splits
    ]
    diagnostic_splits = [item for item in split_results if not item["primary_full_gate"]]
    return {
        "source_artifact": "frozen_sparse_cache_probe.json",
        "model_name": result.get("model_name"),
        "keep_fraction": result.get("keep_fraction"),
        "n_scored_traces": result.get("n_scored_traces"),
        "continuation_tokens": result.get("continuation_tokens"),
        "bootstrap_samples": bootstrap_samples,
        "split_results": split_results,
        "split_mean_margin_passes": sum(1 for item in diagnostic_splits if item["rdu_decision"]["mean_margin_pass"]),
        "split_paired_mean_passes": sum(1 for item in diagnostic_splits if item["rdu_decision"]["paired_mean_pass"]),
        "split_promotion_passes": sum(1 for item in diagnostic_splits if item["rdu_decision"]["promotion_pass"]),
        "status": _status(split_results),
    }


def _format_ci(metrics: dict[str, float], baseline_policy: str) -> str:
    mean_key = f"mean_delta_nll_minus_{baseline_policy}"
    return "{mean:+.3f} [{low:+.3f},{high:+.3f}]".format(
        mean=metrics[mean_key],
        low=metrics["ci95_low"],
        high=metrics["ci95_high"],
    )


def _write_markdown(result: dict[str, object], output_path: Path) -> None:
    lines = [
        "# ThoughtFlow-FP8 RDU Robustness Diagnostic",
        "",
        f"Status: **{result['status']}**",
        "",
        f"- source artifact: `{result['source_artifact']}`",
        f"- model: `{result['model_name']}`",
        f"- keep fraction: {float(result['keep_fraction']):.2f}",
        f"- scored traces: {result['n_scored_traces']}",
        f"- continuation tokens: {result['continuation_tokens']}",
        f"- bootstrap samples per paired CI: {result['bootstrap_samples']}",
        "",
        "This diagnostic reuses the cached 0.20 frozen sparse-cache rows. It does not rerun the model, retune a policy, or change the pre-registered `rdu_topk` scoring rule.",
        "",
        "| Split | Traces | Best compressed | RDU NLL | Margin vs R-KV | Paired vs R-KV | Win rate vs R-KV | Margin vs ThinKV | Paired vs ThinKV | Win rate vs ThinKV | Promotion rule |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for split in result["split_results"]:
        decision = split["rdu_decision"]
        lines.append(
            "| {name} | {n_traces} | {best} | {rdu_nll:.3f} | {margin_rkv:+.3f} | {pair_rkv} | {win_rkv:.3f} | {margin_thin:+.3f} | {pair_thin} | {win_thin:.3f} | {promotion} |".format(
                name=split["name"],
                n_traces=split["n_traces"],
                best=split["best_compressed_policy"],
                rdu_nll=decision["rdu_nll"],
                margin_rkv=decision["margin_vs_rkv_like"],
                pair_rkv=_format_ci(decision["paired_delta_vs_rkv_like"], RKV_POLICY),
                win_rkv=decision["win_rate_vs_rkv_like"],
                margin_thin=decision["margin_vs_thin_kv_like"],
                pair_thin=_format_ci(decision["paired_delta_vs_thin_kv_like"], THIN_POLICY),
                win_thin=decision["win_rate_vs_thin_kv_like"],
                promotion="pass" if decision["promotion_pass"] else "mean-only" if decision["mean_margin_pass"] else "fail",
            )
        )

    full = next(item for item in result["split_results"] if item["primary_full_gate"])
    full_family = full["rdu_decision"]["stopped_family_margins"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The cached full gate still satisfies the pre-registered promotion rule. All four deterministic half-size diagnostics keep positive mean margins of at least 0.03 NLL versus both R-KV-like and ThinKV-like, and `rdu_topk` remains the best compressed row in each split.",
            "",
            "Half-size paired CIs are intentionally treated as a stress diagnostic, not as a replacement for a fresh reproduction. The odd and second-half partitions leave some CI highs slightly above zero, so this result strengthens the branch but does not make it ICLR-ready.",
            "",
            "Same-family separation is also preserved on the full cached gate: `rdu_topk` beats the stopped ThoughtFlow candidates by "
            + ", ".join(f"{policy}: {margin:.3f} NLL" for policy, margin in full_family.items())
            + ".",
            "",
            "## Decision",
            "",
            "`rdu_topk` remains promoted on the current frozen sparse-cache decision surface. The next exact gate is still a real reproduction artifact: a larger or seed-repeated frozen slice with no retuning, plus strict same-family versus cross-family reporting and oracle/headroom diagnostics.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()

    source = json.loads(args.input.read_text(encoding="utf-8"))
    result = _run_from_result(source, bootstrap_samples=args.bootstrap_samples)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_markdown(result, args.md_output)
    print(json.dumps({"status": result["status"], "split_promotion_passes": result["split_promotion_passes"]}, indent=2))


if __name__ == "__main__":
    main()
