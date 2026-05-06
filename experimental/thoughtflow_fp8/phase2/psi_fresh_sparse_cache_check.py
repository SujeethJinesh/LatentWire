"""One-shot prefix-surprisal utility check for ThoughtFlow-FP8.

This script implements the pre-registered `psi_topk` utility and evaluates it
once on a saved-trace surface separate from the historical RDU gates. It does not
tune, sweep, or select policy variants.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from .frozen_sparse_cache_probe import (
        FROZEN_SPARSE_POLICY_NAME,
        _frozen_policies,
        _load_model_for_prefill_attentions,
        _score_continuation_with_cache,
    )
    from .kv_drop_quality_probe import _paired_deltas, _prepare_rows, _summary
    from .run_real_trace_retention import OUT_DIR, ROOT
except ImportError:  # pragma: no cover - supports direct script execution.
    from frozen_sparse_cache_probe import (
        FROZEN_SPARSE_POLICY_NAME,
        _frozen_policies,
        _load_model_for_prefill_attentions,
        _score_continuation_with_cache,
    )
    from kv_drop_quality_probe import _paired_deltas, _prepare_rows, _summary
    from run_real_trace_retention import OUT_DIR, ROOT


PSI_POLICY_NAME = "psi_topk"
DEFAULT_JSON_OUTPUT = OUT_DIR / "psi_fresh_sparse_cache_check.json"
DEFAULT_MD_OUTPUT = OUT_DIR / "psi_fresh_sparse_cache_check.md"
DEFAULT_TRACE_INPUTS = (
    ROOT / "results/c2c_gsm70_20260418/qwen_gsm70_c2c.jsonl",
)
PROMOTION_MARGIN = 0.03


def _prefix_surprisal_scores(logits: torch.Tensor, prefix_ids: list[int]) -> list[dict[str, float]]:
    """Return model self-surprisal for each prefix token position."""

    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(f"expected logits with shape [1, seq, vocab], got {tuple(logits.shape)}")
    if logits.shape[1] != len(prefix_ids):
        raise ValueError(f"logit sequence length {logits.shape[1]} does not match prefix length {len(prefix_ids)}")
    scores = [{"index": 0, "surprisal": 0.0}]
    if len(prefix_ids) == 1:
        return scores
    log_probs = F.log_softmax(logits[0, :-1].float(), dim=-1)
    targets = torch.tensor(prefix_ids[1:], dtype=torch.long)
    token_surprisals = -log_probs[torch.arange(len(targets)), targets]
    scores.extend(
        {"index": idx, "surprisal": float(value.item())}
        for idx, value in enumerate(token_surprisals, start=1)
    )
    return scores


def _psi_topk_from_logits(logits: torch.Tensor, prefix_ids: list[int], budget: int) -> tuple[set[int], list[dict[str, float]]]:
    scores = _prefix_surprisal_scores(logits, prefix_ids)
    budget = min(max(0, budget), len(scores))
    ranked = sorted(scores, key=lambda item: (-float(item["surprisal"]), int(item["index"])))
    return {int(item["index"]) for item in ranked[:budget]}, scores


def _psi_telemetry(scores: list[dict[str, float]], kept: set[int]) -> dict[str, float]:
    values = [float(item["surprisal"]) for item in scores]
    kept_values = [float(item["surprisal"]) for item in scores if int(item["index"]) in kept]
    nonzero = [value for value in values if value > 0.0]
    return {
        "mean_surprisal": sum(values) / len(values) if values else 0.0,
        "mean_kept_surprisal": sum(kept_values) / len(kept_values) if kept_values else 0.0,
        "max_surprisal": max(values) if values else 0.0,
        "nonzero_tokens": float(len(nonzero)),
    }


def _aggregate_psi_telemetry(per_trace: list[dict[str, object]]) -> dict[str, float]:
    if not per_trace:
        return {
            "mean_surprisal": 0.0,
            "mean_kept_surprisal": 0.0,
            "max_surprisal": 0.0,
            "nonzero_tokens": 0.0,
        }
    aggregate = {
        "mean_surprisal": 0.0,
        "mean_kept_surprisal": 0.0,
        "max_surprisal": 0.0,
        "nonzero_tokens": 0.0,
    }
    for item in per_trace:
        telemetry = item["telemetry"]
        aggregate["mean_surprisal"] += float(telemetry["mean_surprisal"])
        aggregate["mean_kept_surprisal"] += float(telemetry["mean_kept_surprisal"])
        aggregate["max_surprisal"] = max(aggregate["max_surprisal"], float(telemetry["max_surprisal"]))
        aggregate["nonzero_tokens"] += float(telemetry["nonzero_tokens"])
    aggregate["mean_surprisal"] /= len(per_trace)
    aggregate["mean_kept_surprisal"] /= len(per_trace)
    return aggregate


def _resolve_input_paths(paths: list[Path]) -> list[Path]:
    selected = paths or list(DEFAULT_TRACE_INPUTS)
    return [path if path.is_absolute() else ROOT / path for path in selected]


def _path_label(path: Path) -> str:
    if path.is_absolute() and path.is_relative_to(ROOT):
        return str(path.relative_to(ROOT))
    return str(path)


def _promotion_decision(
    summary: dict[str, dict[str, float]],
    paired_vs_rkv: dict[str, dict[str, float]],
    paired_vs_thin: dict[str, dict[str, float]],
) -> dict[str, object]:
    psi_nll = float(summary[PSI_POLICY_NAME]["nll"])
    compressed = {policy: metrics for policy, metrics in summary.items() if policy != "full_cache"}
    best_policy, best_metrics = min(compressed.items(), key=lambda item: item[1]["nll"])
    margin_vs_rkv = float(summary["rkv_like"]["nll"]) - psi_nll
    margin_vs_thin = float(summary["thin_kv_like"]["nll"]) - psi_nll
    rkv_ci_high = float(paired_vs_rkv.get(PSI_POLICY_NAME, {}).get("ci95_high", float("inf")))
    thin_ci_high = float(paired_vs_thin.get(PSI_POLICY_NAME, {}).get("ci95_high", float("inf")))
    promotion_pass = (
        margin_vs_rkv >= PROMOTION_MARGIN
        and margin_vs_thin >= PROMOTION_MARGIN
        and rkv_ci_high < 0.0
        and thin_ci_high < 0.0
        and best_policy == PSI_POLICY_NAME
    )
    return {
        "promotion_pass": promotion_pass,
        "psi_nll": psi_nll,
        "best_compressed_policy": best_policy,
        "best_compressed_nll": float(best_metrics["nll"]),
        "margin_vs_rkv_like": margin_vs_rkv,
        "margin_vs_thin_kv_like": margin_vs_thin,
        "paired_delta_vs_rkv_like": paired_vs_rkv.get(PSI_POLICY_NAME, {}),
        "paired_delta_vs_thin_kv_like": paired_vs_thin.get(PSI_POLICY_NAME, {}),
    }


def _status(decision: dict[str, object]) -> str:
    if decision["promotion_pass"]:
        return "ALIVE on one-shot fresh sparse-cache surface; psi_topk clears the preregistered promotion rule."
    return "KILLED on one-shot fresh sparse-cache surface; psi_topk fails the preregistered promotion rule."


def run(
    model_name: str,
    keep_fraction: float,
    max_traces: int,
    max_length: int,
    continuation_tokens: int,
    trace_paths: list[Path] | None = None,
) -> dict[str, object]:
    cache_dir = ROOT / "experimental/thoughtflow_fp8/.debug/hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = _load_model_for_prefill_attentions(model_name)
    model.eval()
    resolved_trace_paths = _resolve_input_paths(trace_paths or [])
    prepared = _prepare_rows(tokenizer, max_traces, max_length, continuation_tokens, resolved_trace_paths)
    policies = _frozen_policies()
    rows = []
    psi_per_trace = []
    with torch.no_grad():
        for row in prepared:
            prefix_ids = row["prefix_ids"]
            continuation_ids = row["continuation_ids"]
            trace = row["trace"]
            assert isinstance(prefix_ids, list)
            assert isinstance(continuation_ids, list)
            assert isinstance(trace, list)

            prefix = torch.tensor([prefix_ids], dtype=torch.long)
            prefix_outputs = model(input_ids=prefix, use_cache=True)
            full_kept = set(range(len(prefix_ids)))
            full_loss, scored_tokens = _score_continuation_with_cache(
                model,
                prefix_outputs,
                len(prefix_ids),
                continuation_ids,
                full_kept,
            )
            rows.append(
                {
                    "trace_id": row["trace_id"],
                    "policy": "full_cache",
                    "keep_rate": 1.0,
                    "retained_prefix_tokens": len(prefix_ids),
                    "continuation_tokens": scored_tokens,
                    "nll": full_loss,
                    "delta_nll_vs_full": 0.0,
                }
            )
            budget = max(1, math.ceil(len(prefix_ids) * keep_fraction))
            for name, policy in policies.items():
                kept = policy(trace, budget)
                loss, scored_tokens = _score_continuation_with_cache(
                    model,
                    prefix_outputs,
                    len(prefix_ids),
                    continuation_ids,
                    kept,
                )
                rows.append(
                    {
                        "trace_id": row["trace_id"],
                        "policy": name,
                        "keep_rate": len(kept) / len(prefix_ids),
                        "retained_prefix_tokens": len(kept),
                        "continuation_tokens": scored_tokens,
                        "nll": loss,
                        "delta_nll_vs_full": loss - full_loss,
                    }
                )
            psi_kept, psi_scores = _psi_topk_from_logits(prefix_outputs.logits, prefix_ids, budget)
            psi_loss, scored_tokens = _score_continuation_with_cache(
                model,
                prefix_outputs,
                len(prefix_ids),
                continuation_ids,
                psi_kept,
            )
            rows.append(
                {
                    "trace_id": row["trace_id"],
                    "policy": PSI_POLICY_NAME,
                    "keep_rate": len(psi_kept) / len(prefix_ids),
                    "retained_prefix_tokens": len(psi_kept),
                    "continuation_tokens": scored_tokens,
                    "nll": psi_loss,
                    "delta_nll_vs_full": psi_loss - full_loss,
                }
            )
            psi_per_trace.append(
                {
                    "trace_id": row["trace_id"],
                    "telemetry": _psi_telemetry(psi_scores, psi_kept),
                }
            )

    summary = _summary(rows)
    paired_vs_rkv = _paired_deltas(rows, baseline_policy="rkv_like")
    paired_vs_thin = _paired_deltas(rows, baseline_policy="thin_kv_like")
    decision = _promotion_decision(summary, paired_vs_rkv, paired_vs_thin)
    return {
        "model_name": model_name,
        "input_paths": [_path_label(path) for path in resolved_trace_paths],
        "keep_fraction": keep_fraction,
        "max_traces": max_traces,
        "max_length": max_length,
        "continuation_tokens": continuation_tokens,
        "n_scored_traces": int(summary.get("full_cache", {}).get("n_traces", 0)),
        "policy_name": PSI_POLICY_NAME,
        "frozen_policy_names": ["thoughtflow_saliency_recent", FROZEN_SPARSE_POLICY_NAME, PSI_POLICY_NAME],
        "rows": rows,
        "summary": summary,
        "paired_delta_nll_vs_rkv_like": paired_vs_rkv,
        "paired_delta_nll_vs_thin_kv_like": paired_vs_thin,
        "psi_topk_telemetry": {
            "aggregate": _aggregate_psi_telemetry(psi_per_trace),
            "per_trace": psi_per_trace,
        },
        "decision": decision,
        "status": _status(decision),
    }


def _fmt_pair(metrics: dict[str, float], key: str) -> str:
    if not metrics:
        return "n/a"
    return f"{metrics[key]:+.3f} [{metrics['ci95_low']:+.3f},{metrics['ci95_high']:+.3f}]"


def write_markdown(result: dict[str, object], output_path: Path = DEFAULT_MD_OUTPUT) -> None:
    decision = result["decision"]
    lines = [
        "# ThoughtFlow-FP8 Prefix-Surprisal Fresh Sparse-Cache Check",
        "",
        f"Status: **{result['status']}**",
        "",
        f"- model: `{result['model_name']}`",
        f"- policy: `{result['policy_name']}`",
        f"- scored traces: {result['n_scored_traces']}",
        f"- keep fraction: {result['keep_fraction']:.2f}",
        f"- max length: {result['max_length']}",
        f"- continuation tokens: {result['continuation_tokens']}",
        "",
        "This is a one-shot run of the pre-registered prefix-surprisal utility on a saved-trace surface not used by the RDU promotion gate. It performs no sweep, retuning, or variant selection.",
        "",
        "## Trace Inputs",
        "",
    ]
    lines.extend(f"- `{path}`" for path in result["input_paths"])
    lines.extend(
        [
            "",
            "## Policy Table",
            "",
            "| Policy | Traces | Keep rate | NLL | Delta NLL vs full cache |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for policy, metrics in sorted(result["summary"].items(), key=lambda item: item[1]["nll"]):
        lines.append(
            "| {policy} | {n_traces:.0f} | {keep_rate:.3f} | {nll:.3f} | {delta_nll_vs_full:.3f} |".format(
                policy=policy,
                **metrics,
            )
        )
    lines.extend(
        [
            "",
            "## Promotion Readout",
            "",
            f"- best compressed policy: `{decision['best_compressed_policy']}`",
            f"- `psi_topk` margin vs R-KV-like: {decision['margin_vs_rkv_like']:+.3f}",
            f"- `psi_topk` paired delta vs R-KV-like: {_fmt_pair(decision['paired_delta_vs_rkv_like'], 'mean_delta_nll_minus_rkv_like')}",
            f"- `psi_topk` margin vs ThinKV-like: {decision['margin_vs_thin_kv_like']:+.3f}",
            f"- `psi_topk` paired delta vs ThinKV-like: {_fmt_pair(decision['paired_delta_vs_thin_kv_like'], 'mean_delta_nll_minus_thin_kv_like')}",
            f"- promotion pass: {decision['promotion_pass']}",
            "",
            "## Prefix-Surprisal Telemetry",
            "",
            "| Mean surprisal | Mean kept surprisal | Max surprisal | Nonzero tokens |",
            "|---:|---:|---:|---:|",
            "| {mean_surprisal:.3f} | {mean_kept_surprisal:.3f} | {max_surprisal:.3f} | {nonzero_tokens:.0f} |".format(
                **result["psi_topk_telemetry"]["aggregate"]
            ),
            "",
            "## Decision",
            "",
            "Promotion required `psi_topk` to beat both R-KV-like and ThinKV-like by at least 0.03 NLL with paired CIs below zero and to be the best compressed row. Failure rules out this exact signal in the current Mac sparse-cache harness.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="distilgpt2")
    parser.add_argument("--keep-fraction", type=float, default=0.20)
    parser.add_argument("--max-traces", type=int, default=70)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--continuation-tokens", type=int, default=24)
    parser.add_argument("--input-jsonl", action="append", type=Path, default=[])
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()

    result = run(
        args.model_name,
        args.keep_fraction,
        args.max_traces,
        args.max_length,
        args.continuation_tokens,
        trace_paths=args.input_jsonl,
    )
    args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_markdown(result, args.md_output)
    print(json.dumps({"status": result["status"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()

