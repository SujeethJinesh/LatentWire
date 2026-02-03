#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path


def find_latest_summary(results_dir: Path):
    summaries = sorted(results_dir.glob("*_summary.json"), key=lambda p: p.stat().st_mtime)
    return summaries[-1] if summaries else None


def load_manifest(run_root: Path):
    for name in ("step_8_manifest.json", "step_11_manifest.json", "step_12_manifest.json", "step_13_manifest.json"):
        path = run_root / "manifests" / name
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                return {}
    return {}


def infer_method_family(manifest: dict) -> str:
    if manifest.get("baseline_family"):
        return manifest.get("baseline_family")
    kv_transfer = manifest.get("kv_transfer_config") or {}
    if kv_transfer.get("token_precision_mode") == "rd_greedy":
        return "rd_c2c"
    if kv_transfer.get("enabled", False):
        return "c2c"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Regime map analysis.")
    parser.add_argument("--runs-root", default="quantization/data", help="Runs root")
    parser.add_argument("--output-dir", default="quantization/analysis/regime_map")
    parser.add_argument("--budget-bins", default="", help="Comma-separated byte bins (e.g., 1e5,2e5,4e5)")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bins = []
    if args.budget_bins:
        for item in args.budget_bins.split(","):
            try:
                bins.append(float(item))
            except Exception:
                pass
    bins = sorted(bins)

    rows = []
    for run_root in sorted(runs_root.glob("step_*/*")):
        if not run_root.is_dir():
            continue
        results_root = run_root / "results"
        if not results_root.exists():
            continue
        manifest = load_manifest(run_root)
        pair_id = manifest.get("pair_id")
        context_len_bucket = manifest.get("context_len_bucket")
        method_family = infer_method_family(manifest)
        bytes_measured = manifest.get("bytes_measured_total")
        bytes_estimated = manifest.get("bytes_estimated_total") or manifest.get("bytes_estimate")
        for dataset_dir in sorted(results_root.iterdir()):
            if not dataset_dir.is_dir():
                continue
            summary_path = find_latest_summary(dataset_dir)
            if not summary_path:
                continue
            try:
                summary = json.loads(summary_path.read_text())
            except Exception:
                continue
            accuracy = summary.get("overall_accuracy")
            if isinstance(summary.get("length_statistics"), dict):
                accuracy = summary.get("length_statistics", {}).get("overall", {}).get("accuracy", accuracy)
            bytes_total = None
            if isinstance(bytes_measured, dict):
                bytes_total = bytes_measured.get(dataset_dir.name)
            elif bytes_measured is not None:
                bytes_total = bytes_measured
            if bytes_total is None:
                if isinstance(bytes_estimated, dict):
                    bytes_total = bytes_estimated.get(dataset_dir.name)
                elif bytes_estimated is not None:
                    bytes_total = bytes_estimated
            budget_bucket = None
            if bins and bytes_total is not None:
                for b in bins:
                    if bytes_total <= b:
                        budget_bucket = b
                        break
                if budget_bucket is None:
                    budget_bucket = bins[-1]
            rows.append({
                "run_root": str(run_root),
                "dataset": dataset_dir.name,
                "pair_id": pair_id,
                "context_len_bucket": context_len_bucket,
                "method_family": method_family,
                "bytes_measured_total": bytes_total,
                "budget_bucket": budget_bucket,
                "accuracy": accuracy,
            })

    out_csv = output_dir / "regime_table.csv"
    with out_csv.open("w", encoding="utf-8") as handle:
        handle.write("dataset,pair_id,context_len_bucket,method_family,bytes_measured_total,budget_bucket,accuracy,run_root\n")
        for row in rows:
            handle.write(
                f"{row['dataset']},{row['pair_id']},{row['context_len_bucket']},{row['method_family']},{row['bytes_measured_total']},{row['budget_bucket']},{row['accuracy']},{row['run_root']}\n"
            )

    def better_row(a, b):
        if a is None:
            return b
        if b is None:
            return a
        acc_a = a.get("accuracy")
        acc_b = b.get("accuracy")
        if acc_a is None:
            return b
        if acc_b is None:
            return a
        if acc_b > acc_a:
            return b
        if acc_b < acc_a:
            return a
        # tie-breaker: lower bytes
        bytes_a = a.get("bytes_measured_total")
        bytes_b = b.get("bytes_measured_total")
        if bytes_a is None:
            return b
        if bytes_b is None:
            return a
        return b if bytes_b < bytes_a else a

    summary_rows = []
    cells = {}
    for row in rows:
        key = (row.get("dataset"), row.get("pair_id"), row.get("context_len_bucket"), row.get("budget_bucket"))
        cell = cells.setdefault(key, {"cache": None, "text": None})
        family = row.get("method_family")
        if family == "text":
            cell["text"] = better_row(cell["text"], row)
        else:
            cell["cache"] = better_row(cell["cache"], row)
    for key, cell in cells.items():
        dataset, pair_id, context_len_bucket, budget_bucket = key
        cache = cell.get("cache")
        text = cell.get("text")
        cache_acc = cache.get("accuracy") if cache else None
        text_acc = text.get("accuracy") if text else None
        delta = None
        if cache_acc is not None and text_acc is not None:
            delta = cache_acc - text_acc
        winner = None
        if delta is not None:
            winner = "cache" if delta > 0 else ("text" if delta < 0 else "tie")
        summary_rows.append({
            "dataset": dataset,
            "pair_id": pair_id,
            "context_len_bucket": context_len_bucket,
            "budget_bucket": budget_bucket,
            "best_cache_family": cache.get("method_family") if cache else None,
            "best_cache_accuracy": cache_acc,
            "best_cache_bytes": cache.get("bytes_measured_total") if cache else None,
            "best_text_accuracy": text_acc,
            "best_text_bytes": text.get("bytes_measured_total") if text else None,
            "delta_accuracy": delta,
            "winner": winner,
        })

    summary_csv = output_dir / "regime_summary.csv"
    with summary_csv.open("w", encoding="utf-8") as handle:
        handle.write(
            "dataset,pair_id,context_len_bucket,budget_bucket,best_cache_family,best_cache_accuracy,best_cache_bytes,best_text_accuracy,best_text_bytes,delta_accuracy,winner\n"
        )
        for row in summary_rows:
            handle.write(
                f"{row['dataset']},{row['pair_id']},{row['context_len_bucket']},{row['budget_bucket']},{row['best_cache_family']},{row['best_cache_accuracy']},{row['best_cache_bytes']},{row['best_text_accuracy']},{row['best_text_bytes']},{row['delta_accuracy']},{row['winner']}\n"
            )

    summary_md = output_dir / "regime_summary.md"
    with summary_md.open("w", encoding="utf-8") as handle:
        handle.write("# Regime Map Summary\n\n")
        handle.write(f"Cells: {len(summary_rows)}\n\n")
        wins_cache = sum(1 for r in summary_rows if r.get("winner") == "cache")
        wins_text = sum(1 for r in summary_rows if r.get("winner") == "text")
        wins_tie = sum(1 for r in summary_rows if r.get("winner") == "tie")
        handle.write(f"- cache wins: {wins_cache}\n")
        handle.write(f"- text wins: {wins_text}\n")
        handle.write(f"- ties: {wins_tie}\n")

    print(f"Wrote {out_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
