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
                "method_family": method_family,
                "bytes_measured_total": bytes_total,
                "budget_bucket": budget_bucket,
                "accuracy": accuracy,
            })

    out_csv = output_dir / "regime_table.csv"
    with out_csv.open("w", encoding="utf-8") as handle:
        handle.write("dataset,pair_id,method_family,bytes_measured_total,budget_bucket,accuracy,run_root\n")
        for row in rows:
            handle.write(
                f"{row['dataset']},{row['pair_id']},{row['method_family']},{row['bytes_measured_total']},{row['budget_bucket']},{row['accuracy']},{row['run_root']}\n"
            )

    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
