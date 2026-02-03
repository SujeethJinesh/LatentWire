#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path


def collect_manifests(root: Path):
    for manifest in root.rglob("*_manifest.json"):
        yield manifest


def main():
    parser = argparse.ArgumentParser(description="Report drift between estimated and measured bytes.")
    parser.add_argument("--runs-root", default="quantization/data", help="Runs root to scan")
    parser.add_argument("--out", default=None, help="Optional path to write JSON report")
    args = parser.parse_args()

    root = Path(args.runs_root)
    if not root.exists():
        print(f"No runs root: {root}")
        return 1

    drifts = []
    skipped = 0
    for manifest_path in collect_manifests(root):
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            continue
        measured = manifest.get("bytes_measured_total")
        estimated = manifest.get("bytes_estimated_total") or manifest.get("bytes_estimate")
        if isinstance(measured, dict):
            for key, val in measured.items():
                if val is None:
                    continue
                est_val = None
                if isinstance(estimated, dict):
                    est_val = estimated.get(key)
                if est_val is None:
                    skipped += 1
                    continue
                try:
                    drifts.append(float(val) - float(est_val))
                except Exception:
                    skipped += 1
        elif measured is not None:
            if isinstance(estimated, (int, float)):
                drifts.append(float(measured) - float(estimated))
            else:
                skipped += 1
    if not drifts:
        print("No drift samples found.")
        return 0
    report = {
        "count": len(drifts),
        "mean": statistics.mean(drifts),
        "median": statistics.median(drifts),
        "p95": statistics.quantiles(drifts, n=20)[-1] if len(drifts) >= 20 else max(drifts),
        "min": min(drifts),
        "max": max(drifts),
        "skipped": skipped,
    }
    report_json = json.dumps(report, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_json)
    print(report_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
