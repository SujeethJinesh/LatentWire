#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run Step 14 regime map analysis.")
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--runs-root", default="quantization/data")
    parser.add_argument("--output-dir", default="quantization/analysis/regime_map")
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    if args.run_tag is None:
        args.run_tag = time.strftime("step14_%Y%m%d_%H%M%S")
    run_root = project_root / "quantization" / "data" / "step_14_regime_map" / args.run_tag
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "manifests").mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        str(project_root / "quantization" / "scripts" / "analyze_regime_map.py"),
        "--runs-root", str(args.runs_root),
        "--output-dir", str(args.output_dir),
    ]
    subprocess.check_call(cmd)
    manifest = {
        "runs_root": args.runs_root,
        "output_dir": args.output_dir,
        "run_tag": args.run_tag,
    }
    (run_root / "manifests" / "step_14_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
