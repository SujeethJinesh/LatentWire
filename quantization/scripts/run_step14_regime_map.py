#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run Step 14 regime map analysis.")
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--runs-root", default="quantization/data")
    parser.add_argument("--output-dir", default="quantization/analysis/regime_map")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    cmd = [
        "python",
        str(project_root / "quantization" / "scripts" / "analyze_regime_map.py"),
        "--runs-root", str(args.runs_root),
        "--output-dir", str(args.output_dir),
    ]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
