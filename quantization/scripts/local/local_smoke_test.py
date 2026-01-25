#!/usr/bin/env python3
"""Wrapper for local smoke tests (delegates to run_step0_baselines.py)."""
import os
import sys
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "quantization" / "scripts" / "run_step0_baselines.py"
    args = [
        sys.executable,
        str(script),
        "--mode",
        "local",
        "--project-root",
        str(repo_root),
    ] + sys.argv[1:]
    os.execv(sys.executable, args)


if __name__ == "__main__":
    main()
