"""Verify the release environment."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import sys


def main() -> None:
    """Check required package imports and pinned versions."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    required = {"numpy": "2.1.2", "PyYAML": "6.0.3"}
    observed = {name: metadata.version(name) for name in required}
    mismatches = {name: observed[name] for name, version in required.items() if observed[name] != version}
    gpu = False
    if args.require_gpu:
        import torch

        gpu = bool(torch.cuda.is_available())
        if not gpu:
            raise RuntimeError("GPU was required but torch.cuda.is_available() is false")
    payload = {"python": sys.version, "packages": observed, "mismatches": mismatches, "gpu": gpu}
    if mismatches:
        raise RuntimeError(json.dumps(payload, indent=2))
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
