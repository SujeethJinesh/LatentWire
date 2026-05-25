"""Reproduce four-model set-leaving claims."""

from __future__ import annotations

import argparse
from pathlib import Path

from outlier_migrate.data import build_reproduction_payload, claim_subset, load_config, write_result


def main() -> None:
    """Run set-leaving reproduction or fast verification."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="results/set_leaving")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fast-verify", action="store_true")
    args = parser.parse_args()
    load_config(args.config)
    keys = [
        "set_leaving_granite_small",
        "set_leaving_nemotron",
        "set_leaving_deepseek",
        "set_leaving_falcon",
    ]
    claims = claim_subset(keys)
    payload = build_reproduction_payload(args, claims)
    write_result(Path(args.output_dir) / "set_leaving.json", payload)


if __name__ == "__main__":
    main()
