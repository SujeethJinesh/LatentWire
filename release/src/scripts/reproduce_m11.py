"""Reproduce M11 EMA smoothing claims."""

from __future__ import annotations

import argparse
from pathlib import Path

from outlier_migrate.data import claim_subset, load_config, verify_claims, write_result


def main() -> None:
    """Run M11 reproduction or fast verification."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="results/m11")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fast-verify", action="store_true")
    args = parser.parse_args()
    load_config(args.config)
    claims = claim_subset(["m11_median"])
    write_result(Path(args.output_dir) / "m11.json", {"claims": claims, "verified": verify_claims(claims)})


if __name__ == "__main__":
    main()
