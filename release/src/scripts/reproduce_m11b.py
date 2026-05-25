"""Reproduce M11b budget-tuned EMA claims."""

from __future__ import annotations

import argparse
from pathlib import Path

from outlier_migrate.data import build_reproduction_payload, claim_subset, load_config, write_result


def main() -> None:
    """Run M11b reproduction or fast verification."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="results/m11b")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fast-verify", action="store_true")
    args = parser.parse_args()
    load_config(args.config)
    claims = claim_subset(["m11b_granite_top5", "m11b_nemotron_top5", "m11b_nemotron_top10"])
    write_result(Path(args.output_dir) / "m11b.json", build_reproduction_payload(args, claims))


if __name__ == "__main__":
    main()
