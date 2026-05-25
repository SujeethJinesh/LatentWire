"""Reproduce KL accumulation claims."""

from __future__ import annotations

import argparse
from pathlib import Path

from outlier_migrate.data import build_reproduction_payload, claim_subset, load_config, write_result


def main() -> None:
    """Run KL reproduction or fast verification."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="results/kl_accumulation")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fast-verify", action="store_true")
    args = parser.parse_args()
    load_config(args.config)
    claims = claim_subset(["kl_static_mean", "kl_decdec_mean", "kl_m11_mean"])
    write_result(Path(args.output_dir) / "kl_accumulation.json", build_reproduction_payload(args, claims))


if __name__ == "__main__":
    main()
