"""Reproduce FFT spectral characterization claims."""

from __future__ import annotations

import argparse
from pathlib import Path

from outlier_migrate.data import build_reproduction_payload, claim_subset, load_config, write_result


def main() -> None:
    """Run FFT reproduction or fast verification."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="results/fft")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fast-verify", action="store_true")
    args = parser.parse_args()
    load_config(args.config)
    claims = claim_subset(["fft_entropy", "fft_autocorr_tokens"])
    write_result(Path(args.output_dir) / "fft.json", build_reproduction_payload(args, claims))


if __name__ == "__main__":
    main()
