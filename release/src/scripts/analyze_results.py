"""Dry-run analysis command for release artifacts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from outlier_migrate.analysis import write_analysis_manifest


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the analysis command."""

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()))
    if not args.dry_run:
        raise SystemExit("full result analysis is not included in the initial release scaffold")
    write_analysis_manifest(
        args.results_dir / "analysis_manifest.json",
        status="dry_run",
        rows=[],
    )
    logging.info("wrote %s", args.results_dir / "analysis_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
