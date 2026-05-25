"""Verify the CPU release environment and config placeholders."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from outlier_migrate.data import write_json
from outlier_migrate.environment import collect_environment


DEFAULT_CONFIGS = (
    Path("configs/granite_tiny.yaml"),
    Path("configs/granite_small.yaml"),
    Path("configs/nemotron3_nano.yaml"),
    Path("configs/qwen36.yaml"),
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", dest="configs")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run release environment verification."""

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()))
    if not args.dry_run:
        raise SystemExit("environment verification currently supports dry-run only")
    config_paths = tuple(args.configs) if args.configs else DEFAULT_CONFIGS
    report = collect_environment(config_paths)
    output_path = args.output_dir / "environment_manifest.json"
    write_json(output_path, report.to_dict())
    logging.info("wrote %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
