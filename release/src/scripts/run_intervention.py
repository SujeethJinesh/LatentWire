"""Dry-run intervention reproduction command."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from outlier_migrate.config import load_experiment_config
from outlier_migrate.data import write_json
from outlier_migrate.registry import default_registry


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the intervention command."""

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()))
    config = load_experiment_config(args.config)
    registry = default_registry()
    if not args.dry_run:
        raise SystemExit("full intervention execution is not included in the initial release scaffold")
    manifest = {
        "command": "run_intervention",
        "dry_run": True,
        "model_id": config.model.model_id,
        "registered_methods": registry.names(),
        "default_output_dir": str(args.output_dir),
    }
    write_json(args.output_dir / "intervention_manifest.json", manifest)
    logging.info("wrote %s", args.output_dir / "intervention_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
