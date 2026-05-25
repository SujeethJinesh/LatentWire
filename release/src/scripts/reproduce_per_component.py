"""Reproduce per-component set-leaving claims."""

from __future__ import annotations

import argparse
from pathlib import Path

from outlier_migrate.data import build_reproduction_payload, claim_subset, load_config, write_result


def main() -> None:
    """Run per-component reproduction or fast verification."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="results/per_component")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fast-verify", action="store_true")
    args = parser.parse_args()
    load_config(args.config)
    keys = [
        "component_granite_attention",
        "component_granite_ssm",
        "component_nemotron_attention",
        "component_nemotron_moe",
        "component_nemotron_ssm",
    ]
    claims = claim_subset(keys)
    write_result(Path(args.output_dir) / "per_component.json", build_reproduction_payload(args, claims))


if __name__ == "__main__":
    main()
