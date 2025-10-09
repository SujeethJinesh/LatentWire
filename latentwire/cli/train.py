import argparse
import os
import sys
from copy import deepcopy
from typing import Any, Dict, List, Optional

from latentwire.cli.utils import (
    append_metrics_history,
    apply_overrides,
    config_to_argv,
    flatten_training_config,
    load_training_config,
    summarize_features,
)
from latentwire.config import TrainingConfig


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LatentWire training CLI")
    parser.add_argument("--config", required=True, help="Path to training config JSON")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override configuration values (use dot notation, e.g. data.samples=128)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print derived arguments without launching training",
    )
    parser.add_argument(
        "--print-argv",
        action="store_true",
        help="Only print generated argv for latentwire.train and exit",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="run",
        help="Identifier recorded in metrics history entries",
    )
    return parser


def config_to_argv_list(config_path: str, overrides: List[str]) -> Dict[str, Any]:
    cfg_obj, cfg_dict = load_training_config(config_path)
    effective_dict = apply_overrides(cfg_dict, overrides) if overrides else cfg_dict
    cfg = TrainingConfig.from_dict(effective_dict)
    flat = flatten_training_config(cfg.to_dict())
    argv = config_to_argv(flat)
    return {
        "config": cfg,
        "config_dict": cfg.to_dict(),
        "argv": argv,
    }


def run_train(argv: List[str]) -> None:
    from latentwire import train as train_module

    argv = ["latentwire-train"] + argv
    old_argv = sys.argv
    try:
        sys.argv = argv
        train_module.main()
    finally:
        sys.argv = old_argv


def main(cli_args: Optional[List[str]] = None) -> None:
    parser = build_cli_parser()
    args = parser.parse_args(cli_args)

    result = config_to_argv_list(args.config, args.override)
    cfg: TrainingConfig = result["config"]
    cfg_dict: Dict[str, Any] = result["config_dict"]
    argv: List[str] = result["argv"]

    print("=== LatentWire Train CLI ===")
    print(f"Config: {os.path.abspath(args.config)}")
    if args.override:
        print("Overrides:")
        for item in args.override:
            print(f"  - {item}")
    print("Feature toggles:")
    feature_summary = summarize_features(cfg_dict)
    for name, enabled in feature_summary.items():
        status = "ON" if enabled else "off"
        print(f"  {name}: {status}")
    print("Derived argv:", " ".join(argv))

    if args.print_argv or args.dry_run:
        print("[CLI] Dry run complete; training not launched.")
        return

    run_train(argv)

    save_dir = cfg.checkpoint.save_dir
    record = {
        "kind": "train",
        "tag": args.tag,
        "config_path": os.path.abspath(args.config),
        "overrides": deepcopy(args.override),
        "argv": argv,
        "features": feature_summary,
    }
    try:
        append_metrics_history(save_dir, record)
        print(f"[CLI] Recorded metrics history entry in {save_dir}")
    except Exception as exc:
        print(f"[CLI] Warning: failed to append metrics history ({exc})")


if __name__ == "__main__":
    main()
