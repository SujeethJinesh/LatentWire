import argparse
import json
import os
from copy import deepcopy
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Tuple

from latentwire.cli import train as train_cli
from latentwire.cli.utils import append_metrics_history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LatentWire ablation harness")
    parser.add_argument("--config", required=True, help="Ablation config JSON path")
    parser.add_argument("--dry-run", action="store_true", help="Show planned runs without executing")
    parser.add_argument("--tag-prefix", type=str, default="ablation", help="Prefix applied to run tags")
    return parser


def load_ablation_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Ablation config must be a JSON object")
    return data


def expand_grid(sweeps: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    keys = list(sweeps.keys())
    values = [list(v) for v in sweeps.values()]
    combinations = []
    for combo in product(*values):
        overrides = {}
        for key, value in zip(keys, combo):
            overrides[key] = value
        combinations.append(overrides)
    return combinations


def format_override_dict(overrides: Dict[str, Any]) -> List[str]:
    formatted = []
    for key, value in overrides.items():
        if isinstance(value, (list, dict)):
            formatted.append(f"{key}={json.dumps(value)}")
        else:
            formatted.append(f"{key}={value}")
    return formatted


def run_ablation(config_path: str, args: argparse.Namespace) -> None:
    cfg = load_ablation_config(config_path)
    base_config = cfg.get("base_config")
    if not base_config:
        raise ValueError("Ablation config must include 'base_config'")
    base_overrides = cfg.get("base_overrides", [])
    runs: List[Tuple[str, List[str]]] = []

    explicit_runs = cfg.get("runs", [])
    for entry in explicit_runs:
        name = entry.get("name")
        if not name:
            raise ValueError("Each explicit run must include a 'name'")
        override_list = []
        extra = entry.get("overrides", {})
        for key, value in extra.items():
            if isinstance(value, (list, dict)):
                override_list.append(f"{key}={json.dumps(value)}")
            else:
                override_list.append(f"{key}={value}")
        runs.append((name, override_list))

    sweeps = cfg.get("sweeps", {})
    if sweeps:
        grid_overrides = expand_grid(sweeps)
        for idx, overrides in enumerate(grid_overrides, start=1):
            name = f"sweep_{idx}"
            runs.append((name, format_override_dict(overrides)))

    if not runs:
        runs.append(("default", []))

    for name, overrides in runs:
        merged_overrides = list(base_overrides) + list(overrides)
        tag = f"{args.tag_prefix}-{name}"
        print(f"\n[Ablation] Run '{name}' overrides: {merged_overrides}")

        if args.dry_run:
            continue

        result = train_cli.config_to_argv_list(base_config, merged_overrides)
        cfg_obj = result["config"]
        argv = result["argv"]
        print(f"[Ablation] Launching training for '{name}'")
        train_cli.run_train(argv)

        record = {
            "kind": "ablation-train",
            "tag": tag,
            "base_config": os.path.abspath(base_config),
            "overrides": merged_overrides,
            "argv": argv,
        }
        try:
            append_metrics_history(cfg_obj.checkpoint.save_dir, record)
        except Exception as exc:
            print(f"[Ablation] Warning: failed to append metrics history ({exc})")


def main(cli_args: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(cli_args)
    run_ablation(args.config, args)


if __name__ == "__main__":
    main()

