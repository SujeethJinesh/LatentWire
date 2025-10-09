import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from latentwire.cli.utils import append_metrics_history


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LatentWire evaluation CLI")
    parser.add_argument("--config", required=True, help="Path to evaluation config JSON")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config entries using dot notation",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print command and exit")
    parser.add_argument("--tag", type=str, default="eval", help="Label stored in metrics history")
    return parser


def load_eval_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Evaluation config must be a JSON object")
    return data


def apply_eval_overrides(base: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    if not overrides:
        return base
    cfg = json.loads(json.dumps(base))
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        value = raw_value.strip()
        cursor = cfg
        parts = key.split(".")
        for segment in parts[:-1]:
            cursor = cursor.setdefault(segment, {})
        cursor[parts[-1]] = _infer_type(value)
    return cfg


def _infer_type(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith("[") and value.endswith("]"):
        try:
            return json.loads(value)
        except Exception:
            pass
    return value


def dict_to_argv(config: Dict[str, Any]) -> List[str]:
    argv: List[str] = []
    for key, value in config.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                argv.append(f"--{key}")
            continue
        if isinstance(value, list):
            value_str = json.dumps(value)
        else:
            value_str = str(value)
        argv.extend([f"--{key}", value_str])
    return argv


def run_eval(argv: List[str]) -> None:
    from latentwire import eval as eval_module

    argv = ["latentwire-eval"] + argv
    old_argv = sys.argv
    try:
        sys.argv = argv
        eval_module.main()
    finally:
        sys.argv = old_argv


def main(cli_args: Optional[List[str]] = None) -> None:
    parser = build_cli_parser()
    args = parser.parse_args(cli_args)

    base_cfg = load_eval_config(args.config)
    effective_cfg = apply_eval_overrides(base_cfg, args.override)
    argv = dict_to_argv(effective_cfg)

    print("=== LatentWire Eval CLI ===")
    print(f"Config: {os.path.abspath(args.config)}")
    if args.override:
        print("Overrides:")
        for item in args.override:
            print(f"  - {item}")
    print("Derived argv:", " ".join(argv))

    if args.dry_run:
        print("[CLI] Dry run complete; evaluation not launched.")
        return

    run_eval(argv)

    out_dir = effective_cfg.get("out_dir") or effective_cfg.get("save_dir") or "."
    record = {
        "kind": "eval",
        "tag": args.tag,
        "config_path": os.path.abspath(args.config),
        "overrides": list(args.override or []),
        "argv": argv,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    try:
        append_metrics_history(out_dir, record)
        print(f"[CLI] Appended metrics history entry to {out_dir}")
    except Exception as exc:
        print(f"[CLI] Warning: failed to append metrics history ({exc})")


if __name__ == "__main__":
    main()
