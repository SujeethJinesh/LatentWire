from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from latent_bridge.baselines import C2CAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the published C2C bundle for the requested pair.",
    )
    parser.add_argument(
        "--c2c-root",
        default="references/repos/C2C",
        help="Optional local C2C repository path for bootstrap notes.",
    )
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = C2CAdapter.prepare_published_artifact(
        args.source_model,
        args.target_model,
        download=args.download,
    )
    payload = {
        "baseline": artifact.baseline,
        "repo_id": artifact.repo_id,
        "source_model": artifact.source_model,
        "target_model": artifact.target_model,
        "subdir": artifact.subdir,
        "config_path": artifact.config_path,
        "checkpoint_dir": artifact.checkpoint_dir,
        "local_root": artifact.local_root,
        "local_config_path": C2CAdapter.local_config_path(artifact),
        "local_checkpoint_dir": C2CAdapter.local_checkpoint_dir(artifact),
        "local_repo_exists": pathlib.Path(args.c2c_root).exists(),
        "local_repo_root": str(pathlib.Path(args.c2c_root).resolve()),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json:
        out = pathlib.Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
