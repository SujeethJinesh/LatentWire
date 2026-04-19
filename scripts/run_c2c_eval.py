from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from latent_bridge.c2c_eval import run_c2c_generation_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prediction-output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_c2c_generation_eval(
        source_model=args.source_model,
        target_model=args.target_model,
        eval_file=args.eval_file,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        prediction_output=args.prediction_output,
    )
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
