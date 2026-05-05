#!/usr/bin/env python3
"""Collect a small no-behavior-change C2C generation trace smoke artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latent_bridge.c2c_eval import (
    extract_c2c_generation_trace_features,
    load_c2c_model,
)
from latent_bridge.evaluate import _generation_example_id, load_generation


def _resolve(path: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _feature_digest(features) -> dict[str, Any]:
    array = features.detach().cpu().contiguous().numpy()
    return {
        "shape": [int(dim) for dim in features.shape],
        "dtype": str(features.dtype),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "preview": [float(value) for value in features.flatten()[:16].tolist()],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--residual-projection-dim", type=int, default=4)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_path = _resolve(args.eval_file)
    examples = load_generation(str(eval_path))[: int(args.limit)]
    model, tokenizer, artifact = load_c2c_model(
        source_model=str(args.source_model),
        target_model=str(args.target_model),
        device=str(args.device),
        max_new_tokens=int(args.max_new_tokens),
    )
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        features, metadata = extract_c2c_generation_trace_features(
            model,
            tokenizer,
            example.prompt,
            device=str(args.device),
            max_new_tokens=int(args.max_new_tokens),
            residual_projection_dim=int(args.residual_projection_dim),
        )
        components = metadata.get("components", {})
        projector_meta = components.get("projector", {})
        logits_meta = components.get("target_logits", {})
        rows.append(
            {
                "index": int(index),
                "example_id": _generation_example_id(example),
                "feature_digest": _feature_digest(features),
                "feature_family": metadata.get("feature_family"),
                "formatted_prompt_tokens": metadata.get("formatted_prompt_tokens"),
                "generated_tokens": metadata.get("generated_tokens"),
                "decoded_prediction": metadata.get("decoded_prediction"),
                "projector_history_lengths": [
                    int(item.get("history_length", 0))
                    for item in projector_meta.get("projectors", [])
                ],
                "target_logit_steps": int(logits_meta.get("step_count", 0)),
            }
        )
    payload = {
        "run_config": {
            "source_model": str(args.source_model),
            "target_model": str(args.target_model),
            "eval_file": _display_path(eval_path),
            "device": str(args.device),
            "max_new_tokens": int(args.max_new_tokens),
            "limit": int(args.limit),
            "residual_projection_dim": int(args.residual_projection_dim),
            "published_repo_id": artifact.repo_id,
            "published_subdir": artifact.subdir,
            "published_config_path": artifact.config_path,
            "published_checkpoint_dir": artifact.checkpoint_dir,
            "local_root": artifact.local_root,
        },
        "rows": rows,
    }
    output_path = _resolve(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": _display_path(output_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
