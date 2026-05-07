"""Synthetic HORN H1a real-schema rehearsal.

This no-download gate writes rows with the same schema as a real HORN H1a
boundary-activation packet, then marks the decision schema-rehearsal and
non-promotable. It validates prompt-paired boundary, non-boundary, and
direction-permutation controls before running on real hybrid-model activations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experimental.shared.hybrid_gate_evaluators import evaluate_horn_h1


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "experimental/horn/phase2/results/horn_synthetic_h1"
MODEL_ID = "synthetic-hybrid-horn-schema-rehearsal"
PROMPT_IDS = tuple(f"synthetic_prompt_{index:02d}" for index in range(12))


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _config(seed: int) -> dict[str, object]:
    prompt_manifest = "\n".join(PROMPT_IDS) + "\n"
    architecture_manifest = json.dumps(
        {
            "model_id": MODEL_ID,
            "boundary_count": 2,
            "directions": ["attention->ssm", "ssm->attention"],
            "note": "synthetic schema rehearsal; no model weights loaded",
        },
        sort_keys=True,
    )
    return {
        "model_id": MODEL_ID,
        "model_revision": "synthetic-schema-rehearsal",
        "tokenizer_revision": "synthetic-schema-rehearsal",
        "prompt_source": "synthetic_horn_schema_prompt_ids",
        "prompt_ids_hash": _sha256_text(prompt_manifest),
        "seed_list": [seed],
        "context_lengths": [2048],
        "dtype": "float32_synthetic_activation",
        "device": "cpu",
        "command": (
            "./venv_arm64/bin/python -m "
            "experimental.horn.phase2.horn_synthetic_h1_gate"
        ),
        "architecture_map_hash": _sha256_text(architecture_manifest),
        "schema_rehearsal": True,
    }


def _boundary_rows(prompt_id: str, prompt_index: int) -> list[dict[str, object]]:
    low_max = 2.0 + 0.01 * prompt_index
    high_max = 8.2 + 0.02 * prompt_index
    low_kurtosis = 3.0 + 0.01 * prompt_index
    high_kurtosis = 5.0 + 0.01 * prompt_index
    return [
        {
            "model_id": MODEL_ID,
            "prompt_id": prompt_id,
            "layer_left": 2,
            "layer_right": 3,
            "direction": "attention->ssm",
            "matched_boundary_direction": "attention->ssm",
            "boundary_index": 0,
            "pre_norm_position": "post_norm",
            "post_norm_position": "pre_norm",
            "max_abs": low_max,
            "rms": 0.75 + 0.001 * prompt_index,
            "kurtosis": low_kurtosis,
            "control_type": "boundary",
        },
        {
            "model_id": MODEL_ID,
            "prompt_id": prompt_id,
            "layer_left": 5,
            "layer_right": 6,
            "direction": "ssm->attention",
            "matched_boundary_direction": "ssm->attention",
            "boundary_index": 1,
            "pre_norm_position": "post_norm",
            "post_norm_position": "pre_norm",
            "max_abs": high_max,
            "rms": 0.95 + 0.001 * prompt_index,
            "kurtosis": high_kurtosis,
            "control_type": "boundary",
        },
    ]


def _non_boundary_rows(prompt_id: str, prompt_index: int) -> list[dict[str, object]]:
    return [
        {
            "model_id": MODEL_ID,
            "prompt_id": prompt_id,
            "layer_left": 8,
            "layer_right": 9,
            "direction": "ssm->ssm",
            "matched_boundary_direction": "attention->ssm",
            "boundary_index": 2,
            "pre_norm_position": "post_norm",
            "post_norm_position": "pre_norm",
            "max_abs": 2.3 + 0.01 * prompt_index,
            "rms": 0.72 + 0.001 * prompt_index,
            "kurtosis": 3.1 + 0.01 * prompt_index,
            "control_type": "non_boundary",
        },
        {
            "model_id": MODEL_ID,
            "prompt_id": prompt_id,
            "layer_left": 10,
            "layer_right": 11,
            "direction": "ssm->ssm",
            "matched_boundary_direction": "ssm->attention",
            "boundary_index": 3,
            "pre_norm_position": "post_norm",
            "post_norm_position": "pre_norm",
            "max_abs": 2.4 + 0.01 * prompt_index,
            "rms": 0.73 + 0.001 * prompt_index,
            "kurtosis": 3.2 + 0.01 * prompt_index,
            "control_type": "non_boundary",
        },
    ]


def _permuted_row(boundary_row: dict[str, object]) -> dict[str, object]:
    flipped = dict(boundary_row)
    flipped["direction"] = (
        "ssm->attention"
        if boundary_row["direction"] == "attention->ssm"
        else "attention->ssm"
    )
    flipped["matched_boundary_direction"] = flipped["direction"]
    flipped["control_type"] = "permuted_direction"
    return flipped


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for prompt_index, prompt_id in enumerate(PROMPT_IDS):
        boundary_rows = _boundary_rows(prompt_id, prompt_index)
        rows.extend(boundary_rows)
        rows.extend(_non_boundary_rows(prompt_id, prompt_index))
        rows.extend(_permuted_row(row) for row in boundary_rows)
    return rows


def _summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# HORN Synthetic H1a Real-Schema Rehearsal",
            "",
            f"Decision: `{summary['decision']}`.",
            "",
            "This packet is intentionally a schema rehearsal and non-promotable.",
            "It validates the real H1a row schema, prompt-paired controls, and",
            "recomputed evaluator summary using synthetic rows.",
            "",
            "Selected evaluator fields:",
            f"- `gate_status`: `{summary['gate_status']}`",
            f"- `prompt_count`: `{summary['prompt_count']}`",
            f"- `selected_h1_metric`: `{summary['selected_h1_metric']}`",
            f"- `selected_h1_direction`: `{summary['selected_h1_direction']}`",
            f"- `selected_h1_ratio`: `{summary['selected_h1_ratio']:.3f}`",
            f"- `permuted_direction_ratio`: `{summary['permuted_direction_ratio']:.3f}`",
            "",
        ]
    )


def run_gate(*, seed: int = 20260506, output_dir: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _rows()
    evaluated = evaluate_horn_h1(rows)
    decision = "SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_HORN_H1A"
    summary = {
        "seed": seed,
        "surface": "horn_h1a_real_schema_rehearsal",
        "decision": decision,
        "row_count": len(rows),
        "rows": rows,
        "claim_boundary": [
            "synthetic-only",
            "not model evidence",
            "schema rehearsal only",
            "not promotable",
            "not GPU evidence",
        ],
        **evaluated,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "raw_rows.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    (output_dir / "config.json").write_text(json.dumps(_config(seed), indent=2, sort_keys=True) + "\n")
    (output_dir / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    (output_dir / "decision.md").write_text(
        f"# HORN Synthetic H1a Real-Schema Rehearsal\n\n`{decision}`\n\n"
        f"- evaluator gate status: `{evaluated['gate_status']}`\n"
        f"- selected metric: `{evaluated['selected_h1_metric']}`\n"
        f"- selected direction: `{evaluated['selected_h1_direction']}`\n"
        f"- selected H1a ratio: `{evaluated['selected_h1_ratio']:.3f}`\n\n"
        "Synthetic CPU rows validate the real packet schema and checker path only. "
        "They do not promote H1a/H1 and are not model, GPU, quality, or systems evidence.\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_gate(seed=args.seed, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
