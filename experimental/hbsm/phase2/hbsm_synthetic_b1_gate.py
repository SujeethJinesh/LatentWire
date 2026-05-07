"""Synthetic HBSM B1 real-schema rehearsal.

This no-download gate writes the same row schema expected from a real hybrid
layer-sensitivity sweep, then marks the decision schema-rehearsal/non-promotable.
It validates prompt-to-layer aggregation, controls, and summary recomputation
before running B1 on live hybrid model outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experimental.shared.hybrid_gate_evaluators import evaluate_hbsm_b1


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "experimental/hbsm/phase2/results/hbsm_synthetic_b1"
MODEL_ID = "synthetic-hybrid-hbsm-schema-rehearsal"
PROMPT_IDS = tuple(f"synthetic_prompt_{index:02d}" for index in range(12))
LAYERS = tuple(range(40))
BOUNDARY_LAYERS = set(range(30, 40))
TOP_DECILE_LAYERS = {36, 37, 38, 39}
RANDOM_TOP_DECILE_LAYERS = {0, 1, 2, 3}
CONTROL_TYPES = (
    "perturbation_off",
    "random_flags",
    "layer_index",
    "parameter_count_norm",
    "kl_lens_rank",
    "activation_outlier",
)


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _config(seed: int) -> dict[str, object]:
    prompt_manifest = "\n".join(PROMPT_IDS) + "\n"
    architecture_manifest = json.dumps(
        {
            "model_id": MODEL_ID,
            "layers": list(LAYERS),
            "boundary_layers": sorted(BOUNDARY_LAYERS),
            "note": "synthetic schema rehearsal; no model weights loaded",
        },
        sort_keys=True,
    )
    return {
        "model_id": MODEL_ID,
        "model_revision": "synthetic-schema-rehearsal",
        "tokenizer_revision": "synthetic-schema-rehearsal",
        "prompt_source": "synthetic_hbsm_schema_prompt_ids",
        "prompt_ids_hash": _sha256_text(prompt_manifest),
        "seed_list": [seed],
        "context_lengths": [2048],
        "dtype": "float32_synthetic_sensitivity",
        "device": "cpu",
        "command": (
            "./venv_arm64/bin/python -m "
            "experimental.hbsm.phase2.hbsm_synthetic_b1_gate"
        ),
        "architecture_map_hash": _sha256_text(architecture_manifest),
        "schema_rehearsal": True,
    }


def _primary_row(prompt_id: str, prompt_index: int, layer: int) -> dict[str, object]:
    boundary = layer in BOUNDARY_LAYERS
    top_decile = layer in TOP_DECILE_LAYERS
    random_top_decile = layer in RANDOM_TOP_DECILE_LAYERS
    boundary_boost = 0.18 if boundary else 0.02
    drift = boundary_boost + 0.006 * layer + 0.0003 * prompt_index
    if top_decile:
        drift += 0.18
    cheap_predictor = (2.0 if boundary else 0.5) + 0.15 * layer + 0.001 * prompt_index
    return {
        "model_id": MODEL_ID,
        "prompt_id": prompt_id,
        "layer": layer,
        "boundary_flag": boundary,
        "precision_perturbation": "mxfp4_e2m1",
        "kl_or_nll_drift": drift,
        "cheap_predictor": cheap_predictor,
        "parameter_count": 1024 + 16 * layer,
        "weight_norm": 1.0 + 0.05 * layer,
        "top_decile_flag": top_decile,
        "random_top_decile": random_top_decile,
        "train_test_split": "train" if layer % 2 == 0 else "test",
        "control_type": "boundary_only",
    }


def _control_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for control_type in CONTROL_TYPES:
        for layer in LAYERS:
            boundary = layer in BOUNDARY_LAYERS
            drift = 0.0 if control_type == "perturbation_off" else 0.01 * (layer + 1)
            if control_type == "kl_lens_rank":
                drift = 0.04 + 0.015 * layer
            if control_type == "activation_outlier":
                drift = 0.03 + 0.01 * layer
            rows.append(
                {
                    "model_id": MODEL_ID,
                    "prompt_id": f"control_{control_type}_{layer}",
                    "layer": layer,
                    "boundary_flag": boundary,
                    "precision_perturbation": "mxfp4_e2m1",
                    "kl_or_nll_drift": drift,
                    "cheap_predictor": 0.5 + 0.2 * layer,
                    "parameter_count": 1024 + layer,
                    "weight_norm": 1.0 + 0.01 * layer,
                    "top_decile_flag": False,
                    "random_top_decile": False,
                    "train_test_split": "train" if layer % 2 == 0 else "test",
                    "control_type": control_type,
                }
            )
    return rows


def _rows() -> list[dict[str, object]]:
    rows = [
        _primary_row(prompt_id, prompt_index, layer)
        for prompt_index, prompt_id in enumerate(PROMPT_IDS)
        for layer in LAYERS
    ]
    rows.extend(_control_rows())
    return rows


def _summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# HBSM Synthetic B1 Real-Schema Rehearsal",
            "",
            f"Decision: `{summary['decision']}`.",
            "",
            "This packet is intentionally a schema rehearsal and non-promotable.",
            "It validates the real B1 row schema, controls, prompt-to-layer",
            "aggregation, and recomputed evaluator summary using synthetic rows.",
            "",
            "Selected evaluator fields:",
            f"- `gate_status`: `{summary['gate_status']}`",
            f"- `primary_row_count`: `{summary['primary_row_count']}`",
            f"- `scoring_layer_count`: `{summary['scoring_layer_count']}`",
            f"- `boundary_top_decile_enrichment`: `{summary['boundary_top_decile_enrichment']:.3f}`",
            f"- `cheap_predictor_spearman`: `{summary['cheap_predictor_spearman']:.3f}`",
            "",
        ]
    )


def run_gate(*, seed: int = 20260506, output_dir: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _rows()
    evaluated = evaluate_hbsm_b1(rows)
    decision = "SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_HBSM_B1"
    summary = {
        "seed": seed,
        "surface": "hbsm_b1_real_schema_rehearsal",
        "decision": decision,
        "evidence_kind": "schema_rehearsal",
        "promotable": False,
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
        f"# HBSM Synthetic B1 Real-Schema Rehearsal\n\n`{decision}`\n\n"
        f"- evaluator gate status: `{evaluated['gate_status']}`\n"
        f"- boundary top-decile enrichment: `{evaluated['boundary_top_decile_enrichment']:.3f}`\n"
        f"- cheap-predictor Spearman: `{evaluated['cheap_predictor_spearman']:.3f}`\n\n"
        "Synthetic CPU rows validate the real packet schema and checker path only. "
        "They do not promote B1 and are not model, GPU, quality, or systems evidence.\n",
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
