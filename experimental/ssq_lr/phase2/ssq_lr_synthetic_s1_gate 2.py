"""Synthetic SSQ-LR S1 real-schema rehearsal.

This no-download gate writes rows with the same schema as a real SSQ-LR S1
packet, then marks the decision schema-rehearsal/non-promotable. It validates
the packet checker and evaluator path before running on real hybrid-model state
dumps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from experimental.shared.hybrid_gate_evaluators import evaluate_ssq_lr_s1
from experimental.shared.sensitivity_metrics import kurtosis, max_abs


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "experimental/ssq_lr/phase2/results/ssq_lr_synthetic_s1"
BUCKETS = ("prefill_end", "2k_or_end", "8k_or_end", "final_minus_128")
BUCKET_SCALES = {
    "prefill_end": 0.45,
    "2k_or_end": 0.75,
    "8k_or_end": 1.15,
    "final_minus_128": 1.75,
}
PROMPT_IDS = tuple(f"synthetic_prompt_{index:02d}" for index in range(12))
LAYERS = tuple(range(6))
MODEL_ID = "synthetic-hybrid-ssq-lr-schema-rehearsal"


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tensor_provenance(name: str, shape: list[int]) -> dict[str, object]:
    safe_name = name.replace("/", "_").replace(" ", "_")
    return {
        "tensor_name": name,
        "tensor_source_name": name,
        "tensor_storage_name": f"{safe_name}.pt",
        "tensor_sha256": _sha256_text(f"{name}:{shape}:synthetic-ssq-lr"),
        "tensor_dtype": "torch.float32",
        "tensor_shape": shape,
    }


def _rms(tensor: torch.Tensor) -> float:
    values = tensor.float()
    return float(torch.sqrt(torch.mean(values * values)))


def _outlier_mass(tensor: torch.Tensor) -> float:
    values = tensor.float().reshape(-1).abs()
    if values.numel() == 0:
        return 0.0
    threshold = values.mean() + 3.0 * values.std(unbiased=False)
    return float(torch.mean((values > threshold).float()))


def _state_tensor(*, seed: int, prompt_index: int, layer: int, bucket: str) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(
        seed + 1009 * prompt_index + 9173 * layer
    )
    base = torch.randn(4, 32, generator=generator)
    outlier = torch.zeros_like(base)
    outlier[:, :4] = torch.randn(4, 4, generator=generator) * (2.0 + 0.15 * layer)
    scale = BUCKET_SCALES[bucket]
    layer_gain = 1.0 + 0.05 * layer
    prompt_gain = 1.0 + 0.01 * prompt_index
    return (scale * layer_gain * prompt_gain) * (base + outlier)


def _rows(seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for prompt_index, prompt_id in enumerate(PROMPT_IDS):
        for layer in LAYERS:
            for bucket in BUCKETS:
                state = _state_tensor(
                    seed=seed,
                    prompt_index=prompt_index,
                    layer=layer,
                    bucket=bucket,
                )
                rows.append(
                    {
                        "model_id": MODEL_ID,
                        "model_revision": "synthetic-schema-rehearsal",
                        "prompt_id": prompt_id,
                        "layer": layer,
                        "layer_kind": "mamba2",
                        "position_bucket": bucket,
                        "state_tensor_kind": "mamba2_recurrent_state",
                        "state_shape": list(state.shape),
                        "max_abs": max_abs(state),
                        "rms": _rms(state),
                        "std": float(torch.std(state.float(), unbiased=False)),
                        "kurtosis": kurtosis(state),
                        "outlier_mass": _outlier_mass(state),
                        "control_type": "bf16_no_quant",
                        **_tensor_provenance(
                            f"ssq_lr/{prompt_id}/layer_{layer}/{bucket}",
                            list(state.shape),
                        ),
                    }
                )
    return rows


def _config(seed: int) -> dict[str, object]:
    prompt_manifest = "\n".join(PROMPT_IDS) + "\n"
    architecture_manifest = json.dumps(
        {
            "model_id": MODEL_ID,
            "layers": list(LAYERS),
            "layer_kind": "mamba2",
            "note": "synthetic schema rehearsal; no model weights loaded",
        },
        sort_keys=True,
    )
    return {
        "model_id": MODEL_ID,
        "model_revision": "synthetic-schema-rehearsal",
        "tokenizer_revision": "synthetic-schema-rehearsal",
        "prompt_source": "synthetic_ssq_lr_schema_prompt_ids",
        "prompt_ids_hash": _sha256_text(prompt_manifest),
        "seed_list": [seed],
        "context_lengths": list(BUCKETS),
        "dtype": "float32_synthetic_state",
        "device": "cpu",
        "command": (
            "./venv_arm64/bin/python -m "
            "experimental.ssq_lr.phase2.ssq_lr_synthetic_s1_gate"
        ),
        "architecture_map_hash": _sha256_text(architecture_manifest),
        "schema_rehearsal": True,
    }


def _summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# SSQ-LR Synthetic S1 Real-Schema Rehearsal",
            "",
            f"Decision: `{summary['decision']}`.",
            "",
            "This packet is intentionally a schema rehearsal and non-promotable.",
            "It validates the real S1 row schema, provenance fields, and",
            "recomputed evaluator summary using synthetic CPU tensors.",
            "",
            "Fixture-only evaluator fields, not a promotion decision:",
            f"- `gate_status`: `{summary['gate_status']}` (raw evaluator readout only; "
            f"decision remains `{summary['decision']}`)",
            f"- `prompt_count`: `{summary['prompt_count']}`",
            f"- `ssm_layer_count`: `{summary['ssm_layer_count']}`",
            f"- `selected_s1_ratio`: `{summary['selected_s1_ratio']:.3f}`",
            f"- `selected_s1_ci_low`: `{summary['selected_s1_ci_low']:.3f}`",
            "",
        ]
    )


def run_gate(*, seed: int = 20260506, output_dir: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _rows(seed)
    evaluated = evaluate_ssq_lr_s1(rows)
    decision = "SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_SSQ_LR_S1"

    summary = {
        "seed": seed,
        "surface": "ssq_lr_s1_real_schema_rehearsal",
        "decision": decision,
        "evidence_kind": "schema_rehearsal",
        "schema_rehearsal": True,
        "positive_evidence": False,
        "promotion_gate_pass": False,
        "scrape_exclude": True,
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
        f"# SSQ-LR Synthetic S1 Real-Schema Rehearsal\n\n`{decision}`\n\n"
        f"- evaluator gate status: `{evaluated['gate_status']}` "
        "(raw evaluator readout only; not the packet decision)\n"
        f"- selected S1 ratio: `{evaluated['selected_s1_ratio']:.3f}`\n"
        f"- selected S1 CI low: `{evaluated['selected_s1_ci_low']:.3f}`\n\n"
        "Synthetic CPU tensors validate the real packet schema and checker path only. "
        "They do not promote S1 and are not model, GPU, quality, or systems evidence.\n",
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
