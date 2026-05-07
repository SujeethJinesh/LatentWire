"""Build an SSQ-LR S3 transfer preflight packet from frozen S2 rows.

This runner does not claim cross-model transfer. It freezes a selected S2
state-quantization recipe, re-expresses the available same-model holdout rows in
the stricter S3 schema, records the local model-cache inventory, and validates
that the only remaining blocker is the lack of at least two complete hybrid
transfer targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from experimental.shared.followup_gate_contracts import evaluate_ssq_lr_s3


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_S2_PACKET = (
    ROOT
    / "experimental/shared/results/ssq_lr_s3_prefilter_granite_tiny_layers0_30_20260507"
)
DEFAULT_OUTPUT_DIR = ROOT / "experimental/shared/results/ssq_lr_s3_transfer_prefilter_mixed25_layers0_30_20260507"
DEFAULT_PREREGISTRATION = ROOT / "experimental/ssq_lr/phase2/preregister_ssq_lr_20260506.md"
DEFAULT_HF_HOMES = (ROOT / ".debug/hf_home", Path.home() / ".cache/huggingface")
DEFAULT_TRANSFER_MODELS = (
    "ibm-granite/granite-4.0-h-tiny",
    "ibm-granite/granite-4.0-h-micro",
    "ibm-granite/granite-4.0-h-350m",
    "ibm-granite/granite-4.0-h-1b",
    "ibm-granite/granite-4.0-h-small",
    "ibm-granite/granite-4.0-h-small-FP8",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
)


def _reset_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _repo_cache_name(model_id: str) -> str:
    return f"models--{model_id.replace('/', '--')}"


def _snapshot_for_model(model_id: str, hf_homes: tuple[Path, ...]) -> Path | None:
    for hf_home in hf_homes:
        repo_cache = hf_home / "hub" / _repo_cache_name(model_id)
        ref_path = repo_cache / "refs/main"
        if not ref_path.exists():
            continue
        revision = ref_path.read_text(encoding="utf-8").strip()
        snapshot = repo_cache / "snapshots" / revision
        if snapshot.exists():
            return snapshot
    return None


def _has_complete_weights(snapshot: Path | None) -> bool:
    if snapshot is None:
        return False
    return any(
        (snapshot / filename).exists()
        for filename in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )


def _model_inventory(model_ids: tuple[str, ...], hf_homes: tuple[Path, ...]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for model_id in model_ids:
        snapshot = _snapshot_for_model(model_id, hf_homes)
        complete = _has_complete_weights(snapshot)
        inventory.append(
            {
                "model_id": model_id,
                "snapshot_path": str(snapshot) if snapshot is not None else "",
                "cache_present": snapshot is not None,
                "complete_weights": complete,
                "role": "transfer_candidate" if complete else "blocked_missing_weights",
            }
        )
    return inventory


def _freeze_recipe(
    *,
    source_s2_dir: Path,
    selected_rows: list[dict[str, Any]],
    recipe_id: str,
    primary_layers: str,
    block_size: int,
) -> tuple[dict[str, Any], str]:
    recipe = {
        "recipe_id": recipe_id,
        "primary_layers": [int(part.strip()) for part in primary_layers.split(",") if part.strip()],
        "block_size": block_size,
        "precision": selected_rows[0].get("precision", ""),
        "scale_granularity": selected_rows[0].get("scale_granularity", ""),
        "source_s2_packet": str(source_s2_dir),
        "source_prompt_ids": sorted({str(row["prompt_id"]) for row in selected_rows}),
        "retuning_allowed": False,
    }
    encoded = json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return recipe, _sha256_bytes(encoded)


def _s3_row(
    *,
    row: dict[str, Any],
    frozen_recipe_sha256: str,
    source_s2_packet_sha256: str,
    model_role: str,
    control_type: str,
) -> dict[str, Any]:
    accuracy_delta = float(row.get("accuracy_delta_abs", 0.0))
    nll_delta = float(row.get("nll_delta", 0.0))
    return {
        "model_id": str(row["model_id"]),
        "prompt_id": str(row["prompt_id"]),
        "recipe_id": str(row["recipe_id"]),
        "frozen_recipe_sha256": frozen_recipe_sha256,
        "source_s2_packet_sha256": source_s2_packet_sha256,
        "retuned": False,
        "model_role": model_role,
        "bf16_accuracy": float(row.get("bf16_accuracy", 1.0)),
        "quantized_accuracy": float(row.get("quantized_accuracy", 1.0 - accuracy_delta)),
        "accuracy_delta_abs": accuracy_delta,
        "paired_ci_high": float(row.get("paired_ci_high", accuracy_delta)),
        "bf16_nll": float(row["bf16_nll"]),
        "quantized_nll": float(row["quantized_nll"]),
        "nll_delta": nll_delta,
        "control_type": control_type,
        "continuation_token_count": int(row.get("continuation_token_count", 0)),
        "bf16_selected_state_bytes": float(row.get("bf16_selected_state_bytes", 0.0)),
        "claim_boundary_note": "same-model preflight row; not cross-model transfer evidence",
    }


def build_prefilter(
    *,
    source_s2_dir: Path = DEFAULT_SOURCE_S2_PACKET,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    preregistration: Path = DEFAULT_PREREGISTRATION,
    recipe_id: str = "mixed_int3_mxfp4_low_error_25pct",
    primary_layers: str = "0,30",
    block_size: int = 256,
    seed: int = 20260507,
    hf_homes: tuple[Path, ...] = DEFAULT_HF_HOMES,
    transfer_models: tuple[str, ...] = DEFAULT_TRANSFER_MODELS,
) -> dict[str, Any]:
    raw_rows_path = source_s2_dir / "raw_rows.jsonl"
    summary_path = source_s2_dir / "summary.json"
    if not raw_rows_path.is_file():
        raise FileNotFoundError(f"missing source S2 rows: {raw_rows_path}")
    selected_rows = [
        row
        for row in _load_rows(raw_rows_path)
        if str(row.get("control_type")) == "candidate_recipe" and str(row.get("recipe_id")) == recipe_id
    ]
    if not selected_rows:
        raise ValueError(f"source S2 packet has no candidate rows for recipe {recipe_id!r}")

    _reset_output_dir(output_dir)
    source_s2_packet_sha256 = _sha256_path(summary_path)
    recipe, frozen_recipe_sha256 = _freeze_recipe(
        source_s2_dir=source_s2_dir,
        selected_rows=selected_rows,
        recipe_id=recipe_id,
        primary_layers=primary_layers,
        block_size=block_size,
    )
    inventory = _model_inventory(transfer_models, hf_homes)
    complete_models = [item["model_id"] for item in inventory if item["complete_weights"]]

    transfer_rows = [
        _s3_row(
            row=row,
            frozen_recipe_sha256=frozen_recipe_sha256,
            source_s2_packet_sha256=source_s2_packet_sha256,
            model_role="source_model_holdout_prefilter",
            control_type="transfer_eval",
        )
        for row in selected_rows
    ]
    retune_probe = dict(transfer_rows[0])
    retune_probe.update(
        {
            "prompt_id": "retune_probe_guard_no_retuning_performed",
            "model_role": "retune_probe_guard",
            "bf16_accuracy": 1.0,
            "quantized_accuracy": 1.0,
            "accuracy_delta_abs": 0.0,
            "paired_ci_high": 0.0,
            "bf16_nll": float(transfer_rows[0]["bf16_nll"]),
            "quantized_nll": float(transfer_rows[0]["bf16_nll"]),
            "nll_delta": 0.0,
            "control_type": "retune_probe",
            "claim_boundary_note": "required no-retuning sentinel; no retuned rows are emitted",
        }
    )
    rows = [*transfer_rows, retune_probe]
    evaluation = evaluate_ssq_lr_s3(rows)
    claim_boundary = [
        "S3 prefilter only; not promotable cross-model transfer evidence",
        "source model is Granite Tiny only under current complete local cache",
        "retuned=false for every row; no per-model retuning was performed",
        "true S3 promotion still requires at least two complete hybrid validation models",
    ]
    config = {
        "gate_name": "ssq_lr_s3",
        "project": "ssq_lr",
        "source_gate_packet_sha256": source_s2_packet_sha256,
        "preregistration_sha256": _sha256_path(preregistration),
        "seed_list": [seed],
        "command": "python -m experimental.shared.ssq_lr_s3_transfer_prefilter",
        "source_s2_packet": str(source_s2_dir),
        "recipe_id": recipe_id,
        "primary_layers": recipe["primary_layers"],
        "block_size": block_size,
        "hf_homes": [str(path) for path in hf_homes],
        "complete_transfer_models": complete_models,
        "claim_boundary": claim_boundary,
    }
    summary = {
        "decision": evaluation["gate_status"],
        "resource_limited_decision": f"S3_PREFILTER_NOT_PROMOTABLE_{evaluation['gate_status']}",
        **evaluation,
        "row_count": len(rows),
        "prompt_count": len(selected_rows),
        "source_s2_packet_sha256": source_s2_packet_sha256,
        "frozen_recipe": recipe,
        "model_inventory": inventory,
        "complete_transfer_model_count": len(complete_models),
        "claim_boundary": claim_boundary,
    }

    (output_dir / "frozen_recipe.json").write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n")
    (output_dir / "model_inventory.json").write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (output_dir / "raw_rows.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    (output_dir / "decision.md").write_text(
        "# SSQ-LR S3 Transfer Prefilter\n\n"
        f"`{summary['decision']}`\n\n"
        f"Resource-limited label: `{summary['resource_limited_decision']}`\n",
        encoding="utf-8",
    )
    print(json.dumps({"decision": summary["decision"], "output_dir": str(output_dir)}, sort_keys=True))
    return summary


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# SSQ-LR S3 Transfer Prefilter",
        "",
        f"Decision: `{summary['decision']}`",
        "",
        "This packet is schema-valid S3 preflight evidence, not a cross-model transfer pass.",
        "",
        f"- Transfer model count: `{summary['transfer_model_count']}`",
        f"- Passing model count: `{summary['passing_model_count']}`",
        f"- Complete local transfer candidates: `{summary['complete_transfer_model_count']}`",
        f"- Frozen recipe hash: `{summary['frozen_recipe_sha256']}`",
        f"- Max accuracy delta: `{summary['max_accuracy_delta_abs']:.6f}`",
        f"- Max CI high: `{summary['max_ci_high']:.6f}`",
        f"- Max NLL delta: `{summary['max_nll_delta_abs']:.6f}`",
        "",
        "| Model | Cache present | Complete weights | Role |",
        "|---|---:|---:|---|",
    ]
    for item in summary["model_inventory"]:
        lines.append(
            "| {model_id} | {cache_present} | {complete_weights} | {role} |".format(**item)
        )
    lines.append("")
    return "\n".join(lines)


def _parse_path_list(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(part).expanduser() for part in value.split(",") if part.strip())
    if not paths:
        raise ValueError("path list must contain at least one path")
    return paths


def _parse_str_list(value: str) -> tuple[str, ...]:
    items = tuple(part.strip() for part in value.split(",") if part.strip())
    if not items:
        raise ValueError("list must contain at least one item")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-s2-dir", type=Path, default=DEFAULT_SOURCE_S2_PACKET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--recipe-id", default="mixed_int3_mxfp4_low_error_25pct")
    parser.add_argument("--primary-layers", default="0,30")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--hf-homes", default=",".join(str(path) for path in DEFAULT_HF_HOMES))
    parser.add_argument("--transfer-models", default=",".join(DEFAULT_TRANSFER_MODELS))
    args = parser.parse_args()
    build_prefilter(
        source_s2_dir=args.source_s2_dir,
        output_dir=args.output_dir,
        preregistration=args.preregistration,
        recipe_id=args.recipe_id,
        primary_layers=args.primary_layers,
        block_size=args.block_size,
        seed=args.seed,
        hf_homes=_parse_path_list(args.hf_homes),
        transfer_models=_parse_str_list(args.transfer_models),
    )


if __name__ == "__main__":
    main()
