#!/usr/bin/env python3
"""Materialize selected ParoQuant residual columns for DriftRot correction."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.outlier_migrate.phase4 import run_om_phase4_intervention as phase4_runner
from experimental.outlier_migrate.phase9 import check_om_paroquant_baseline as checker
from experimental.outlier_migrate.phase9 import run_om_paroquant_baseline as paro_runner
from experimental.outlier_migrate.phase9 import run_om_phase9_m2_position_conditional as m2_runner
from experimental.shared import run_phase0_branch as shared


SCHEMA_VERSION = "om_driftrot_delta_columns_v1"
DEFAULT_CANDIDATE_POOL = ROOT / "artifacts/rot_resid_correction/residual_candidate_pool_tail4_top8.json"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/rot_resid_correction"


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def select_candidates(
    candidate_pool: dict[str, Any],
    *,
    top_modules: int,
    columns_per_module: int,
    score_key: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for module in candidate_pool["modules"][:top_modules]:
        rows = module.get(score_key)
        if not rows:
            raise ValueError(f"candidate module {module.get('name')} has no {score_key}")
        columns: list[int] = []
        scores_by_column: dict[str, dict[str, float]] = {}
        for row in rows:
            column = int(row["column"])
            if column in columns:
                continue
            columns.append(column)
            scores_by_column[str(column)] = {
                key: float(value)
                for key, value in row.items()
                if key != "column" and isinstance(value, int | float)
            }
            if len(columns) >= columns_per_module:
                break
        if not columns:
            raise ValueError(f"candidate module {module.get('name')} selected zero columns")
        selected.append(
            {
                "name": str(module["name"]),
                "columns": columns,
                "scores_by_column": scores_by_column,
                "source_relative_residual_energy": module.get("relative_residual_energy"),
                "source_count": module.get("count"),
                "source_samples": module.get("samples"),
            }
        )
    return selected


def quantized_delta_columns(
    weight: Any,
    *,
    columns: list[int],
    group_size: int,
    num_rotations: int,
    row_chunk: int,
    scale_clip: tuple[float, float],
) -> tuple[Any, dict[str, Any]]:
    import torch

    if weight.ndim not in {2, 3}:
        raise ValueError(f"weight tensor must be 2D or 3D, got shape {list(weight.shape)}")
    flat = weight.view(-1, weight.shape[-1])
    in_features = int(flat.shape[-1])
    if in_features < group_size or in_features % group_size != 0:
        raise ValueError(f"weight last dimension {in_features} is not aligned to group size {group_size}")
    for column in columns:
        if column < 0 or column >= in_features:
            raise ValueError(f"column {column} outside in_features={in_features}")

    sumsq = torch.zeros(in_features, device=flat.device, dtype=torch.float32)
    count = 0
    for start in range(0, flat.shape[0], row_chunk):
        block = flat[start : start + row_chunk].float()
        sumsq.add_((block * block).sum(dim=0))
        count += int(block.shape[0])
    rms = (sumsq / max(1, count)).sqrt().clamp_min(1e-6)
    median_rms = torch.median(rms)
    scales = (median_rms / rms).clamp(float(scale_clip[0]), float(scale_clip[1])).to(dtype=torch.float32)
    rotations = paro_runner.build_pair_indices(
        in_features=in_features,
        group_size=group_size,
        num_rotations=num_rotations,
        device=flat.device,
        dtype=torch.float32,
    )

    column_index = torch.tensor(columns, device=flat.device, dtype=torch.long)
    delta_cpu = torch.empty((flat.shape[0], len(columns)), dtype=torch.float16, device="cpu")
    for start in range(0, flat.shape[0], row_chunk):
        end = min(start + row_chunk, flat.shape[0])
        block = flat[start:end]
        original_columns = block.float().index_select(1, column_index)
        work = block.float().clone()
        work.mul_(scales.view(1, -1))
        paro_runner.apply_rotations(work, rotations, inverse=False)
        work = paro_runner.quantize_affine_groupwise_4bit(work, group_size)
        paro_runner.apply_rotations(work, rotations, inverse=True)
        work.div_(scales.view(1, -1))
        delta = original_columns - work.index_select(1, column_index)
        delta_cpu[start:end].copy_(delta.to(dtype=torch.float16, device="cpu"))

    delta_shape = list(weight.shape[:-1]) + [len(columns)]
    delta_cpu = delta_cpu.reshape(delta_shape).contiguous()
    metadata = {
        "shape": list(weight.shape),
        "delta_shape": delta_shape,
        "in_features": in_features,
        "row_count": int(flat.shape[0]),
        "columns": columns,
        "group_size": group_size,
        "num_rotations": num_rotations,
        "scale_clip": list(scale_clip),
        "scale_min": float(scales.min().item()),
        "scale_median": float(torch.median(scales).item()),
        "scale_max": float(scales.max().item()),
        "delta_dtype": "float16",
        "working_set_bytes": int(delta_cpu.numel() * 2),
    }
    return delta_cpu, metadata


def materialize_delta_columns(
    *,
    model: Any,
    selected: list[dict[str, Any]],
    run_dir: Path,
    group_size: int,
    num_rotations: int,
    row_chunk: int,
    scale_clip: tuple[float, float],
) -> dict[str, Any]:
    import torch

    modules = dict(model.named_modules())
    delta_dir = run_dir / "delta_columns"
    delta_dir.mkdir(parents=True, exist_ok=False)
    manifest: list[dict[str, Any]] = []
    with torch.no_grad():
        for candidate in selected:
            name = candidate["name"]
            module = modules.get(name)
            if module is None:
                raise RuntimeError(f"candidate module missing from loaded model: {name}")
            weight = getattr(module, "weight", None)
            if not torch.is_tensor(weight):
                raise RuntimeError(f"candidate module has no tensor weight: {name}")
            if phase4_runner.is_tied_lm_head(model, name, module):
                raise RuntimeError(f"candidate module is tied lm head and cannot be corrected: {name}")
            delta_columns, metadata = quantized_delta_columns(
                weight,
                columns=[int(value) for value in candidate["columns"]],
                group_size=group_size,
                num_rotations=num_rotations,
                row_chunk=row_chunk,
                scale_clip=scale_clip,
            )
            file_name = f"{safe_filename(name)}.pt"
            file_path = delta_dir / file_name
            torch.save(
                {
                    "schema_version": f"{SCHEMA_VERSION}_tensor",
                    "module_name": name,
                    "columns": metadata["columns"],
                    "delta_columns": delta_columns,
                    "metadata": metadata,
                    "scores_by_column": candidate["scores_by_column"],
                },
                file_path,
            )
            manifest.append(
                {
                    **metadata,
                    "module_name": name,
                    "artifact_path": str(file_path.relative_to(ROOT)),
                    "artifact_bytes": file_path.stat().st_size,
                    "selection_source_relative_residual_energy": candidate.get("source_relative_residual_energy"),
                    "selection_source_count": candidate.get("source_count"),
                    "selection_source_samples": candidate.get("source_samples"),
                    "score_columns_sample": candidate["columns"][:8],
                }
            )

    total_working_set = sum(int(item["working_set_bytes"]) for item in manifest)
    total_artifact_bytes = sum(int(item["artifact_bytes"]) for item in manifest)
    return {
        "schema_version": f"{SCHEMA_VERSION}_manifest",
        "created_at_utc": shared.utc_now(),
        "module_count": len(manifest),
        "total_working_set_bytes_fp16": total_working_set,
        "total_working_set_mib_fp16": total_working_set / (1024 * 1024),
        "total_artifact_bytes": total_artifact_bytes,
        "modules": manifest,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"delta_columns_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-pool", type=Path, default=DEFAULT_CANDIDATE_POOL)
    parser.add_argument("--model-id", default=checker.MODEL_ID)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--num-rotations", type=int, default=8)
    parser.add_argument("--row-chunk", type=int, default=256)
    parser.add_argument("--scale-clip-min", type=float, default=0.5)
    parser.add_argument("--scale-clip-max", type=float, default=2.0)
    parser.add_argument("--top-modules", type=int, default=8)
    parser.add_argument("--columns-per-module", type=int, default=32)
    parser.add_argument("--score-key", choices=["top_score_columns", "top_residual_columns"], default="top_score_columns")
    args = parser.parse_args(argv)

    run_dir = args.output_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    candidate_pool = json.loads(args.candidate_pool.read_text(encoding="utf-8"))
    selected = select_candidates(
        candidate_pool,
        top_modules=args.top_modules,
        columns_per_module=args.columns_per_module,
        score_key=args.score_key,
    )
    model_provenance = m2_runner.resolve_model_snapshot_light(args.model_id)
    if model_provenance.get("hf_snapshot_commit") != checker.MODEL_SNAPSHOT:
        raise SystemExit("Granite model snapshot missing or mismatch")
    model, tokenizer, device = shared.load_model_and_tokenizer(model_provenance, dtype_name=args.dtype, device_name=args.device)
    del tokenizer, device
    scale_clip = (float(args.scale_clip_min), float(args.scale_clip_max))
    manifest = materialize_delta_columns(
        model=model,
        selected=selected,
        run_dir=run_dir,
        group_size=args.group_size,
        num_rotations=args.num_rotations,
        row_chunk=args.row_chunk,
        scale_clip=scale_clip,
    )
    config = {
        "schema_version": f"{SCHEMA_VERSION}_config",
        "model_id": args.model_id,
        "model_snapshot_commit": model_provenance.get("hf_snapshot_commit"),
        "candidate_pool": str(args.candidate_pool),
        "candidate_pool_sha256": shared.file_sha256(args.candidate_pool),
        "group_size": args.group_size,
        "num_rotations": args.num_rotations,
        "row_chunk": args.row_chunk,
        "scale_clip": list(scale_clip),
        "top_modules": args.top_modules,
        "columns_per_module": args.columns_per_module,
        "score_key": args.score_key,
        "dtype": args.dtype,
    }
    metrics = {
        "schema_version": f"{SCHEMA_VERSION}_metrics",
        "created_at_utc": shared.utc_now(),
        "decision_surface": "artifact_materialization_only",
        "module_count": manifest["module_count"],
        "total_working_set_mib_fp16": manifest["total_working_set_mib_fp16"],
        "total_artifact_bytes": manifest["total_artifact_bytes"],
    }
    decision = {
        "schema_version": f"{SCHEMA_VERSION}_decision",
        "decision": "DELTA_COLUMNS_MATERIALIZED_SMOKE_RUNNER_NEXT",
        "promote_to_gpu": False,
        "reason": "Selected rotated-basis residual columns are now available; a correction wrapper and smoke scorer are still required.",
        "not_final_method_claim": True,
    }
    shared.write_json(run_dir / "config.json", config)
    shared.write_json(run_dir / "model_provenance.json", model_provenance)
    shared.write_json(run_dir / "delta_columns_manifest.json", manifest)
    shared.write_json(run_dir / "metrics.json", metrics)
    shared.write_json(run_dir / "decision.json", decision)
    (run_dir / "command.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + " ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "command.sh").chmod(0o755)
    shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
    print(json.dumps({"run_dir": str(run_dir), "metrics": metrics, "decision": decision}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
