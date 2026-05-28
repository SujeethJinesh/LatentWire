#!/usr/bin/env python3
"""Collect ParoQuant residual-column caches for DriftRot residual correction."""

from __future__ import annotations

import argparse
import json
import math
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


SCHEMA_VERSION = "om_driftrot_residual_cache_v1"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/rot_resid_correction"


def quantize_weight_and_measure_residual(
    weight: Any,
    *,
    group_size: int,
    num_rotations: int,
    row_chunk: int,
    scale_clip: tuple[float, float],
    topn: int,
) -> dict[str, Any]:
    import torch

    if weight.ndim not in {2, 3}:
        return {"reason": "weight tensor is not 2D or expert-bank 3D", "shape": list(weight.shape)}
    try:
        flat = weight.view(-1, weight.shape[-1])
    except RuntimeError:
        return {"reason": "weight tensor is not contiguous in last dimension", "shape": list(weight.shape)}

    in_features = int(flat.shape[-1])
    if in_features < group_size or in_features % group_size != 0:
        return {
            "reason": "last dimension is not group aligned for ParoQuant residual cache",
            "shape": list(weight.shape),
            "in_features": in_features,
            "group_size": group_size,
        }

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

    delta_col_norm_sq = torch.zeros(in_features, device=flat.device, dtype=torch.float64)
    fp_col_norm_sq = torch.zeros(in_features, device=flat.device, dtype=torch.float64)
    for start in range(0, flat.shape[0], row_chunk):
        block = flat[start : start + row_chunk]
        original = block.float().clone()
        work = block.float()
        work.mul_(scales.view(1, -1))
        paro_runner.apply_rotations(work, rotations, inverse=False)
        work = paro_runner.quantize_affine_groupwise_4bit(work, group_size)
        paro_runner.apply_rotations(work, rotations, inverse=True)
        work.div_(scales.view(1, -1))
        delta = original - work
        delta_col_norm_sq.add_((delta.double() * delta.double()).sum(dim=0))
        fp_col_norm_sq.add_((original.double() * original.double()).sum(dim=0))

    topk = min(int(topn), in_features)
    values, indices = torch.topk(delta_col_norm_sq.float(), k=topk, largest=True, sorted=True)
    total_residual = float(delta_col_norm_sq.sum().item())
    total_fp = float(fp_col_norm_sq.sum().item())
    top_residual = float(values.double().sum().item())
    return {
        "shape": list(weight.shape),
        "mode": "scaled_pairwise_rotation_groupwise_affine_int4_folded_back_residual_norms",
        "in_features": in_features,
        "row_count": int(flat.shape[0]),
        "group_size": group_size,
        "num_rotations": num_rotations,
        "scale_clip": list(scale_clip),
        "scale_min": float(scales.min().item()),
        "scale_median": float(torch.median(scales).item()),
        "scale_max": float(scales.max().item()),
        "residual_norm_sq_total": total_residual,
        "fp_norm_sq_total": total_fp,
        "relative_residual_energy": None if total_fp == 0.0 else total_residual / total_fp,
        "topn": topk,
        "topn_residual_energy_fraction": None if total_residual == 0.0 else top_residual / total_residual,
        "top_columns": [
            {"column": int(index), "delta_norm_sq": float(value)}
            for index, value in zip(indices.detach().cpu().tolist(), values.detach().cpu().tolist(), strict=True)
        ],
    }


def collect_residual_cache(
    *,
    model: Any,
    group_size: int,
    num_rotations: int,
    row_chunk: int,
    scale_clip: tuple[float, float],
    topn: int,
) -> dict[str, Any]:
    import torch

    measured: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    with torch.no_grad():
        for name, module in model.named_modules():
            weight = getattr(module, "weight", None)
            if not torch.is_tensor(weight):
                continue
            if phase4_runner.is_tied_lm_head(model, name, module):
                skipped.append({"name": name, "reason": "tied input/output embedding excluded", "shape": list(weight.shape)})
                continue
            item = quantize_weight_and_measure_residual(
                weight,
                group_size=group_size,
                num_rotations=num_rotations,
                row_chunk=row_chunk,
                scale_clip=scale_clip,
                topn=topn,
            )
            item["name"] = name
            if "reason" in item:
                skipped.append(item)
            else:
                measured.append(item)
    total_residual = sum(float(item["residual_norm_sq_total"]) for item in measured)
    return {
        "schema_version": f"{SCHEMA_VERSION}_residual_cache",
        "created_at_utc": shared.utc_now(),
        "measured_tensor_count": len(measured),
        "skipped_tensor_count": len(skipped),
        "total_residual_norm_sq": total_residual,
        "tensors": measured,
        "skipped_tensors": skipped,
    }


def build_summary(cache: dict[str, Any], *, protected_columns_per_tensor: int, dtype_bytes: int) -> dict[str, Any]:
    tensors = cache["tensors"]
    ranked = sorted(tensors, key=lambda item: float(item["residual_norm_sq_total"]), reverse=True)
    total_working_set_bytes = 0
    rows: list[dict[str, Any]] = []
    for item in ranked[:25]:
        protected = min(protected_columns_per_tensor, int(item["in_features"]))
        bytes_for_tensor = int(item["row_count"]) * protected * dtype_bytes
        total_working_set_bytes += bytes_for_tensor
        rows.append(
            {
                "name": item["name"],
                "shape": item["shape"],
                "relative_residual_energy": item["relative_residual_energy"],
                "topn_residual_energy_fraction": item["topn_residual_energy_fraction"],
                "suggested_protected_columns": protected,
                "working_set_bytes_fp16": bytes_for_tensor,
                "top_columns_sample": item["top_columns"][:10],
            }
        )
    return {
        "schema_version": f"{SCHEMA_VERSION}_summary",
        "created_at_utc": shared.utc_now(),
        "ranked_by": "residual_norm_sq_total",
        "protected_columns_per_tensor": protected_columns_per_tensor,
        "dtype_bytes": dtype_bytes,
        "top_tensor_count": len(rows),
        "top_tensor_working_set_bytes_fp16": total_working_set_bytes,
        "top_tensor_working_set_mib_fp16": total_working_set_bytes / (1024 * 1024),
        "top_tensors": rows,
        "energy_estimate_status": "not_measured",
        "energy_estimate_requirement": "Report pJ/token after a kernel or profiler-backed reference path exists.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"driftrot_residual_cache_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id", default=checker.MODEL_ID)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--num-rotations", type=int, default=8)
    parser.add_argument("--row-chunk", type=int, default=256)
    parser.add_argument("--scale-clip-min", type=float, default=0.5)
    parser.add_argument("--scale-clip-max", type=float, default=2.0)
    parser.add_argument("--topn", type=int, default=128)
    parser.add_argument("--protected-columns-per-tensor", type=int, default=64)
    args = parser.parse_args(argv)

    run_dir = args.output_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    model_provenance = m2_runner.resolve_model_snapshot_light(args.model_id)
    if model_provenance.get("hf_snapshot_commit") != checker.MODEL_SNAPSHOT:
        raise SystemExit("Granite model snapshot missing or mismatch")
    model, tokenizer, device = shared.load_model_and_tokenizer(model_provenance, dtype_name=args.dtype, device_name=args.device)
    del tokenizer, device
    scale_clip = (float(args.scale_clip_min), float(args.scale_clip_max))
    cache = collect_residual_cache(
        model=model,
        group_size=args.group_size,
        num_rotations=args.num_rotations,
        row_chunk=args.row_chunk,
        scale_clip=scale_clip,
        topn=args.topn,
    )
    summary = build_summary(cache, protected_columns_per_tensor=args.protected_columns_per_tensor, dtype_bytes=2)
    shared.write_json(
        run_dir / "config.json",
        {
            "schema_version": f"{SCHEMA_VERSION}_config",
            "model_id": args.model_id,
            "model_snapshot_commit": model_provenance.get("hf_snapshot_commit"),
            "group_size": args.group_size,
            "num_rotations": args.num_rotations,
            "scale_clip": list(scale_clip),
            "topn": args.topn,
            "protected_columns_per_tensor": args.protected_columns_per_tensor,
        },
    )
    shared.write_json(run_dir / "model_provenance.json", model_provenance)
    shared.write_json(run_dir / "residual_column_cache.json", cache)
    shared.write_json(run_dir / "summary.json", summary)
    (run_dir / "command.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + " ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "command.sh").chmod(0o755)
    shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
    print(json.dumps({"run_dir": str(run_dir), "summary": summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
