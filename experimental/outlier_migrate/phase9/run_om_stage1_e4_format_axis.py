#!/usr/bin/env python3
"""Prepare or package Stage 1 E4 format-axis triangulation metrics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.outlier_migrate.phase9 import check_om_stage1_e4_format_axis as checker
from experimental.shared import run_phase0_branch as shared


SCHEMA_VERSION = checker.SCHEMA_VERSION
DEFAULT_RESULTS_DIR = checker.RESULTS_DIR
DEFAULT_AIME_FILE = ROOT / "experimental/shared/prompts/aime_2025_indices_0_11.jsonl"
MODEL_ID = "ibm-granite/granite-4.0-h-small"
MATH_DATASET = "HuggingFaceH4/MATH-500"


def load_json(path: Path) -> Any:
    """Load a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_prompt_rows(path: Path, *, source: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Load prompt rows from a local JSONL file."""
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip() or (limit is not None and len(rows) >= limit):
            continue
        item = json.loads(line)
        prompt = item.get("prompt") or item.get("problem") or item.get("question")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{path}: row {index} has no prompt/problem/question")
        rows.append(
            {
                "index": int(item.get("index", len(rows))),
                "prompt_id": str(item.get("prompt_id", item.get("id", len(rows)))),
                "prompt": prompt,
                "answer": item.get("answer"),
                "source_dataset": source,
                "source_file": str(path),
            }
        )
    if not rows:
        raise ValueError(f"{path}: no prompts loaded")
    return rows


def load_math_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load or explicitly defer MATH-500 prompt selection."""
    if args.skip_math_download:
        return [], {"status": "SKIPPED_DRY_RUN", "reason": "explicit --skip-math-download"}
    if args.math_file:
        return load_prompt_rows(args.math_file, source="math_500", limit=args.math_count), {"status": "LOCAL_FILE", "path": str(args.math_file)}
    from datasets import load_dataset

    ds = load_dataset(args.math_dataset, split=args.math_split)
    rows: list[dict[str, Any]] = []
    for item in ds:
        if len(rows) >= args.math_count:
            break
        prompt = item.get("problem") or item.get("question")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        rows.append(
            {
                "index": len(rows),
                "prompt_id": f"math_500_{len(rows)}",
                "prompt": prompt,
                "answer": item.get("answer"),
                "source_dataset": "math_500",
                "source_dataset_id": args.math_dataset,
                "source_split": args.math_split,
            }
        )
    if len(rows) < args.math_count:
        raise ValueError(f"{args.math_dataset}: loaded {len(rows)} rows, expected {args.math_count}")
    return rows, {"status": "HF_DATASET", "dataset": args.math_dataset, "split": args.math_split}


def prompt_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Build the E4 benchmark prompt manifest."""
    aime_rows = load_prompt_rows(args.aime_file, source="aime_2025", limit=args.aime_count)
    math_rows, math_source = load_math_rows(args)
    prompts = {"aime_2025": aime_rows, "math_500": math_rows}
    payload = json.dumps(prompts, sort_keys=True).encode("utf-8")
    return {
        "schema_version": f"{SCHEMA_VERSION}_prompt_manifest",
        "created_at_utc": shared.utc_now(),
        "benchmarks": {"aime_2025": len(aime_rows), "math_500": len(math_rows)},
        "aime_source_file": str(args.aime_file),
        "math_source": math_source,
        "prompt_sha256": shared.bytes_sha256(payload),
        "prompts": prompts,
    }


def module_available(name: str) -> bool:
    """Return whether a Python module can be found without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def kernel_probe(selected: list[str]) -> dict[str, Any]:
    """Probe exact format support without substituting fallbacks."""
    import torch

    probes: dict[str, Any] = {}
    cuda_ok = torch.cuda.is_available()
    vllm_ok = module_available("vllm")
    fp4_modules = [
        "vllm.model_executor.layers.quantization.utils.fp4_utils",
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a16_fp4",
        "transformer_engine.pytorch",
    ]
    fp4_hits = [name for name in fp4_modules if module_available(name)]
    for format_id in selected:
        if format_id == "fp8":
            ok = cuda_ok and vllm_ok and hasattr(torch, "float8_e4m3fn")
            reason = None if ok else "requires CUDA, vLLM, and torch.float8_e4m3fn support"
        elif format_id == "w4a16":
            ok = module_available("experimental.outlier_migrate.phase4.run_om_phase4_intervention")
            reason = None if ok else "local W4A16 dequantized INT4 implementation is unavailable"
        elif format_id in {"nvfp4_w4a16", "nvfp4_w4a4"}:
            ok = cuda_ok and vllm_ok and bool(fp4_hits)
            reason = None if ok else "requires CUDA, vLLM, and an importable NVFP4/FP4 runtime kernel module"
        else:
            raise ValueError(f"unknown format {format_id}")
        probes[format_id] = {
            "available": bool(ok),
            "backend": "runtime_probe",
            "reason": reason,
            "detected_fp4_modules": fp4_hits if format_id.startswith("nvfp4") else [],
        }
    return {
        "schema_version": f"{SCHEMA_VERSION}_kernel_probe",
        "created_at_utc": shared.utc_now(),
        "format_probes": probes,
        "probe_policy": "unavailable formats must be SKIPPED_INFRA; no simulated substitution",
    }


def normalize_metrics(metrics_path: Path, selected: list[str], probe: dict[str, Any]) -> dict[str, Any]:
    """Merge external benchmark metrics with format support probes."""
    raw = load_json(metrics_path)
    rows = raw.get("format_results", raw)
    if not isinstance(rows, dict):
        raise ValueError("--metrics-json must contain a format_results object or a format-keyed object")
    normalized: dict[str, Any] = {}
    probes = probe["format_probes"]
    for format_id in checker.FORMATS:
        if format_id not in selected:
            normalized[format_id] = {"status": "PENDING_GPU_RUN", "pending_reason": "not selected in this invocation"}
            continue
        if not probes[format_id]["available"]:
            normalized[format_id] = {"status": "SKIPPED_INFRA", "skip_reason": probes[format_id]["reason"]}
            continue
        row = rows.get(format_id)
        if row is None:
            normalized[format_id] = {"status": "PENDING_GPU_RUN", "pending_reason": "no metrics supplied for supported format"}
            continue
        status = row.get("status", "COMPLETED")
        if status == "SKIPPED_INFRA":
            normalized[format_id] = {"status": status, "skip_reason": row.get("skip_reason", "external metrics marked skipped")}
        elif status == "COMPLETED":
            normalized[format_id] = {"status": status, "metrics_by_benchmark": row.get("metrics_by_benchmark", {})}
        else:
            raise ValueError(f"{format_id}: unsupported external status {status}")
    return normalized


def pending_results(selected: list[str], probe: dict[str, Any]) -> dict[str, Any]:
    """Create explicit pending/skipped rows when metrics are not supplied."""
    rows: dict[str, Any] = {}
    for format_id in checker.FORMATS:
        if format_id not in selected:
            rows[format_id] = {"status": "PENDING_GPU_RUN", "pending_reason": "not selected in this invocation"}
        elif probe["format_probes"][format_id]["available"]:
            rows[format_id] = {
                "status": "PENDING_GPU_RUN",
                "pending_reason": "kernel probe passed; GPU benchmark metrics not supplied",
            }
        else:
            rows[format_id] = {"status": "SKIPPED_INFRA", "skip_reason": probe["format_probes"][format_id]["reason"]}
    return rows


def headline(rows: dict[str, Any]) -> dict[str, Any]:
    """Extract compact reporting rows for completed formats."""
    return {
        key: value.get("metrics_by_benchmark", {})
        for key, value in rows.items()
        if value.get("status") == "COMPLETED"
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"om_stage1_e4_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--formats", default=",".join(checker.FORMATS))
    parser.add_argument("--aime-file", type=Path, default=DEFAULT_AIME_FILE)
    parser.add_argument("--aime-count", type=int, default=12)
    parser.add_argument("--math-dataset", default=MATH_DATASET)
    parser.add_argument("--math-split", default="test")
    parser.add_argument("--math-count", type=int, default=30)
    parser.add_argument("--math-file", type=Path)
    parser.add_argument("--skip-math-download", action="store_true")
    parser.add_argument("--metrics-json", type=Path)
    parser.add_argument("--cap-hours", type=float, default=15.0)
    parser.add_argument("--skip-model-resolve", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    selected = [item.strip() for item in args.formats.split(",") if item.strip()]
    if any(item not in checker.FORMATS for item in selected):
        raise SystemExit(f"--formats must be drawn from {checker.FORMATS}")
    prompts = prompt_manifest(args)
    probe = kernel_probe(selected)
    model_snapshot = (
        {"schema_version": f"{SCHEMA_VERSION}_model_snapshot", "status": "SKIPPED_DRY_RUN"}
        if args.skip_model_resolve
        else shared.resolve_model_snapshot(MODEL_ID, schema_version=SCHEMA_VERSION)
    )
    if args.dry_run:
        print(json.dumps({"selected_formats": selected, "model_snapshot": model_snapshot, "prompt_manifest": prompts, "kernel_probe": probe}, indent=2, sort_keys=True))
        return 0

    run_dir = args.results_dir / args.run_id
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    shared.write_json(run_dir / "environment.json", shared.build_environment(schema_version=SCHEMA_VERSION))
    shared.write_json(run_dir / "command_metadata.json", {"schema_version": f"{SCHEMA_VERSION}_command", "argv": sys.argv if argv is None else argv, "cap_hours": args.cap_hours, "model_id": MODEL_ID, "selected_formats": selected})
    shared.write_json(run_dir / "prompt_manifest.json", prompts)
    shared.write_json(run_dir / "kernel_probe.json", probe)
    rows = normalize_metrics(args.metrics_json, selected, probe) if args.metrics_json else pending_results(selected, probe)
    shared.write_json(run_dir / "e4_format_axis.json", {"schema_version": f"{SCHEMA_VERSION}_metrics", "created_at_utc": shared.utc_now(), "model_id": MODEL_ID, "benchmarks": list(checker.BENCHMARKS), "format_results": rows, "headline": headline(rows)})
    shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
    result = checker.evaluate(run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"] != checker.FAIL_INFRA else 1


if __name__ == "__main__":
    raise SystemExit(main())
