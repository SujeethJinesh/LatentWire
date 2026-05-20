#!/usr/bin/env python3
"""Run the Phase 9 ParoQuant Granite-Small baseline."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.outlier_migrate.phase4 import run_om_phase4_intervention as phase4_runner
from experimental.outlier_migrate.phase9 import check_om_paroquant_baseline as checker
from experimental.outlier_migrate.phase9 import run_om_phase9_m11_ema_drift as m11_runner
from experimental.outlier_migrate.phase9 import run_om_phase9_m2_position_conditional as m2_runner
from experimental.shared import run_phase0_branch as shared


DEFAULT_RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase9/results"
DEFAULT_PROMPT_FILE = checker.DEFAULT_PROMPT_FILE
DEFAULT_MODEL_ID = checker.MODEL_ID
SCHEMA_VERSION = checker.SCHEMA_VERSION


def parse_prompt_file(prompt_file: Path) -> tuple[list[dict[str, Any]], list[str]]:
    reasons: list[str] = []
    prompts: list[dict[str, Any]] = []
    if not prompt_file.is_file():
        return [], [f"prompt file missing: {prompt_file}"]
    for row_index, line in enumerate(prompt_file.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        item = json.loads(line)
        index = int(item.get("index", row_index))
        if index >= checker.TRACE_COUNT:
            continue
        prompts.append(
            {
                "index": index,
                "prompt_id": str(item.get("prompt_id", item.get("id", row_index))),
                "prompt": str(item.get("prompt") or item.get("problem") or item.get("question")),
                "answer": item.get("answer"),
                "source_dataset": item.get("source_dataset"),
                "source_file": item.get("source_file"),
                "source_commit": item.get("source_commit"),
            }
        )
        if item.get("source_dataset") != checker.EXPECTED_PROMPT_SOURCE_DATASET:
            reasons.append(f"prompt {index}: source_dataset mismatch")
        if item.get("source_commit") != checker.EXPECTED_PROMPT_SOURCE_COMMIT:
            reasons.append(f"prompt {index}: source_commit mismatch")
        if item.get("source_file") != checker.expected_source_file(index):
            reasons.append(f"prompt {index}: source_file mismatch")
        if item.get("prompt_id") != checker.expected_prompt_id(index):
            reasons.append(f"prompt {index}: prompt_id mismatch")
    if [int(row["index"]) for row in prompts] != list(range(checker.TRACE_COUNT)):
        reasons.append(f"prompt indices are not exactly 0-{checker.TRACE_COUNT - 1}")
    if len(prompts) != checker.TRACE_COUNT:
        reasons.append(f"prompt count is not {checker.TRACE_COUNT}")
    return prompts, reasons


def build_prompt_manifest(prompt_file: Path) -> tuple[dict[str, Any], list[str]]:
    prompts, reasons = parse_prompt_file(prompt_file)
    return (
        {
            "schema_version": f"{SCHEMA_VERSION}_prompt_manifest",
            "created_at_utc": shared.utc_now(),
            "source": "AIME-2025",
            "selection": "deterministic_indices_0_11_vacation_revision",
            "source_dataset": checker.EXPECTED_PROMPT_SOURCE_DATASET,
            "source_dataset_commit": checker.EXPECTED_PROMPT_SOURCE_COMMIT,
            "source_file_order": ["aime2025-I.jsonl", "aime2025-II.jsonl"],
            "prompt_file": str(prompt_file),
            "prompt_file_sha256": shared.file_sha256(prompt_file) if prompt_file.is_file() else None,
            "prompt_count": len(prompts),
            "prompt_sha256": checker.prompt_payload_sha256(prompts) if prompts else None,
            "prompt_sha256_semantics": "sha256 of concatenated prompt text in deterministic index order",
            "prompts": prompts,
        },
        reasons,
    )


def release_model_memory(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def write_failure_packet(run_dir: Path, run_events_path: Path, exc: BaseException | str) -> None:
    payload: dict[str, Any] = {
        "schema_version": f"{SCHEMA_VERSION}_infra_error",
        "created_at_utc": shared.utc_now(),
        "decision": checker.FAIL_INFRA,
        "reason": str(exc),
    }
    if isinstance(exc, BaseException):
        payload["exception_type"] = type(exc).__name__
        payload["traceback"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    try:
        run_events_path.open("a", encoding="utf-8").write(
            json.dumps({"created_at_utc": shared.utc_now(), "event": "run_failed", "reason": str(exc)}, sort_keys=True) + "\n"
        )
        shared.write_json(run_dir / "infra_error.json", payload)
        shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
        checker.evaluate(run_dir)
    except Exception:
        pass


def top_channels(values: list[float], count: int) -> list[int]:
    return sorted(range(len(values)), key=lambda channel: (-float(values[channel]), channel))[:count]


def activation_means_by_layer_position(rows: list[dict[str, Any]]) -> tuple[dict[int, dict[int, list[float]]], dict[int, str]]:
    grouped: dict[int, dict[int, list[list[float]]]] = {}
    layer_names: dict[int, str] = {}
    for row in rows:
        layer_index = int(row["layer_index"])
        position = int(row["decode_position"])
        layer_names[layer_index] = str(row["layer_name"])
        grouped.setdefault(layer_index, {}).setdefault(position, []).append([float(value) for value in row["channel_magnitudes"]])
    means_by_layer: dict[int, dict[int, list[float]]] = {}
    for layer_index, by_position in grouped.items():
        means_by_layer[layer_index] = {}
        for position, vectors in by_position.items():
            channel_count = len(vectors[0])
            means_by_layer[layer_index][position] = [
                float(mean(vector[channel] for vector in vectors)) for channel in range(channel_count)
            ]
    return means_by_layer, layer_names


def build_static_top1_sets(rows: list[dict[str, Any]]) -> dict[str, Any]:
    means_by_layer, layer_names = activation_means_by_layer_position(rows)
    layers: dict[str, Any] = {}
    for layer_index in sorted(means_by_layer):
        if 100 not in means_by_layer[layer_index]:
            raise RuntimeError(f"layer {layer_index} missing decode position 100 activation rows")
        values = means_by_layer[layer_index][100]
        count = max(1, math.ceil(len(values) * 0.01))
        channels = sorted(top_channels(values, count))
        layers[str(layer_index)] = {
            "layer_name": layer_names[layer_index],
            "channel_count": len(values),
            "protected_count": len(channels),
            "protected_channels": channels,
            "source": "position_100_top1",
        }
    return {
        "schema_version": f"{SCHEMA_VERSION}_protected_sets",
        "created_at_utc": shared.utc_now(),
        "selection_basis": "mean absolute layer output activation at decode position 100",
        "regimes": {"static_1pct": {"layers": layers}},
    }


def quantize_affine_groupwise_4bit(work: Any, group_size: int) -> Any:
    import torch

    original_shape = work.shape
    x = work.float().reshape(-1, group_size)
    min_val = x.amin(dim=1, keepdim=True)
    max_val = x.amax(dim=1, keepdim=True)
    scale = (max_val - min_val).clamp_min(1e-5) / 15.0
    zero_point = torch.round(-min_val / scale).clamp(0, 15)
    q = torch.round(x / scale + zero_point).clamp(0, 15)
    dequant = (q - zero_point) * scale
    return dequant.reshape(original_shape)


def quantize_symmetric_int4_work(work: Any) -> Any:
    scale = work.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7.0
    return torch_round(work.float() / scale).clamp(-7, 7).mul(scale)


def torch_round(x: Any) -> Any:
    import torch

    return torch.round(x)


def build_pair_indices(
    *,
    in_features: int,
    group_size: int,
    num_rotations: int,
    device: Any,
    dtype: Any,
) -> list[tuple[Any, Any, Any, Any]]:
    import torch

    pairs: list[tuple[Any, Any, Any, Any]] = []
    for rotation_index in range(num_rotations):
        left: list[int] = []
        right: list[int] = []
        angle_values: list[float] = []
        for start in range(0, in_features, group_size):
            order = list(range(start, start + group_size))
            shift = rotation_index % group_size
            order = order[shift:] + order[:shift]
            for offset in range(group_size // 2):
                left.append(order[offset])
                right.append(order[group_size - 1 - offset])
                angle_values.append(math.pi / 4.0 if rotation_index % 2 == 0 else -math.pi / 4.0)
        i = torch.tensor(left, device=device, dtype=torch.long)
        j = torch.tensor(right, device=device, dtype=torch.long)
        angles = torch.tensor(angle_values, device=device, dtype=dtype)
        pairs.append((i, j, torch.cos(angles), torch.sin(angles)))
    return pairs


def apply_rotations(work: Any, rotations: list[tuple[Any, Any, Any, Any]], *, inverse: bool = False) -> None:
    sequence = reversed(rotations) if inverse else rotations
    for i, j, c, s in sequence:
        if inverse:
            s = -s
        left = work.index_select(1, i).clone()
        right = work.index_select(1, j).clone()
        work.index_copy_(1, i, left * c - right * s)
        work.index_copy_(1, j, left * s + right * c)


def quantize_weight_paroquant_inplace(
    weight: Any,
    *,
    group_size: int,
    num_rotations: int,
    row_chunk: int,
    scale_clip: tuple[float, float],
) -> dict[str, Any]:
    import torch

    if weight.ndim not in {2, 3}:
        return {"reason": "weight tensor is not 2D or expert-bank 3D", "shape": list(weight.shape)}
    original_dtype = weight.dtype
    try:
        flat = weight.view(-1, weight.shape[-1])
    except RuntimeError:
        return {"reason": "weight tensor is not contiguous in last dimension", "shape": list(weight.shape)}
    in_features = int(flat.shape[-1])
    if in_features < group_size or in_features % group_size != 0:
        phase4_runner.quantize_weight_symmetric_int4_inplace(weight)
        return {
            "shape": list(weight.shape),
            "mode": "fallback_symmetric_int4_last_dim_not_group_aligned",
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
    rotations = build_pair_indices(
        in_features=in_features,
        group_size=group_size,
        num_rotations=num_rotations,
        device=flat.device,
        dtype=torch.float32,
    )

    for start in range(0, flat.shape[0], row_chunk):
        block = flat[start : start + row_chunk]
        work = block.float()
        work.mul_(scales.view(1, -1))
        apply_rotations(work, rotations, inverse=False)
        work = quantize_affine_groupwise_4bit(work, group_size)
        apply_rotations(work, rotations, inverse=True)
        work.div_(scales.view(1, -1))
        block.copy_(work.to(dtype=original_dtype))

    return {
        "shape": list(weight.shape),
        "mode": "scaled_pairwise_rotation_groupwise_affine_int4_folded_back",
        "in_features": in_features,
        "group_size": group_size,
        "num_rotations": num_rotations,
        "row_count": int(flat.shape[0]),
        "scale_min": float(scales.min().item()),
        "scale_median": float(torch.median(scales).item()),
        "scale_max": float(scales.max().item()),
        "scale_clip": list(scale_clip),
        "pairing_rule": "deterministic independent high-low within each group, alternating +/- pi/4 across rotations",
    }


def apply_paroquant_quantization(
    model: Any,
    *,
    group_size: int,
    num_rotations: int,
    row_chunk: int,
    scale_clip: tuple[float, float],
) -> dict[str, Any]:
    import torch

    quantized: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    with torch.no_grad():
        for name, module in model.named_modules():
            weight = getattr(module, "weight", None)
            if not torch.is_tensor(weight):
                continue
            if phase4_runner.is_tied_lm_head(model, name, module):
                excluded.append({"name": name, "reason": "tied input/output embedding excluded", "shape": list(weight.shape)})
                continue
            item = quantize_weight_paroquant_inplace(
                weight,
                group_size=group_size,
                num_rotations=num_rotations,
                row_chunk=row_chunk,
                scale_clip=scale_clip,
            )
            item["name"] = name
            if "reason" in item:
                excluded.append(item)
            else:
                quantized.append(item)
    return {
        "regime": "paroquant_w4a16",
        "quantized_tensor_count": len(quantized),
        "quantized_tensors": quantized,
        "excluded_tensors": excluded,
    }


def write_score_cache(run_dir: Path, regime: str, scores: dict[int, dict[str, float]]) -> None:
    shared.write_json(
        run_dir / "score_cache" / f"{regime}.json",
        {
            "schema_version": f"{SCHEMA_VERSION}_score_cache",
            "created_at_utc": shared.utc_now(),
            "regime": regime,
            "scores": {str(index): row for index, row in sorted(scores.items())},
        },
    )


def score_paroquant_regime(
    *,
    model_provenance: dict[str, Any],
    prompts: list[dict[str, Any]],
    target_tokens: dict[int, list[int]],
    batch_size: int,
    dtype_name: str,
    device_name: str,
    run_events_path: Path,
    group_size: int,
    num_rotations: int,
    row_chunk: int,
    scale_clip: tuple[float, float],
) -> tuple[dict[int, dict[str, float]], dict[str, Any]]:
    model, tokenizer, device = shared.load_model_and_tokenizer(model_provenance, dtype_name=dtype_name, device_name=device_name)
    excluded = apply_paroquant_quantization(
        model,
        group_size=group_size,
        num_rotations=num_rotations,
        row_chunk=row_chunk,
        scale_clip=scale_clip,
    )
    scores = phase4_runner.score_targets(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompts=prompts,
        target_tokens=target_tokens,
        max_new_tokens=checker.SCORING_POSITION,
        batch_size=batch_size,
        use_float16_autocast=True,
        run_events_path=run_events_path,
        regime_name="paroquant_w4a16",
    )
    del model, tokenizer, device
    release_model_memory()
    return scores, excluded


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "median_recovery": float(median(values)) if values else None,
        "mean_recovery": float(mean(values)) if values else None,
        "bootstrap_ci95": checker.bootstrap_median(values),
        "included_trace_count": len(values),
    }


def build_metrics(
    *,
    run_dir: Path,
    prompt_manifest: dict[str, Any],
    model_provenance: dict[str, Any],
    per_trace_rows: list[dict[str, Any]],
    m11b_reference_run_dir: Path | None,
    implementation_mode: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    included = [row for row in per_trace_rows if not bool(row["no_recoverable_static_gap"])]
    values = [float(row["recoveries"]["paroquant_w4a16"]) for row in included]
    summary = summarize(values)
    no_gap_count = len(per_trace_rows) - len(included)
    summary.update(
        {
            "total_trace_count": len(per_trace_rows),
            "no_recoverable_static_gap_count": no_gap_count,
            "no_recoverable_static_gap_fraction": no_gap_count / len(per_trace_rows) if per_trace_rows else 0.0,
            "per_trace_recovery_included": values,
        }
    )
    m11b_top5_median = None
    if m11b_reference_run_dir and (m11b_reference_run_dir / "metrics.json").is_file():
        try:
            m11b_metrics = json.loads((m11b_reference_run_dir / "metrics.json").read_text(encoding="utf-8"))
            m11b_top5_median = m11b_metrics["results_by_regime"]["m11b_top5"]["median_recovery"]
        except Exception:
            m11b_top5_median = None
    median_recovery = summary.get("median_recovery")
    metrics = {
        "schema_version": f"{SCHEMA_VERSION}_metrics",
        "created_at_utc": shared.utc_now(),
        "preregistration": str(checker.PREREG_PATH.relative_to(ROOT)),
        "preregistration_sha256": shared.file_sha256(checker.PREREG_PATH),
        "model_id": model_provenance.get("model_id"),
        "model_snapshot_commit": model_provenance.get("hf_snapshot_commit"),
        "prompt_sha256": prompt_manifest["prompt_sha256"],
        "trace_count": checker.TRACE_COUNT,
        "effective_trace_count": len(per_trace_rows),
        "included_trace_count": len(included),
        "no_recoverable_static_gap_count": no_gap_count,
        "no_recoverable_static_gap_fraction": no_gap_count / len(per_trace_rows) if per_trace_rows else 0.0,
        "scoring_position": checker.SCORING_POSITION,
        "scoring_window_tokens": checker.SCORING_WINDOW_TOKENS,
        "implementation_mode": implementation_mode,
        "metric_name": "positive-static-1pct-gap per-trace recovery",
        "metric_formula": "1 - (perplexity_ParoQuant - perplexity_BF16) / (perplexity_static_1pct - perplexity_BF16)",
        "results_by_regime": {"paroquant_w4a16": summary},
        "interpretation_bands": checker.INTERPRETATION_BANDS,
        "interpretation_band": checker.interpretation_band(median_recovery),
        "m11b_top5_reference_median_recovery": m11b_top5_median,
        "paroquant_minus_m11b_top5_median_recovery": (
            None if median_recovery is None or m11b_top5_median is None else float(median_recovery) - float(m11b_top5_median)
        ),
        "artifacts": {"run_dir": str(run_dir)},
    }
    bootstrap = {
        "schema_version": f"{SCHEMA_VERSION}_bootstrap_ci",
        "metric_name": metrics["metric_name"],
        "bootstrap_samples": checker.BOOTSTRAP_SAMPLES,
        "bootstrap_seed": checker.BOOTSTRAP_SEED,
        "results_by_regime": metrics["results_by_regime"],
    }
    controls = {
        "schema_version": f"{SCHEMA_VERSION}_control_metrics",
        "created_at_utc": shared.utc_now(),
        "controls": {
            "static_1pct": {"median_recovery": 0.0},
            "random_channel_matched": {
                "status": "not_applicable_no_channel_selection",
                "reason": "ParoQuant is a transform-based W4A16 baseline, not a channel-selection protection method.",
            },
            "m11b_top5_reference_median_recovery": m11b_top5_median,
        },
    }
    return metrics, bootstrap, controls


def main(argv: list[str] | None = None) -> int:
    shared.SCHEMA_VERSION = SCHEMA_VERSION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"om_paroquant_granite_small_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=checker.BOOTSTRAP_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--num-rotations", type=int, default=8)
    parser.add_argument("--row-chunk", type=int, default=256)
    parser.add_argument("--scale-clip-min", type=float, default=0.25)
    parser.add_argument("--scale-clip-max", type=float, default=4.0)
    parser.add_argument("--reuse-trace-run-dir", type=Path)
    parser.add_argument("--reuse-score-run-dir", type=Path)
    parser.add_argument("--reuse-protected-run-dir", type=Path)
    parser.add_argument("--m11b-reference-run-dir", type=Path)
    args = parser.parse_args(argv)

    if args.model_id != checker.MODEL_ID:
        raise SystemExit(f"ParoQuant Granite baseline requires {checker.MODEL_ID}")
    if args.prompt_file.resolve() != DEFAULT_PROMPT_FILE.resolve():
        raise SystemExit(f"ParoQuant baseline requires canonical prompt file {DEFAULT_PROMPT_FILE}")
    if shared.file_sha256(args.prompt_file) != checker.EXPECTED_PROMPT_FILE_SHA256:
        raise SystemExit("canonical AIME-2025 indices 0-23 prompt file hash drifted")
    if args.seed != checker.BOOTSTRAP_SEED:
        raise SystemExit("ParoQuant baseline preregisters seed 20260601")

    run_dir = args.results_dir / args.run_id
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")
    (run_dir / "logs").mkdir(parents=True)
    stdout_log = (run_dir / "logs/stdout.log").open("w", encoding="utf-8", buffering=1)
    stderr_log = (run_dir / "logs/stderr.log").open("w", encoding="utf-8", buffering=1)
    sys.stdout = shared.Tee(sys.__stdout__, stdout_log)
    sys.stderr = shared.Tee(sys.__stderr__, stderr_log)
    run_events_path = run_dir / "run_events.jsonl"
    previous_excepthook = sys.excepthook

    def paroquant_excepthook(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        write_failure_packet(run_dir, run_events_path, exc)
        previous_excepthook(exc_type, exc, tb)

    sys.excepthook = paroquant_excepthook
    run_events_path.write_text(
        json.dumps({"created_at_utc": shared.utc_now(), "event": "run_started"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    random.seed(args.seed)

    try:
        prompt_manifest, prompt_reasons = build_prompt_manifest(args.prompt_file)
        environment = shared.build_environment(schema_version=SCHEMA_VERSION)
        model_provenance = m2_runner.resolve_model_snapshot_light(args.model_id)
        model_provenance["schema_version"] = f"{SCHEMA_VERSION}_model_provenance"
        implementation_mode = "algorithmic_reproduction_not_full_upstream"
        scale_clip = (float(args.scale_clip_min), float(args.scale_clip_max))
        upstream_repo = ROOT / ".debug/paroquant"
        upstream_commit = None
        if (upstream_repo / ".git").is_dir():
            try:
                import subprocess

                upstream_commit = subprocess.check_output(["git", "-C", str(upstream_repo), "rev-parse", "HEAD"], text=True).strip()
            except Exception:
                upstream_commit = None

        shared.write_json(run_dir / "prompt_manifest.json", prompt_manifest)
        shared.write_json(run_dir / "environment.json", environment)
        phase4_runner.write_environment_text(run_dir / "environment.txt", environment)
        shared.write_json(run_dir / "model_provenance.json", model_provenance)
        shared.write_json(
            run_dir / "command_metadata.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_command",
                "created_at_utc": shared.utc_now(),
                "argv": sys.argv if argv is None else ["run_om_paroquant_baseline.py", *argv],
                "cwd": str(Path.cwd()),
                "branch": "outlier_migrate_phase9_paroquant_baseline",
                "run_dir": str(run_dir),
                "batch_size": args.batch_size,
                "reuse_trace_run_dir": str(args.reuse_trace_run_dir) if args.reuse_trace_run_dir else None,
                "reuse_score_run_dir": str(args.reuse_score_run_dir) if args.reuse_score_run_dir else None,
                "reuse_protected_run_dir": str(args.reuse_protected_run_dir) if args.reuse_protected_run_dir else None,
            },
        )
        shared.write_json(
            run_dir / "random_seed.json",
            {"schema_version": f"{SCHEMA_VERSION}_random_seed", "seed": args.seed, "determinism": {"do_sample": False, "num_beams": 1}},
        )
        shared.write_json(
            run_dir / "decoding_config.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_decoding_config",
                "max_new_tokens_for_scoring": checker.SCORING_POSITION,
                "do_sample": False,
                "num_beams": 1,
                "scoring_position": checker.SCORING_POSITION,
                "scoring_window_tokens": checker.SCORING_WINDOW_TOKENS,
            },
        )
        shared.write_json(
            run_dir / "quantization_config.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_quantization_config",
                "weight_bits": 4,
                "scheme": "ParoQuant-style scaled pairwise rotation plus groupwise affine INT4 folded back to dequantized weights",
                "activation_dtype": "float16",
                "group_size": args.group_size,
                "num_rotations": args.num_rotations,
                "scale_clip": list(scale_clip),
                "implementation_mode": implementation_mode,
            },
        )
        shared.write_json(
            run_dir / "paroquant_config.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_paroquant_config",
                "created_at_utc": shared.utc_now(),
                "implementation_mode": implementation_mode,
                "upstream_reference_repo": "https://github.com/z-lab/paroquant",
                "upstream_reference_commit": upstream_commit,
                "local_reference_clone": str(upstream_repo) if upstream_repo.exists() else None,
                "arxiv": "2511.10645",
                "openreview_pdf": "https://openreview.net/pdf?id=1USeVjsKau",
                "group_size": args.group_size,
                "num_rotations": args.num_rotations,
                "pairing_rule": "deterministic independent high-low within each group, alternating +/- pi/4 across rotations",
                "channel_scale_rule": "median column RMS divided by column RMS, clipped",
                "scale_clip": list(scale_clip),
                "runtime_kernel": "not_used",
            },
        )
        shared.write_json(
            run_dir / "paroquant_limitations.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_paroquant_limitations",
                "created_at_utc": shared.utc_now(),
                "omitted_components": [
                    "fused CUDA transform kernel",
                    "upstream layer-wise learned angle optimization",
                    "QAT-like second-stage quantizer and weight fine-tuning",
                    "throughput measurement",
                ],
                "paper_reporting_requirement": "Report as algorithmic reproduction, not full upstream ParoQuant reproduction.",
            },
        )
        if prompt_reasons:
            shared.write_json(run_dir / "infra_error.json", {"reasons": prompt_reasons})
            shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
            checker.evaluate(run_dir)
            return 1
        if not model_provenance.get("snapshot_path") or model_provenance.get("hf_snapshot_commit") != checker.MODEL_SNAPSHOT:
            shared.write_json(run_dir / "infra_error.json", {"reasons": ["model snapshot missing or mismatch"]})
            shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
            checker.evaluate(run_dir)
            return 1

        prompts = prompt_manifest["prompts"]
        prompt_indices = {int(row["index"]) for row in prompts}
        trace_path = run_dir / "bf16_traces.jsonl.gz"
        activation_path = run_dir / "activation_magnitudes.jsonl.gz"
        source_artifacts: dict[str, Any] = {
            "schema_version": f"{SCHEMA_VERSION}_source_artifacts",
            "created_at_utc": shared.utc_now(),
            "artifacts": [],
        }

        protected_sets: dict[str, Any] | None = None
        if args.reuse_protected_run_dir:
            source = args.reuse_protected_run_dir.resolve() / "protected_sets.json"
            source_payload = json.loads(source.read_text(encoding="utf-8"))
            protected_sets = {
                "schema_version": f"{SCHEMA_VERSION}_protected_sets",
                "created_at_utc": shared.utc_now(),
                "source_artifact": str(source),
                "regimes": {"static_1pct": source_payload["regimes"]["static_1pct"]},
            }
            source_artifacts["artifacts"].append({"name": "protected_sets", "path": str(source), "sha256": shared.file_sha256(source)})

        if args.reuse_trace_run_dir:
            reuse_trace_dir = args.reuse_trace_run_dir.resolve()
            m2_runner.copy_filtered_jsonl_gz(reuse_trace_dir / "bf16_traces.jsonl.gz", trace_path, prompt_indices=prompt_indices)
            trace_manifest = json.loads((reuse_trace_dir / "bf16_trace_manifest.json").read_text(encoding="utf-8"))
            trace_manifest["created_at_utc"] = shared.utc_now()
            trace_manifest["source_run_dir"] = str(reuse_trace_dir)
            shared.write_json(run_dir / "bf16_trace_manifest.json", trace_manifest)
            source_artifacts["artifacts"].append(
                {"name": "bf16_traces", "path": str(reuse_trace_dir / "bf16_traces.jsonl.gz"), "sha256": shared.file_sha256(reuse_trace_dir / "bf16_traces.jsonl.gz")}
            )
        else:
            model, tokenizer, device = shared.load_model_and_tokenizer(model_provenance, dtype_name=args.dtype, device_name=args.device)
            trace_manifest = phase4_runner.generate_bf16_traces(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompts=prompts,
                max_new_tokens=checker.SCORING_POSITION,
                batch_size=args.batch_size,
                output_path=trace_path,
                run_events_path=run_events_path,
            )
            shared.write_json(run_dir / "bf16_trace_manifest.json", trace_manifest)
            del model, tokenizer, device
            release_model_memory()
        target_tokens = phase4_runner.load_trace_tokens(trace_path)

        if protected_sets is None:
            model, tokenizer, device = shared.load_model_and_tokenizer(model_provenance, dtype_name=args.dtype, device_name=args.device)
            activation_manifest = shared.capture_activation_magnitudes(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompts=prompts,
                positions=(100,),
                max_new_tokens=100,
                batch_size=args.batch_size,
                output_path=activation_path,
                run_events_path=run_events_path,
            )
            shared.write_json(run_dir / "activation_magnitude_manifest.json", activation_manifest)
            protected_sets = build_static_top1_sets(list(shared.iter_activation_rows(activation_path)))
            del model, tokenizer, device
            release_model_memory()
        shared.write_json(run_dir / "protected_sets.json", protected_sets)

        all_scores: dict[str, dict[int, dict[str, float]]] = {}
        excluded_by_regime: dict[str, Any] = {}
        score_reuse_dir = args.reuse_score_run_dir.resolve() if args.reuse_score_run_dir else None
        for regime in ["bf16", "static_1pct"]:
            cached = m11_runner.read_score_cache_any(score_reuse_dir, regime, expected_prompt_indices=prompt_indices)
            if cached is not None:
                all_scores[regime] = cached
                write_score_cache(run_dir, regime, all_scores[regime])
                source_artifacts["artifacts"].append(
                    {"name": f"score_cache_{regime}", "path": str(score_reuse_dir / "score_cache" / f"{regime}.json"), "sha256": shared.file_sha256(score_reuse_dir / "score_cache" / f"{regime}.json")}
                )
                continue
            model, tokenizer, device = shared.load_model_and_tokenizer(model_provenance, dtype_name=args.dtype, device_name=args.device)
            if regime == "static_1pct":
                excluded_by_regime[regime] = phase4_runner.apply_quantization(model, protected_sets, regime)
            all_scores[regime] = phase4_runner.score_targets(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompts=prompts,
                target_tokens=target_tokens,
                max_new_tokens=checker.SCORING_POSITION,
                batch_size=args.batch_size,
                use_float16_autocast=regime != "bf16",
                run_events_path=run_events_path,
                regime_name=regime,
            )
            write_score_cache(run_dir, regime, all_scores[regime])
            del model, tokenizer, device
            release_model_memory()

        all_scores["paroquant_w4a16"], excluded_by_regime["paroquant_w4a16"] = score_paroquant_regime(
            model_provenance=model_provenance,
            prompts=prompts,
            target_tokens=target_tokens,
            batch_size=args.batch_size,
            dtype_name=args.dtype,
            device_name=args.device,
            run_events_path=run_events_path,
            group_size=args.group_size,
            num_rotations=args.num_rotations,
            row_chunk=args.row_chunk,
            scale_clip=scale_clip,
        )
        write_score_cache(run_dir, "paroquant_w4a16", all_scores["paroquant_w4a16"])

        shared.write_json(
            run_dir / "excluded_tensors.json",
            {"schema_version": f"{SCHEMA_VERSION}_excluded_tensors", "created_at_utc": shared.utc_now(), "by_regime": excluded_by_regime},
        )
        shared.write_json(run_dir / "source_artifacts.json", source_artifacts)

        per_trace_rows: list[dict[str, Any]] = []
        for prompt in prompts:
            index = int(prompt["index"])
            perplexities = {regime: float(all_scores[regime][index]["perplexity"]) for regime in checker.REGIMES}
            mean_nll = {regime: float(all_scores[regime][index]["mean_nll"]) for regime in checker.REGIMES}
            static_gap = perplexities["static_1pct"] - perplexities["bf16"]
            no_gap = static_gap <= 0.0
            recoveries = {
                "paroquant_w4a16": None
                if no_gap
                else 1.0 - (perplexities["paroquant_w4a16"] - perplexities["bf16"]) / static_gap
            }
            per_trace_rows.append(
                {
                    "prompt_index": index,
                    "prompt_id": prompt["prompt_id"],
                    "perplexities": perplexities,
                    "mean_nll": mean_nll,
                    "static_gap": float(static_gap),
                    "no_recoverable_static_gap": bool(no_gap),
                    "recoveries": recoveries,
                    "scored_tokens": int(all_scores["bf16"][index]["scored_tokens"]),
                    "score_start": int(all_scores["bf16"][index]["score_start"]),
                    "score_end": int(all_scores["bf16"][index]["score_end"]),
                }
            )
        shared.write_json(run_dir / "per_trace_metrics.json", {"schema_version": f"{SCHEMA_VERSION}_per_trace_metrics", "created_at_utc": shared.utc_now(), "traces": per_trace_rows})
        metrics, bootstrap, controls = build_metrics(
            run_dir=run_dir,
            prompt_manifest=prompt_manifest,
            model_provenance=model_provenance,
            per_trace_rows=per_trace_rows,
            m11b_reference_run_dir=args.m11b_reference_run_dir.resolve() if args.m11b_reference_run_dir else None,
            implementation_mode=implementation_mode,
        )
        shared.write_json(run_dir / "metrics.json", metrics)
        shared.write_json(run_dir / "bootstrap_ci.json", bootstrap)
        shared.write_json(run_dir / "control_metrics.json", controls)
        run_events_path.open("a", encoding="utf-8").write(json.dumps({"created_at_utc": shared.utc_now(), "event": "run_completed"}, sort_keys=True) + "\n")
        print(json.dumps({"run_dir": str(run_dir), "results_by_regime": metrics["results_by_regime"]}, indent=2, sort_keys=True))
        sys.stdout.flush()
        sys.stderr.flush()
        shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
        result = checker.evaluate(run_dir)
        print(json.dumps({"checker_decision": result["decision"], "artifact_complete": result.get("artifact_complete", False)}, indent=2, sort_keys=True))
        shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
        checker.evaluate(run_dir)
        sys.excepthook = previous_excepthook
        return 0
    except BaseException as exc:
        write_failure_packet(run_dir, run_events_path, exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
