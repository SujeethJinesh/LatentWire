#!/usr/bin/env python3
"""Run the preregistered Cross-Layer Quantization Error Compounding validation."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.cross_layer_error import check_cle_bound as checker
from experimental.shared import run_phase0_branch as shared
from experimental.shared.fp4_simulator import simulate_mxfp4_e2m1


RESULTS_DIR = ROOT / "experimental/cross_layer_error/results"
SCHEMA_VERSION = checker.SCHEMA_VERSION
MODEL_ID = checker.MODEL_ID
PROMPT_FILE = checker.DEFAULT_PROMPT_FILE
DEPTHS = checker.DEPTHS
DECODE_POSITION = checker.DECODE_POSITION
SEED = 20260508
BLOCK_SIZE = 16
OUTLIER_FRACTION = 0.01
QUANTIZATION_FORMAT = "nvfp4_e2m1_weight_sim"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_event(path: Path, event: str, **payload: Any) -> None:
    row = {"created_at_utc": utc_now(), "event": event, **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_command_metadata(args: argparse.Namespace, run_dir: Path, argv: list[str] | None) -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_VERSION}_command",
        "created_at_utc": utc_now(),
        "argv": sys.argv if argv is None else ["run_cle_validation.py", *argv],
        "cwd": str(Path.cwd()),
        "branch": "cross_layer_error",
        "run_dir": str(run_dir),
        "model_id": MODEL_ID,
        "depths": list(DEPTHS),
        "decode_position": DECODE_POSITION,
        "prompt_file": str(args.prompt_file),
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "device": args.device,
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "manual_cache_decode": True,
            "eos_policy": "continue decoding until decode_position so every prompt has logits at position 1000",
        },
        "measurement_semantics": (
            "A BF16 greedy trace token prefix is generated once and written to trace_tokens.jsonl. "
            "BF16 and each FP4-depth variant are then teacher-forced over that exact prefix; "
            "drift compares logits after consuming generated token 1000."
        ),
    }


def fail_infra(run_dir: Path, run_events_path: Path, reasons: list[str]) -> int:
    append_event(run_events_path, "infra_failed", reasons=reasons)
    write_json(run_dir / "infra_error.json", {"schema_version": f"{SCHEMA_VERSION}_infra_error", "reasons": reasons})
    write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
    print(json.dumps({"run_dir": str(run_dir), "status": checker.FAIL_INFRA, "reasons": reasons}, indent=2))
    return 1


def output_embedding_scale(model: Any) -> dict[str, Any]:
    import torch

    embeddings = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    weight = getattr(embeddings, "weight", None)
    if weight is None or not torch.is_tensor(weight):
        return {
            "source": "fallback_no_output_embedding",
            "scale": 1.0,
            "vocab_size": None,
            "hidden_size": None,
        }
    values = weight.detach().to(torch.float32)
    hidden_size = int(values.shape[1]) if values.ndim == 2 else int(values.numel())
    scale = float(torch.linalg.vector_norm(values).item() / math.sqrt(max(hidden_size, 1)))
    return {
        "source": "lm_head_frobenius_norm_over_sqrt_hidden",
        "scale": scale,
        "vocab_size": int(values.shape[0]) if values.ndim == 2 else None,
        "hidden_size": hidden_size,
    }


def quantization_error_stats(model: Any, layers: list[tuple[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    max_depth = max(DEPTHS)
    if len(layers) < max_depth:
        raise RuntimeError(f"model exposes only {len(layers)} layers; CLE requires at least {max_depth}")
    layer_stats: list[dict[str, Any]] = []
    for layer_index, (layer_name, layer) in enumerate(layers[:max_depth]):
        param_count = 0
        squared_error_sum = 0.0
        outlier_squared_error_sum = 0.0
        outlier_count = 0
        scale_means: list[float] = []
        parameter_rows: list[dict[str, Any]] = []
        for param_name, param in layer.named_parameters(recurse=True):
            if not torch.is_floating_point(param.detach()):
                continue
            tensor = param.detach()
            quantized = simulate_mxfp4_e2m1(tensor, block_size=BLOCK_SIZE)
            error = quantized.dequantized.to(torch.float32) - tensor.to(torch.float32)
            abs_values = tensor.detach().abs().reshape(-1)
            count = int(abs_values.numel())
            if count == 0:
                continue
            k = max(1, math.ceil(count * OUTLIER_FRACTION))
            threshold = torch.topk(abs_values, k=k, largest=True).values[-1]
            outlier_mask = abs_values >= threshold
            flat_error = error.reshape(-1)
            param_sq = float(torch.sum(flat_error * flat_error).item())
            param_outlier_sq = float(torch.sum(flat_error[outlier_mask] * flat_error[outlier_mask]).item())
            param_outliers = int(outlier_mask.sum().item())
            squared_error_sum += param_sq
            outlier_squared_error_sum += param_outlier_sq
            outlier_count += param_outliers
            param_count += count
            if quantized.scale.numel():
                scale_means.append(float(quantized.scale.detach().to(torch.float32).mean().item()))
            parameter_rows.append(
                {
                    "parameter": param_name,
                    "shape": [int(dim) for dim in tensor.shape],
                    "numel": count,
                    "squared_error_sum": param_sq,
                    "outlier_squared_error_sum": param_outlier_sq,
                    "outlier_count": param_outliers,
                    "outlier_threshold_abs": float(threshold.item()),
                }
            )
        if param_count <= 0:
            raise RuntimeError(f"layer {layer_index} has no floating-point parameters to quantize")
        sigma_block = squared_error_sum / param_count
        sigma_outlier = outlier_squared_error_sum / max(outlier_count, 1)
        layer_error_l2 = math.sqrt(max(squared_error_sum + outlier_squared_error_sum, 0.0))
        layer_stats.append(
            {
                "layer_index": layer_index,
                "layer_name": layer_name,
                "parameter_count": param_count,
                "outlier_count": outlier_count,
                "outlier_fraction": OUTLIER_FRACTION,
                "sigma_block": sigma_block,
                "sigma_outlier": sigma_outlier,
                "squared_error_sum": squared_error_sum,
                "outlier_squared_error_sum": outlier_squared_error_sum,
                "layer_error_l2": layer_error_l2,
                "mean_block_scale": float(mean(scale_means)) if scale_means else None,
                "parameters": parameter_rows,
            }
        )
    output_scale = output_embedding_scale(model)
    return layer_stats, output_scale


def derive_bounds(layer_stats: list[dict[str, Any]], output_scale: dict[str, Any]) -> list[dict[str, Any]]:
    bounds: list[dict[str, Any]] = []
    scale = float(output_scale["scale"])
    for depth in DEPTHS:
        selected = layer_stats[:depth]
        layer_error_l2_sum = float(sum(float(row["layer_error_l2"]) for row in selected))
        layer_error_l2_quadrature = math.sqrt(
            sum(float(row["layer_error_l2"]) ** 2 for row in selected)
        )
        predicted = scale * layer_error_l2_quadrature
        bounds.append(
            {
                "depth": depth,
                "predicted_bound_l2": float(predicted),
                "quantized_layers": [str(row["layer_name"]) for row in selected],
                "sigma_block_sum": float(sum(float(row["sigma_block"]) for row in selected)),
                "sigma_outlier_sum": float(sum(float(row["sigma_outlier"]) for row in selected)),
                "layer_error_l2_sum": layer_error_l2_sum,
                "layer_error_l2_quadrature": float(layer_error_l2_quadrature),
                "output_scale": scale,
            }
        )
    return bounds


def derivation_markdown(bounds: list[dict[str, Any]], output_scale: dict[str, Any]) -> str:
    lines = [
        "# Cross-Layer Quantization Error Compounding Derivation",
        "",
        f"Locked before measurement at `{utc_now()}`.",
        "",
        "This packet uses only BF16 model weights and the fixed depth pattern to compute the bound.",
        "No BF16-vs-FP4 logit drift rows are read before this document and `predicted_bounds.json` are written.",
        "",
        "For each quantized layer `l`, each floating-point parameter tensor is block-quantized with the",
        "repo-local E2M1 block-scaled FP4 simulator (`nvfp4_e2m1_weight_sim`, block size 32). Let",
        "`e_l = Q_l(W_l) - W_l`. The recorded variance terms are:",
        "",
        "- `sigma_block_l = mean(e_l^2)` over all layer parameters.",
        "- `sigma_outlier_l = mean(e_l^2)` over the top 1% absolute-weight entries.",
        "- `eta_l = sqrt(sum(e_l^2) + sum(e_l,outlier^2))`.",
        "",
        "The output map is bounded by `C_out = ||W_lm_head||_F / sqrt(hidden_size)` when an LM head is present,",
        "falling back to `1.0` only if the output embedding cannot be resolved.",
        "",
        "For a depth pattern that quantizes the first `N` layers, the preregistered prediction used here is:",
        "",
        "`F(N, sigma_block, sigma_outlier, depth_pattern) = C_out * sqrt(sum_{l in first N layers} eta_l^2)`",
        "",
        "This is a first-principles Lipschitz-style upper-bound attempt. It is not fit to measured drift.",
        "",
        f"- Output-scale source: `{output_scale['source']}`",
        f"- Output scale: `{float(output_scale['scale']):.12g}`",
        f"- Vocab size: `{output_scale.get('vocab_size')}`",
        f"- Hidden size: `{output_scale.get('hidden_size')}`",
        "",
        "| Depth | Predicted F | sigma_block_sum | sigma_outlier_sum | layer_error_l2_quadrature |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in bounds:
        lines.append(
            f"| {row['depth']} | {float(row['predicted_bound_l2']):.12g} | "
            f"{float(row['sigma_block_sum']):.12g} | {float(row['sigma_outlier_sum']):.12g} | "
            f"{float(row['layer_error_l2_quadrature']):.12g} |"
        )
    lines.append("")
    return "\n".join(lines)


@contextmanager
def quantized_first_layers(layers: list[tuple[str, Any]], depth: int):
    import torch

    originals: list[tuple[Any, Any]] = []
    try:
        with torch.no_grad():
            for _layer_name, layer in layers[:depth]:
                for _param_name, param in layer.named_parameters(recurse=True):
                    if not torch.is_floating_point(param.detach()):
                        continue
                    originals.append((param, param.detach().cpu().clone()))
                    quantized = simulate_mxfp4_e2m1(param.data, block_size=BLOCK_SIZE)
                    param.data.copy_(quantized.dequantized.to(device=param.device, dtype=param.dtype))
        yield
    finally:
        with torch.no_grad():
            for param, original_cpu in reversed(originals):
                param.data.copy_(original_cpu.to(device=param.device, dtype=param.dtype))


def prepare_inputs(model: Any, **kwargs: Any) -> dict[str, Any]:
    if not hasattr(model, "prepare_inputs_for_generation"):
        return kwargs
    try:
        return model.prepare_inputs_for_generation(**kwargs)
    except TypeError:
        kwargs.pop("cache_position", None)
        return model.prepare_inputs_for_generation(**kwargs)


def generate_trace_tokens(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    prompts: list[dict[str, Any]],
    batch_size: int,
    run_events_path: Path,
) -> dict[int, list[int]]:
    import torch

    tokens_by_prompt: dict[int, list[int]] = {int(item["index"]): [] for item in prompts}
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            prompt_indices = [int(item["index"]) for item in batch]
            texts = [shared.make_prompt_text(str(item["prompt"])) for item in batch]
            encoded = tokenizer(texts, padding=True, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            cache_position = torch.arange(input_ids.shape[1], device=device)
            model_inputs = prepare_inputs(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                cache_position=cache_position,
                use_cache=True,
            )
            outputs = model(**model_inputs)
            past_key_values = getattr(outputs, "past_key_values", None)
            if past_key_values is None:
                raise RuntimeError("model did not expose past_key_values during CLE decode")
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            for decode_position in range(1, DECODE_POSITION + 1):
                for batch_offset, prompt_index in enumerate(prompt_indices):
                    tokens_by_prompt[prompt_index].append(int(next_token[batch_offset].item()))
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype),
                    ],
                    dim=1,
                )
                cache_position = torch.tensor([attention_mask.shape[1] - 1], device=device, dtype=torch.long)
                model_inputs = prepare_inputs(
                    model,
                    input_ids=next_token[:, None],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    use_cache=True,
                )
                outputs = model(**model_inputs)
                past_key_values = getattr(outputs, "past_key_values", None)
                if past_key_values is None:
                    raise RuntimeError(f"model dropped past_key_values at decode position {decode_position}")
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            append_event(
                run_events_path,
                "trace_tokens_batch_completed",
                prompt_indices=prompt_indices,
                decode_position=DECODE_POSITION,
            )
    return tokens_by_prompt


def write_trace_tokens(
    run_dir: Path,
    *,
    prompts: list[dict[str, Any]],
    tokens_by_prompt: dict[int, list[int]],
) -> None:
    path = run_dir / "trace_tokens.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            prompt_index = int(prompt["index"])
            token_ids = [int(token) for token in tokens_by_prompt[prompt_index]]
            payload = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
            row = {
                "schema_version": f"{SCHEMA_VERSION}_trace_tokens_row",
                "prompt_index": prompt_index,
                "prompt_id": str(prompt["prompt_id"]),
                "decode_position": DECODE_POSITION,
                "token_count": len(token_ids),
                "token_ids": token_ids,
                "token_ids_sha256": shared.bytes_sha256(payload),
                "source": "bf16_greedy_prefix",
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def capture_logits_for_trace(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    prompts: list[dict[str, Any]],
    tokens_by_prompt: dict[int, list[int]],
    batch_size: int,
    run_events_path: Path,
    variant: str,
    depth: int | None = None,
) -> dict[int, Any]:
    import torch

    logits_by_prompt: dict[int, Any] = {}
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            prompt_indices = [int(item["index"]) for item in batch]
            texts = [shared.make_prompt_text(str(item["prompt"])) for item in batch]
            encoded = tokenizer(texts, padding=True, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            cache_position = torch.arange(input_ids.shape[1], device=device)
            model_inputs = prepare_inputs(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                cache_position=cache_position,
                use_cache=True,
            )
            outputs = model(**model_inputs)
            past_key_values = getattr(outputs, "past_key_values", None)
            if past_key_values is None:
                raise RuntimeError("model did not expose past_key_values during CLE teacher-forced decode")
            trace_tensor = torch.tensor(
                [tokens_by_prompt[prompt_index] for prompt_index in prompt_indices],
                device=device,
                dtype=torch.long,
            )
            if trace_tensor.shape != (len(prompt_indices), DECODE_POSITION):
                raise RuntimeError(
                    f"trace token tensor shape {tuple(trace_tensor.shape)} does not match "
                    f"({len(prompt_indices)}, {DECODE_POSITION})"
                )
            final_logits = None
            for token_offset in range(DECODE_POSITION):
                teacher_token = trace_tensor[:, token_offset]
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype),
                    ],
                    dim=1,
                )
                cache_position = torch.tensor([attention_mask.shape[1] - 1], device=device, dtype=torch.long)
                model_inputs = prepare_inputs(
                    model,
                    input_ids=teacher_token[:, None],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    use_cache=True,
                )
                outputs = model(**model_inputs)
                past_key_values = getattr(outputs, "past_key_values", None)
                if past_key_values is None:
                    raise RuntimeError(
                        f"model dropped past_key_values at teacher-forced token offset {token_offset + 1}"
                    )
                if token_offset + 1 == DECODE_POSITION:
                    final_logits = outputs.logits[:, -1, :].detach().to(torch.float32).cpu()
            if final_logits is None:
                raise RuntimeError("CLE teacher-forced decode loop did not produce final logits")
            for offset, prompt_index in enumerate(prompt_indices):
                logits_by_prompt[prompt_index] = final_logits[offset].contiguous()
            append_event(
                run_events_path,
                "completed_prompt_batch",
                variant=variant,
                depth=depth,
                prompt_indices=prompt_indices,
                decode_position=DECODE_POSITION,
                teacher_forced_trace="trace_tokens.jsonl",
            )
    return logits_by_prompt


def write_logits_artifact(
    run_dir: Path,
    *,
    role: str,
    prompt_index: int,
    prompt_id: str,
    logits: Any,
    depth: int | None = None,
) -> dict[str, Any]:
    import numpy as np

    logits_dir = run_dir / "logits"
    logits_dir.mkdir(parents=True, exist_ok=True)
    if role == "bf16":
        rel = Path("logits") / f"bf16_prompt_{prompt_index:03d}.f32"
    else:
        assert depth is not None
        rel = Path("logits") / f"fp4_depth_{depth:02d}_prompt_{prompt_index:03d}.f32"
    path = run_dir / rel
    array = logits.detach().to("cpu").to(dtype=getattr(__import__("torch"), "float32")).contiguous().numpy()
    array = array.astype("<f4", copy=False)
    path.write_bytes(array.tobytes(order="C"))
    entry: dict[str, Any] = {
        "role": role,
        "prompt_index": prompt_index,
        "prompt_id": prompt_id,
        "path": str(rel),
        "dtype": "float32_le",
        "shape": [int(dim) for dim in array.shape],
        "bytes": path.stat().st_size,
        "sha256": shared.file_sha256(path),
    }
    if role == "fp4":
        entry["depth"] = depth
    return entry


def build_metrics_and_bootstrap(
    *,
    prompt_manifest: dict[str, Any],
    predicted_by_depth: dict[int, dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metric_rows = [
        {
            "prompt_index": int(row["prompt_index"]),
            "prompt_id": row["prompt_id"],
            "depth": int(row["depth"]),
            "l2_drift": float(row["l2_drift"]),
        }
        for row in raw_rows
    ]
    metrics = checker.compute_metrics(
        metric_rows,
        predicted_by_depth,
        bootstrap_samples=checker.THRESHOLDS["bootstrap_samples"],
        seed=seed,
    )
    metrics.update(
        {
            "created_at_utc": utc_now(),
            "prompt_sha256": prompt_manifest["prompt_sha256"],
            "quantization_format": QUANTIZATION_FORMAT,
            "aggregation": {
                "prompt_level": "L2 distance between BF16 and FP4 logits at decode position 1000",
                "gate_level": "mean across the 12 preregistered AIME-2025 prompts for each depth",
                "bootstrap_unit": "prompt",
            },
        }
    )
    bootstrap_ci = {
        "schema_version": f"{SCHEMA_VERSION}_bootstrap_ci",
        "metric_name": metrics["metric_name"],
        "bootstrap_samples": metrics["bootstrap_samples"],
        "bootstrap_seed": metrics["bootstrap_seed"],
        "depth_metrics": [
            {
                "depth": row["depth"],
                "mean_l2_drift": row["mean_l2_drift"],
                "bootstrap_ci95": row["bootstrap_ci95"],
            }
            for row in metrics["depth_metrics"]
        ],
    }
    return metrics, bootstrap_ci


def run(args: argparse.Namespace, argv: list[str] | None) -> int:
    run_dir = args.results_dir / args.run_id
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")
    (run_dir / "logs").mkdir(parents=True)
    stdout_log = (run_dir / "logs/stdout.log").open("w", encoding="utf-8", buffering=1)
    stderr_log = (run_dir / "logs/stderr.log").open("w", encoding="utf-8", buffering=1)
    sys.stdout = shared.Tee(sys.__stdout__, stdout_log)
    sys.stderr = shared.Tee(sys.__stderr__, stderr_log)
    run_events_path = run_dir / "run_events.jsonl"
    append_event(run_events_path, "run_started")

    random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

    prompt_manifest, prompt_reasons = shared.build_prompt_manifest(
        args.prompt_file, schema_version=SCHEMA_VERSION
    )
    environment = shared.build_environment(schema_version=SCHEMA_VERSION)
    model_provenance = shared.resolve_model_snapshot(MODEL_ID, schema_version=SCHEMA_VERSION)
    command_metadata = build_command_metadata(args, run_dir, argv)
    random_seed = {
        "schema_version": f"{SCHEMA_VERSION}_random_seed",
        "seed": args.seed,
        "determinism": {"do_sample": False, "num_beams": 1, "torch_manual_seed": args.seed},
    }
    quantization_config = {
        "schema_version": f"{SCHEMA_VERSION}_quantization_config",
        "quantization_format": QUANTIZATION_FORMAT,
        "simulator": "experimental.shared.fp4_simulator.simulate_mxfp4_e2m1",
        "native_kernel_claim": False,
        "block_size": BLOCK_SIZE,
        "block_size_note": (
            "NVFP4-compatible E2M1 simulation with 16-value blocks; simulator uses "
            "float32 scale values and does not claim native Blackwell NVFP4 kernel behavior."
        ),
        "outlier_fraction": OUTLIER_FRACTION,
        "depths": list(DEPTHS),
        "decode_position": DECODE_POSITION,
        "depth_pattern": "first N transformer layers are quantized; all other modules remain in BF16",
    }
    for name, payload in [
        ("environment.json", environment),
        ("model_provenance.json", model_provenance),
        ("prompt_manifest.json", prompt_manifest),
        ("command_metadata.json", command_metadata),
        ("random_seed.json", random_seed),
        ("quantization_config.json", quantization_config),
    ]:
        write_json(run_dir / name, payload)

    if args.model_id != MODEL_ID:
        return fail_infra(run_dir, run_events_path, [f"CLE requires model {MODEL_ID}; got {args.model_id}"])
    if args.prompt_file.resolve() != PROMPT_FILE.resolve():
        return fail_infra(run_dir, run_events_path, [f"CLE requires canonical prompt file {PROMPT_FILE}"])
    if prompt_reasons:
        return fail_infra(run_dir, run_events_path, prompt_reasons)
    if not model_provenance.get("snapshot_path"):
        return fail_infra(run_dir, run_events_path, [str(model_provenance.get("error"))])
    if args.dtype != "bfloat16":
        return fail_infra(run_dir, run_events_path, ["CLE preregisters BF16-vs-FP4; --dtype must be bfloat16"])
    if args.batch_size < 1:
        return fail_infra(run_dir, run_events_path, ["--batch-size must be positive"])

    try:
        model, tokenizer, device = shared.load_model_and_tokenizer(
            model_provenance, dtype_name=args.dtype, device_name=args.device
        )
        layers, layer_origin = shared.discover_transformer_layers(model)
        layer_stats, output_scale = quantization_error_stats(model, layers)
        bounds = derive_bounds(layer_stats, output_scale)
        derivation_path = run_dir / "derivation.md"
        derivation_path.write_text(derivation_markdown(bounds, output_scale), encoding="utf-8")
        predicted_bounds = {
            "schema_version": f"{SCHEMA_VERSION}_predicted_bounds",
            "created_at_utc": utc_now(),
            "created_before_measurement": True,
            "measurement_free_inputs_only": True,
            "model_id": MODEL_ID,
            "depths": list(DEPTHS),
            "bound_formula": "C_out * sqrt(sum_{l in first N layers} eta_l^2)",
            "derivation": "derivation.md",
            "derivation_sha256": shared.file_sha256(derivation_path),
            "layer_origin": layer_origin,
            "output_scale": output_scale,
            "bounds_by_depth": bounds,
            "layer_stats": layer_stats,
        }
        write_json(run_dir / "predicted_bounds.json", predicted_bounds)
        append_event(run_events_path, "bound_predictions_written", depths=list(DEPTHS))
        append_event(run_events_path, "derivation_locked", derivation_sha256=predicted_bounds["derivation_sha256"])

        prompts = prompt_manifest["prompts"]
        append_event(run_events_path, "measurement_started")
        tokens_by_prompt = generate_trace_tokens(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompts=prompts,
            batch_size=args.batch_size,
            run_events_path=run_events_path,
        )
        write_trace_tokens(run_dir, prompts=prompts, tokens_by_prompt=tokens_by_prompt)
        append_event(run_events_path, "trace_tokens_written", artifact_path="trace_tokens.jsonl")
        bf16_logits = capture_logits_for_trace(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompts=prompts,
            tokens_by_prompt=tokens_by_prompt,
            batch_size=args.batch_size,
            run_events_path=run_events_path,
            variant="bf16",
        )
        prompt_id_by_index = {int(row["index"]): str(row["prompt_id"]) for row in prompts}
        logit_entries: list[dict[str, Any]] = []
        for prompt_index, logits in sorted(bf16_logits.items()):
            logit_entries.append(
                write_logits_artifact(
                    run_dir,
                    role="bf16",
                    prompt_index=prompt_index,
                    prompt_id=prompt_id_by_index[prompt_index],
                    logits=logits,
                )
            )

        raw_rows: list[dict[str, Any]] = []
        raw_rows_path = run_dir / "raw_drift_rows.jsonl"
        with raw_rows_path.open("w", encoding="utf-8") as rows_handle:
            for depth in DEPTHS:
                with quantized_first_layers(layers, depth):
                    fp4_logits = capture_logits_for_trace(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        prompts=prompts,
                        tokens_by_prompt=tokens_by_prompt,
                        batch_size=args.batch_size,
                        run_events_path=run_events_path,
                        variant="fp4",
                        depth=depth,
                    )
                for prompt_index, logits in sorted(fp4_logits.items()):
                    prompt_id = prompt_id_by_index[prompt_index]
                    entry = write_logits_artifact(
                        run_dir,
                        role="fp4",
                        prompt_index=prompt_index,
                        prompt_id=prompt_id,
                        logits=logits,
                        depth=depth,
                    )
                    logit_entries.append(entry)
                    diff = bf16_logits[prompt_index].to("cpu").to(dtype=logits.dtype) - logits.to("cpu")
                    l2 = float(__import__("torch").linalg.vector_norm(diff.to(__import__("torch").float32)).item())
                    row = {
                        "schema_version": f"{SCHEMA_VERSION}_raw_drift_row",
                        "prompt_index": prompt_index,
                        "prompt_id": prompt_id,
                        "depth": depth,
                        "decode_position": DECODE_POSITION,
                        "quantization_format": QUANTIZATION_FORMAT,
                        "bf16_logits_path": f"logits/bf16_prompt_{prompt_index:03d}.f32",
                        "fp4_logits_path": entry["path"],
                        "l2_drift": l2,
                        "bf16_argmax_token_id": int(bf16_logits[prompt_index].argmax().item()),
                        "fp4_argmax_token_id": int(logits.argmax().item()),
                    }
                    raw_rows.append(row)
                    rows_handle.write(json.dumps(row, sort_keys=True) + "\n")

        logits_manifest = {
            "schema_version": f"{SCHEMA_VERSION}_logits_manifest",
            "created_at_utc": utc_now(),
            "dtype": "float32_le",
            "storage": "raw little-endian float32 logits, one vector per file",
            "decode_position": DECODE_POSITION,
            "entries": logit_entries,
        }
        write_json(run_dir / "logits_manifest.json", logits_manifest)
        predicted_by_depth = {
            int(row["depth"]): row for row in predicted_bounds["bounds_by_depth"]
        }
        metrics, bootstrap_ci = build_metrics_and_bootstrap(
            prompt_manifest=prompt_manifest,
            predicted_by_depth=predicted_by_depth,
            raw_rows=raw_rows,
            seed=args.seed,
        )
        write_json(run_dir / "drift_metrics.json", metrics)
        write_json(run_dir / "bootstrap_ci.json", bootstrap_ci)
        append_event(run_events_path, "run_completed")
        write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
        print(json.dumps({"run_dir": str(run_dir), "depth_metrics": metrics["depth_metrics"]}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        return fail_infra(run_dir, run_events_path, [repr(exc)])


def main(argv: list[str] | None = None) -> int:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"cle_theoretical_{timestamp}")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--prompt-file", type=Path, default=PROMPT_FILE)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16"], default="bfloat16")
    args = parser.parse_args(argv)
    return run(args, argv)


if __name__ == "__main__":
    raise SystemExit(main())
