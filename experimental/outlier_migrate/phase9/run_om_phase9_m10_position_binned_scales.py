#!/usr/bin/env python3
"""Run Phase 9 M10 position-binned scale tables."""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import math
import random
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.outlier_migrate.phase4 import run_om_phase4_intervention as phase4_runner
from experimental.outlier_migrate.phase9 import check_om_phase9_m10_position_binned_scales as checker
from experimental.outlier_migrate.phase9 import run_om_phase9_m2_position_conditional as m2_runner
from experimental.shared import run_phase0_branch as shared


DEFAULT_RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase9/results"
DEFAULT_PROMPT_FILE = checker.DEFAULT_PROMPT_FILE
DEFAULT_MODEL_ID = "ibm-granite/granite-4.0-h-small"
SCHEMA_VERSION = checker.SCHEMA_VERSION

POSITION_BINS = [
    {"name": "bin_0_500", "start": 1, "end": 500, "positions": [100, 200, 500], "midpoint_position": 200},
    {"name": "bin_500_2000", "start": 501, "end": 2000, "positions": [1000, 2000], "midpoint_position": 1000},
    {"name": "bin_2000_5000", "start": 2001, "end": 5000, "positions": [5000], "midpoint_position": 5000},
    {"name": "bin_5000_10000", "start": 5001, "end": 10000, "positions": [7000, 10000], "midpoint_position": 7000},
    {"name": "bin_10000_20000", "start": 10001, "end": 20000, "positions": [15000], "midpoint_position": 15000},
]
SCORING_SEGMENTS = [item for item in POSITION_BINS if int(item["start"]) <= checker.SCORING_POSITION]
RANDOM_BIN_ASSIGNMENT = {
    "bin_0_500": "bin_5000_10000",
    "bin_500_2000": "bin_0_500",
    "bin_2000_5000": "bin_10000_20000",
    "bin_5000_10000": "bin_2000_5000",
    "bin_10000_20000": "bin_500_2000",
}


def parse_prompt_file(prompt_file: Path, *, trace_count: int) -> tuple[list[dict[str, Any]], list[str]]:
    reasons: list[str] = []
    prompts: list[dict[str, Any]] = []
    if not prompt_file.is_file():
        return [], [f"prompt file missing: {prompt_file}"]
    for row_index, line in enumerate(prompt_file.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        item = json.loads(line)
        index = int(item.get("index", row_index))
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
    prompts = [row for row in prompts if int(row["index"]) < trace_count]
    if [row["index"] for row in prompts] != list(range(trace_count)):
        reasons.append(f"prompt indices are not exactly 0-{trace_count - 1}")
    if len(prompts) != trace_count:
        reasons.append(f"prompt count is not {trace_count}")
    return prompts, reasons


def build_prompt_manifest(prompt_file: Path, *, trace_count: int) -> tuple[dict[str, Any], list[str]]:
    prompts, reasons = parse_prompt_file(prompt_file, trace_count=trace_count)
    selection = "deterministic_indices_0_23"
    if trace_count != checker.TRACE_COUNT:
        selection = f"deterministic_indices_0_{trace_count - 1}_vacation_revision"
    return (
        {
            "schema_version": f"{SCHEMA_VERSION}_prompt_manifest",
            "created_at_utc": shared.utc_now(),
            "source": "AIME-2025",
            "selection": selection,
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


def write_vacation_adaptation(run_dir: Path, *, trace_count: int) -> None:
    if trace_count == checker.TRACE_COUNT:
        return
    shared.write_json(
        run_dir / "vacation_adaptation.json",
        {
            "schema_version": f"{SCHEMA_VERSION}_vacation_adaptation",
            "created_at_utc": shared.utc_now(),
            "authority": "Vacation mode V2/V4",
            "adaptation": "deterministic_trace_count_reduction",
            "original_trace_count": checker.TRACE_COUNT,
            "effective_trace_count": trace_count,
            "prompt_indices": list(range(trace_count)),
            "reason": (
                "The immediately preceding M2 Granite-Small run required a deterministic 12-trace slice "
                "after full-slice OOM and slow dynamic scoring. M10 has more dynamic score arms than M2, "
                "so the same fixed 12-trace vacation-mode slice preserves the decision surface while "
                "keeping paper progress moving."
            ),
            "invalidates_if": "Human requires the original 24-trace M10 packet before interpreting M10.",
        },
    )


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


def activation_vectors_by_layer_position(rows: list[dict[str, Any]]) -> tuple[dict[int, dict[int, list[list[float]]]], dict[int, str]]:
    by_layer_position: dict[int, dict[int, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    layer_names: dict[int, str] = {}
    for row in rows:
        layer_index = int(row["layer_index"])
        position = int(row["decode_position"])
        layer_names[layer_index] = str(row["layer_name"])
        by_layer_position[layer_index][position].append([float(value) for value in row["channel_magnitudes"]])
    return by_layer_position, layer_names


def max_activation_vector(
    by_layer_position: dict[int, dict[int, list[list[float]]]], layer_index: int, positions: list[int]
) -> list[float]:
    vectors: list[list[float]] = []
    for position in positions:
        vectors.extend(by_layer_position[layer_index].get(int(position), []))
    if not vectors:
        available = sorted(by_layer_position[layer_index])
        raise ValueError(f"no activation rows for layer {layer_index} at positions {positions}; available={available}")
    channel_count = len(vectors[0])
    return [max(float(vector[channel]) for vector in vectors) for channel in range(channel_count)]


def weight_maxima_by_layer(model: Any) -> tuple[dict[int, list[float]], str]:
    import torch

    hidden_size = int(getattr(model.config, "hidden_size"))
    layers, layer_origin = shared.discover_transformer_layers(model)
    weight_maxima: dict[int, list[float]] = {}
    with torch.no_grad():
        for layer_index, (_layer_name, layer) in enumerate(layers):
            maxima = torch.full((hidden_size,), 1e-8, dtype=torch.float32)
            for module in layer.modules():
                weight = getattr(module, "weight", None)
                if not torch.is_tensor(weight) or weight.ndim not in {2, 3}:
                    continue
                work = weight.detach().abs()
                candidates: list[Any] = []
                if weight.ndim == 2:
                    if weight.shape[1] == hidden_size:
                        candidates.append(work.float().amax(dim=0).cpu())
                    if weight.shape[0] == hidden_size:
                        candidates.append(work.float().amax(dim=1).cpu())
                else:
                    if weight.shape[2] == hidden_size:
                        candidates.append(work.float().amax(dim=(0, 1)).cpu())
                    if weight.shape[1] == hidden_size:
                        candidates.append(work.float().amax(dim=(0, 2)).cpu())
                for candidate in candidates:
                    maxima = torch.maximum(maxima, candidate.to(dtype=torch.float32))
            weight_maxima[layer_index] = [float(value) for value in maxima.tolist()]
    return weight_maxima, layer_origin


def build_scale_tables(rows: list[dict[str, Any]], model: Any) -> dict[str, Any]:
    import torch

    by_layer_position, layer_names = activation_vectors_by_layer_position(rows)
    weight_maxima, layer_origin = weight_maxima_by_layer(model)
    tables: dict[str, Any] = {
        "static_position_100": {"by_bin": {}},
        "m10_bins": {"by_bin": {}},
        "midpoint_bins": {"by_bin": {}},
        "random_bin_assignment": {"seed": checker.BOOTSTRAP_SEED, "assignment": RANDOM_BIN_ASSIGNMENT, "by_bin": {}},
    }

    def make_layer_scales(layer_index: int, positions: list[int]) -> dict[str, Any]:
        a = torch.tensor(max_activation_vector(by_layer_position, layer_index, positions), dtype=torch.float32).clamp_min(1e-8)
        w = torch.tensor(weight_maxima[layer_index], dtype=torch.float32).clamp_min(1e-8)
        if a.numel() != w.numel():
            raise ValueError(f"activation/weight channel mismatch for layer {layer_index}: {a.numel()} vs {w.numel()}")
        scale = (a.pow(0.5) / w.pow(0.5)).clamp(0.01, 100.0)
        scale = scale / scale.median().clamp_min(1e-8)
        values = [round(float(value), 8) for value in scale.tolist()]
        payload = json.dumps(values, separators=(",", ":")).encode("utf-8")
        return {
            "layer_name": layer_names.get(layer_index, f"layer_{layer_index}"),
            "source_positions": [int(position) for position in positions],
            "channel_count": len(values),
            "scale_sha256": checker.bytes_sha256(payload),
            "scale": values,
        }

    layer_indices = sorted(by_layer_position)
    for bin_spec in POSITION_BINS:
        bin_name = str(bin_spec["name"])
        static_layers = {str(layer): make_layer_scales(layer, [100]) for layer in layer_indices}
        m10_layers = {str(layer): make_layer_scales(layer, list(bin_spec["positions"])) for layer in layer_indices}
        midpoint_layers = {str(layer): make_layer_scales(layer, [int(bin_spec["midpoint_position"])]) for layer in layer_indices}
        tables["static_position_100"]["by_bin"][bin_name] = {"source": "position_100", "layers": static_layers}
        tables["m10_bins"]["by_bin"][bin_name] = {"source": "all_recorded_positions_in_bin", "layers": m10_layers}
        tables["midpoint_bins"]["by_bin"][bin_name] = {"source": "nearest_recorded_midpoint_position", "layers": midpoint_layers}
    for target_bin, source_bin in RANDOM_BIN_ASSIGNMENT.items():
        tables["random_bin_assignment"]["by_bin"][target_bin] = tables["m10_bins"]["by_bin"][source_bin]
    return {
        "schema_version": f"{SCHEMA_VERSION}_scale_tables",
        "created_at_utc": shared.utc_now(),
        "alpha": 0.5,
        "formula": "S[l,b,c] = clamp((A[l,b,c] ** 0.5) / (W[l,c] ** 0.5), 0.01, 100.0); median-normalized",
        "activation_floor": 1e-8,
        "weight_floor": 1e-8,
        "position_bins": POSITION_BINS,
        "layer_origin": layer_origin,
        "tables": tables,
    }


def make_input_scale_hook(scale: Any):
    import torch

    def hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        if not inputs:
            return inputs
        first = inputs[0]
        if torch.is_tensor(first) and first.shape and first.shape[-1] == scale.numel():
            local_scale = scale.to(device=first.device, dtype=first.dtype)
            return (first / local_scale, *inputs[1:])
        return inputs

    return hook


def scaled_quantize_module(
    *,
    module_name: str,
    module: Any,
    layer_scale: Any,
    hidden_size: int,
    scale_source: str,
) -> dict[str, Any] | None:
    import torch

    weight = getattr(module, "weight", None)
    if not torch.is_tensor(weight):
        return None
    if weight.ndim not in {2, 3}:
        return {"name": module_name, "reason": "weight tensor is not 2D or expert-bank 3D", "shape": list(weight.shape)}
    scaled_hidden_input = False
    with torch.no_grad():
        if weight.ndim == 2 and weight.shape[1] == hidden_size:
            weight.mul_(layer_scale.to(device=weight.device, dtype=weight.dtype).view(1, -1))
            module.register_forward_pre_hook(make_input_scale_hook(layer_scale.detach().cpu()))
            scaled_hidden_input = True
        elif weight.ndim == 3 and weight.shape[2] == hidden_size:
            weight.mul_(layer_scale.to(device=weight.device, dtype=weight.dtype).view(1, 1, -1))
            module.register_forward_pre_hook(make_input_scale_hook(layer_scale.detach().cpu()))
            scaled_hidden_input = True
        phase4_runner.quantize_weight_symmetric_int4_inplace(weight)
    return {
        "name": module_name,
        "shape": list(weight.shape),
        "scaled_hidden_input": bool(scaled_hidden_input),
        "scale_source": scale_source if scaled_hidden_input else "none_hidden_input_axis_not_found",
        "weight_kind": "linear_2d" if weight.ndim == 2 else "expert_bank_3d",
    }


def apply_scaled_quantization(model: Any, scale_tables: dict[str, Any], table_name: str, bin_name: str) -> dict[str, Any]:
    import torch

    hidden_size = int(getattr(model.config, "hidden_size"))
    layers, layer_origin = shared.discover_transformer_layers(model)
    table = scale_tables["tables"][table_name]["by_bin"][bin_name]
    quantized: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    processed_module_ids: set[int] = set()
    with torch.no_grad():
        for layer_index, (layer_name, layer) in enumerate(layers):
            scale_values = table["layers"][str(layer_index)]["scale"]
            layer_scale = torch.tensor(scale_values, dtype=torch.float32)
            for module_name, module in layer.named_modules():
                full_name = f"{layer_name}.{module_name}" if module_name else layer_name
                item = scaled_quantize_module(
                    module_name=full_name,
                    module=module,
                    layer_scale=layer_scale,
                    hidden_size=hidden_size,
                    scale_source=f"{table_name}:{bin_name}:layer_{layer_index}",
                )
                if item is None:
                    continue
                processed_module_ids.add(id(module))
                if "reason" in item:
                    excluded.append(item)
                else:
                    quantized.append(item)
        last_layer_index = str(max(int(key) for key in table["layers"]))
        last_scale = torch.tensor(table["layers"][last_layer_index]["scale"], dtype=torch.float32)
        for name, module in model.named_modules():
            if id(module) in processed_module_ids:
                continue
            weight = getattr(module, "weight", None)
            if not torch.is_tensor(weight):
                continue
            if phase4_runner.is_tied_lm_head(model, name, module):
                excluded.append({"name": name, "reason": "tied input/output embedding", "shape": list(weight.shape)})
                continue
            item = scaled_quantize_module(
                module_name=name,
                module=module,
                layer_scale=last_scale,
                hidden_size=hidden_size,
                scale_source=f"{table_name}:{bin_name}:outside_stack_last_layer",
            )
            if item is None:
                continue
            if "reason" in item:
                excluded.append(item)
            else:
                quantized.append(item)
    return {
        "table": table_name,
        "bin": bin_name,
        "layer_origin": layer_origin,
        "quantized_tensor_count": len(quantized),
        "scaled_hidden_input_tensor_count": sum(1 for item in quantized if item.get("scaled_hidden_input")),
        "quantized_tensors": quantized,
        "excluded_tensors": excluded,
    }


def score_dynamic_scale_segments(
    *,
    model_provenance: dict[str, Any],
    prompts: list[dict[str, Any]],
    target_tokens: dict[int, list[int]],
    scale_tables: dict[str, Any],
    table_name: str,
    segments: list[dict[str, Any]],
    batch_size: int,
    dtype_name: str,
    device_name: str,
    run_events_path: Path,
    regime_name: str,
) -> tuple[dict[int, dict[str, float]], dict[str, Any]]:
    import torch

    results: dict[int, dict[str, float]] = {}
    quantized_segments: list[dict[str, Any]] = []
    score_start = checker.SCORING_POSITION - checker.SCORING_WINDOW_TOKENS + 1
    score_end = checker.SCORING_POSITION
    previous_fast_path = None
    try:
        previous_fast_path = phase4_runner.set_granite_fast_path_enabled(False)
    except Exception:
        previous_fast_path = None
    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            batch_indices = [int(item["index"]) for item in batch]
            texts = [shared.make_prompt_text(str(item["prompt"])) for item in batch]
            target_tensor_cpu = [target_tokens[index][: checker.SCORING_POSITION] for index in batch_indices]
            attention_mask = None
            past_key_values = None
            logits = None
            nll = None
            scored = 0
            for segment in segments:
                bin_name = str(segment["name"])
                model, tokenizer, device = shared.load_model_and_tokenizer(
                    model_provenance, dtype_name=dtype_name, device_name=device_name
                )
                quantized = apply_scaled_quantization(model, scale_tables, table_name, bin_name)
                quantized_segments.append({"segment": segment, "quantization": quantized})
                normalize_cache_inputs, output_cache = phase4_runner.make_cache_helpers(model, cache_dtype=torch.float16)
                target_tensor = torch.tensor(target_tensor_cpu, dtype=torch.long, device=device)
                if attention_mask is None:
                    encoded = tokenizer(texts, padding=True, return_tensors="pt")
                    input_ids = encoded["input_ids"].to(device)
                    attention_mask = encoded["attention_mask"].to(device)
                    nll = torch.zeros((len(batch),), device=device, dtype=torch.float64)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                        cache_position = torch.arange(input_ids.shape[1], device=device)
                        model_inputs = model.prepare_inputs_for_generation(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            cache_position=cache_position,
                            use_cache=True,
                        )
                        outputs = model(**normalize_cache_inputs(model_inputs))
                        past_key_values = output_cache(outputs)
                        if past_key_values is None:
                            raise RuntimeError(f"model did not return cache for {regime_name}")
                        logits = outputs.logits[:, -1, :]
                    del encoded, input_ids, cache_position, model_inputs, outputs
                else:
                    attention_mask = attention_mask.to(device)
                    target_tensor = target_tensor.to(device)
                    if nll is not None:
                        nll = nll.to(device)
                    if logits is not None:
                        logits = logits.to(device)
                    if past_key_values is not None:
                        past_key_values = m2_runner.move_tensor_state(past_key_values, device)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                    for decode_position in range(int(segment["start"]), min(int(segment["end"]), checker.SCORING_POSITION) + 1):
                        current_target = target_tensor[:, decode_position - 1]
                        if score_start <= decode_position <= score_end:
                            log_probs = torch.log_softmax(logits.float(), dim=-1)
                            nll -= log_probs.gather(1, current_target[:, None]).squeeze(1).to(torch.float64)
                            scored += 1
                        if decode_position == checker.SCORING_POSITION:
                            break
                        attention_mask = torch.cat(
                            [
                                attention_mask,
                                torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype),
                            ],
                            dim=1,
                        )
                        cache_position = torch.tensor([attention_mask.shape[1] - 1], device=device, dtype=torch.long)
                        model_inputs = model.prepare_inputs_for_generation(
                            input_ids=current_target[:, None],
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            cache_position=cache_position,
                            use_cache=True,
                        )
                        outputs = model(**normalize_cache_inputs(model_inputs))
                        past_key_values = output_cache(outputs)
                        if past_key_values is None:
                            raise RuntimeError(f"model dropped cache for {regime_name} at {decode_position}")
                        logits = outputs.logits[:, -1, :].detach()
                if past_key_values is not None:
                    past_key_values = m2_runner.move_tensor_state(past_key_values, "cpu")
                if attention_mask is not None:
                    attention_mask = attention_mask.cpu()
                if logits is not None:
                    logits = logits.cpu()
                if nll is not None:
                    nll = nll.cpu()
                del target_tensor, model, tokenizer, normalize_cache_inputs, output_cache
                release_model_memory()
            assert nll is not None
            for offset, prompt_index in enumerate(batch_indices):
                mean_nll = float((nll[offset] / max(1, scored)).item())
                results[prompt_index] = {
                    "mean_nll": mean_nll,
                    "perplexity": float(math.exp(min(80.0, mean_nll))),
                    "scored_tokens": scored,
                    "score_start": score_start,
                    "score_end": score_end,
                }
            with run_events_path.open("a", encoding="utf-8") as event_handle:
                event_handle.write(
                    json.dumps(
                        {
                            "created_at_utc": shared.utc_now(),
                            "event": "completed_dynamic_scale_scoring_batch",
                            "regime": regime_name,
                            "prompt_indices": batch_indices,
                            "table": table_name,
                            "segments": segments,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            del attention_mask, past_key_values, logits, nll
            release_model_memory()
    finally:
        if previous_fast_path is not None:
            phase4_runner.set_granite_fast_path_enabled(bool(previous_fast_path))
    return results, {"regime": regime_name, "table": table_name, "segments": quantized_segments}


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "median_recovery": float(median(values)) if values else None,
        "mean_recovery": float(mean(values)) if values else None,
        "bootstrap_ci95": checker.bootstrap_median(values),
        "included_trace_count": len(values),
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


def read_score_cache(score_dir: Path | None, regime: str, *, expected_prompt_indices: set[int]) -> dict[int, dict[str, float]] | None:
    if score_dir is None:
        return None
    path = score_dir / "score_cache" / f"{regime}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    scores = {int(index): row for index, row in payload.get("scores", {}).items()}
    if set(scores) != expected_prompt_indices:
        return None
    return scores


def build_metrics(
    *,
    run_dir: Path,
    prompt_manifest: dict[str, Any],
    model_provenance: dict[str, Any],
    per_trace_rows: list[dict[str, Any]],
    activation_path: Path,
    bf16_trace_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    included = [row for row in per_trace_rows if not bool(row["no_recoverable_static_gap"])]
    by_regime = {
        regime: [float(row["recoveries"][regime]) for row in included]
        for regime in ["m10_position_binned", "midpoint_matched_cost", "random_bin_assignment"]
    }
    primary = summarize(by_regime["m10_position_binned"])
    midpoint = summarize(by_regime["midpoint_matched_cost"])
    random_bin = summarize(by_regime["random_bin_assignment"])
    no_gap_count = len(per_trace_rows) - len(included)
    for item in [primary, midpoint, random_bin]:
        item["total_trace_count"] = len(per_trace_rows)
        item["no_recoverable_static_gap_count"] = no_gap_count
        item["no_recoverable_static_gap_fraction"] = no_gap_count / len(per_trace_rows) if per_trace_rows else 0.0
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
        "calibration_positions": list(checker.CALIBRATION_POSITIONS),
        "scoring_position": checker.SCORING_POSITION,
        "scoring_window_tokens": checker.SCORING_WINDOW_TOKENS,
        "metric_name": "positive-static-SmoothQuant-gap per-trace recovery",
        "metric_formula": "1 - (perplexity_regime - perplexity_BF16) / (perplexity_static_smoothquant - perplexity_BF16)",
        "primary_result": primary,
        "thresholds": checker.THRESHOLDS,
        "artifacts": {
            "activation_magnitudes": "activation_magnitudes.jsonl.gz",
            "activation_magnitudes_sha256": shared.file_sha256(activation_path),
            "bf16_traces": "bf16_traces.jsonl.gz",
            "bf16_traces_sha256": shared.file_sha256(bf16_trace_path),
            "run_dir": str(run_dir),
        },
    }
    bootstrap = {
        "schema_version": f"{SCHEMA_VERSION}_bootstrap_ci",
        "metric_name": metrics["metric_name"],
        "bootstrap_samples": checker.BOOTSTRAP_SAMPLES,
        "bootstrap_seed": checker.BOOTSTRAP_SEED,
        "primary_result": primary,
    }
    controls = {
        "schema_version": f"{SCHEMA_VERSION}_control_metrics",
        "created_at_utc": shared.utc_now(),
        "controls": {
            "static_smoothquant": {"median_recovery": 0.0},
            "midpoint_matched_cost": midpoint,
            "random_bin_assignment": random_bin,
        },
        "m10_minus_midpoint_median": (
            None
            if primary["median_recovery"] is None or midpoint["median_recovery"] is None
            else float(primary["median_recovery"]) - float(midpoint["median_recovery"])
        ),
        "m10_minus_random_bin_median": (
            None
            if primary["median_recovery"] is None or random_bin["median_recovery"] is None
            else float(primary["median_recovery"]) - float(random_bin["median_recovery"])
        ),
    }
    return metrics, bootstrap, controls


def main(argv: list[str] | None = None) -> int:
    shared.SCHEMA_VERSION = SCHEMA_VERSION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"om_phase9_m10_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=checker.BOOTSTRAP_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--reuse-activation-run-dir", type=Path)
    parser.add_argument("--reuse-trace-run-dir", type=Path)
    parser.add_argument("--reuse-score-run-dir", type=Path)
    parser.add_argument("--trace-count", type=int, default=checker.TRACE_COUNT)
    args = parser.parse_args(argv)

    if args.model_id not in checker.MODEL_SNAPSHOTS:
        raise SystemExit(f"M10 model is not preregistered: {args.model_id}")
    if args.prompt_file.resolve() != DEFAULT_PROMPT_FILE.resolve():
        raise SystemExit(f"M10 requires canonical prompt file {DEFAULT_PROMPT_FILE}")
    if shared.file_sha256(args.prompt_file) != checker.EXPECTED_PROMPT_FILE_SHA256:
        raise SystemExit("canonical AIME-2025 indices 0-23 prompt file hash drifted")
    if args.seed != checker.BOOTSTRAP_SEED:
        raise SystemExit("M10 preregisters bootstrap/random seed 20260514")
    if args.trace_count != checker.TRACE_COUNT and not (12 <= args.trace_count < checker.TRACE_COUNT):
        raise SystemExit("vacation trace-count adaptation requires 12 <= trace-count < 24")

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

    def m10_excepthook(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        write_failure_packet(run_dir, run_events_path, exc)
        previous_excepthook(exc_type, exc, tb)

    sys.excepthook = m10_excepthook
    run_events_path.write_text(
        json.dumps({"created_at_utc": shared.utc_now(), "event": "run_started"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

    prompt_manifest, prompt_reasons = build_prompt_manifest(args.prompt_file, trace_count=args.trace_count)
    environment = shared.build_environment(schema_version=SCHEMA_VERSION)
    model_provenance = m2_runner.resolve_model_snapshot_light(args.model_id)
    model_provenance["schema_version"] = f"{SCHEMA_VERSION}_model_provenance"
    command_metadata = {
        "schema_version": f"{SCHEMA_VERSION}_command",
        "created_at_utc": shared.utc_now(),
        "argv": sys.argv if argv is None else ["run_om_phase9_m10_position_binned_scales.py", *argv],
        "cwd": str(Path.cwd()),
        "branch": "outlier_migrate_phase9_m10_position_binned_scales",
        "run_dir": str(run_dir),
        "batch_size": args.batch_size,
        "reuse_activation_run_dir": str(args.reuse_activation_run_dir) if args.reuse_activation_run_dir else None,
        "reuse_trace_run_dir": str(args.reuse_trace_run_dir) if args.reuse_trace_run_dir else None,
        "reuse_score_run_dir": str(args.reuse_score_run_dir) if args.reuse_score_run_dir else None,
        "trace_count": args.trace_count,
    }
    random_seed = {
        "schema_version": f"{SCHEMA_VERSION}_random_seed",
        "seed": args.seed,
        "determinism": {"do_sample": False, "num_beams": 1, "torch_manual_seed": args.seed},
    }
    decoding_config = {
        "schema_version": f"{SCHEMA_VERSION}_decoding_config",
        "max_new_tokens_for_scoring": checker.SCORING_POSITION,
        "max_new_tokens_for_calibration": max(checker.CALIBRATION_POSITIONS),
        "do_sample": False,
        "num_beams": 1,
        "eos_policy": "record first EOS but continue to fixed decode positions",
        "scoring_position": checker.SCORING_POSITION,
        "scoring_window_tokens": checker.SCORING_WINDOW_TOKENS,
        "target_trace": "BF16 deterministic greedy trace",
        "position_bins": POSITION_BINS,
        "random_bin_assignment": RANDOM_BIN_ASSIGNMENT,
    }
    quantization_config = {
        "schema_version": f"{SCHEMA_VERSION}_quantization_config",
        "weight_bits": 4,
        "scheme": "symmetric_per_output_channel_int4_after_scale_folding",
        "signed_integer_range": [-7, 7],
        "activation_dtype": "float16",
        "scale_alpha": 0.5,
        "implementation_note": (
            "Scale vectors are folded into dequantized INT4 weights on hidden-input axes, "
            "with matching activation pre-hooks that divide the incoming hidden state."
        ),
    }
    shared.write_json(run_dir / "prompt_manifest.json", prompt_manifest)
    write_vacation_adaptation(run_dir, trace_count=args.trace_count)
    shared.write_json(run_dir / "environment.json", environment)
    phase4_runner.write_environment_text(run_dir / "environment.txt", environment)
    shared.write_json(run_dir / "model_provenance.json", model_provenance)
    shared.write_json(run_dir / "command_metadata.json", command_metadata)
    shared.write_json(run_dir / "random_seed.json", random_seed)
    shared.write_json(run_dir / "decoding_config.json", decoding_config)
    shared.write_json(run_dir / "quantization_config.json", quantization_config)
    if prompt_reasons:
        shared.write_json(run_dir / "infra_error.json", {"reasons": prompt_reasons})
        shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
        checker.evaluate(run_dir)
        return 1
    if not model_provenance.get("snapshot_path"):
        shared.write_json(run_dir / "infra_error.json", {"reasons": [model_provenance.get("error")]})
        shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
        checker.evaluate(run_dir)
        return 1
    if model_provenance.get("hf_snapshot_commit") != checker.MODEL_SNAPSHOTS[args.model_id]:
        shared.write_json(run_dir / "infra_error.json", {"reasons": ["model snapshot commit mismatch"]})
        shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
        checker.evaluate(run_dir)
        return 1

    prompts = prompt_manifest["prompts"]
    prompt_indices = {int(row["index"]) for row in prompts}
    activation_path = run_dir / "activation_magnitudes.jsonl.gz"
    trace_path = run_dir / "bf16_traces.jsonl.gz"
    if args.reuse_activation_run_dir:
        reuse_dir = args.reuse_activation_run_dir.resolve()
        copied_rows = m2_runner.copy_filtered_jsonl_gz(
            reuse_dir / "activation_magnitudes.jsonl.gz",
            activation_path,
            prompt_indices=prompt_indices,
        )
        activation_manifest = json.loads((reuse_dir / "activation_magnitude_manifest.json").read_text(encoding="utf-8"))
        activation_manifest["created_at_utc"] = shared.utc_now()
        activation_manifest["source_run_dir"] = str(reuse_dir)
        activation_manifest["vacation_filtered_prompt_indices"] = sorted(prompt_indices)
        activation_manifest["filtered_row_count"] = copied_rows
        shared.write_json(run_dir / "activation_magnitude_manifest.json", activation_manifest)
    else:
        model, tokenizer, device = shared.load_model_and_tokenizer(model_provenance, dtype_name=args.dtype, device_name=args.device)
        activation_manifest = shared.capture_activation_magnitudes(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompts=prompts,
            positions=checker.CALIBRATION_POSITIONS,
            max_new_tokens=max(checker.CALIBRATION_POSITIONS),
            batch_size=args.batch_size,
            output_path=activation_path,
            run_events_path=run_events_path,
        )
        shared.write_json(run_dir / "activation_magnitude_manifest.json", activation_manifest)
        del model, tokenizer, device
        release_model_memory()

    model, tokenizer, device = shared.load_model_and_tokenizer(model_provenance, dtype_name=args.dtype, device_name=args.device)
    scale_tables = build_scale_tables(list(shared.iter_activation_rows(activation_path)), model)
    shared.write_json(run_dir / "scale_tables.json", scale_tables)
    shared.write_json(
        run_dir / "scale_table_manifest.json",
        {
            "schema_version": f"{SCHEMA_VERSION}_scale_table_manifest",
            "created_at_utc": shared.utc_now(),
            "scale_tables_path": str((run_dir / "scale_tables.json").resolve()),
            "scale_tables_sha256": shared.file_sha256(run_dir / "scale_tables.json"),
            "implementation_route": "fold_scale_into_dequantized_int4_weights_with_activation_pre_hooks",
            "position_bins": POSITION_BINS,
        },
    )
    del model, tokenizer, device
    release_model_memory()

    if args.reuse_trace_run_dir:
        reuse_trace_dir = args.reuse_trace_run_dir.resolve()
        copied_traces = m2_runner.copy_filtered_jsonl_gz(
            reuse_trace_dir / "bf16_traces.jsonl.gz",
            trace_path,
            prompt_indices=prompt_indices,
        )
        trace_manifest = json.loads((reuse_trace_dir / "bf16_trace_manifest.json").read_text(encoding="utf-8"))
        trace_manifest["created_at_utc"] = shared.utc_now()
        trace_manifest["source_run_dir"] = str(reuse_trace_dir)
        trace_manifest["vacation_filtered_prompt_indices"] = sorted(prompt_indices)
        trace_manifest["filtered_trace_count"] = copied_traces
        shared.write_json(run_dir / "bf16_trace_manifest.json", trace_manifest)
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

    all_scores: dict[str, dict[int, dict[str, float]]] = {}
    expected_prompt_indices = {int(item["index"]) for item in prompts}
    score_reuse_dir = args.reuse_score_run_dir.resolve() if args.reuse_score_run_dir else None
    cached_bf16 = read_score_cache(score_reuse_dir, "bf16", expected_prompt_indices=expected_prompt_indices)
    if cached_bf16 is not None:
        all_scores["bf16"] = cached_bf16
        write_score_cache(run_dir, "bf16", all_scores["bf16"])
        with run_events_path.open("a", encoding="utf-8") as event_handle:
            event_handle.write(
                json.dumps(
                    {
                        "created_at_utc": shared.utc_now(),
                        "event": "reused_score_cache",
                        "regime": "bf16",
                        "source_run_dir": str(score_reuse_dir),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    else:
        model, tokenizer, device = shared.load_model_and_tokenizer(model_provenance, dtype_name=args.dtype, device_name=args.device)
        all_scores["bf16"] = phase4_runner.score_targets(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompts=prompts,
            target_tokens=target_tokens,
            max_new_tokens=checker.SCORING_POSITION,
            batch_size=args.batch_size,
            use_float16_autocast=False,
            run_events_path=run_events_path,
            regime_name="bf16",
        )
        write_score_cache(run_dir, "bf16", all_scores["bf16"])
        del model, tokenizer, device
        release_model_memory()

    excluded_by_regime: dict[str, Any] = {}
    cached_static = read_score_cache(score_reuse_dir, "static_smoothquant", expected_prompt_indices=expected_prompt_indices)
    if cached_static is not None:
        all_scores["static_smoothquant"] = cached_static
        write_score_cache(run_dir, "static_smoothquant", all_scores["static_smoothquant"])
        excluded_by_regime["static_smoothquant"] = {"regime": "static_smoothquant", "reused_score_cache": str(score_reuse_dir)}
    else:
        model, tokenizer, device = shared.load_model_and_tokenizer(model_provenance, dtype_name=args.dtype, device_name=args.device)
        excluded_by_regime["static_smoothquant"] = apply_scaled_quantization(
            model, scale_tables, "static_position_100", "bin_0_500"
        )
        all_scores["static_smoothquant"] = phase4_runner.score_targets(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompts=prompts,
            target_tokens=target_tokens,
            max_new_tokens=checker.SCORING_POSITION,
            batch_size=args.batch_size,
            use_float16_autocast=True,
            run_events_path=run_events_path,
            regime_name="static_smoothquant",
        )
        write_score_cache(run_dir, "static_smoothquant", all_scores["static_smoothquant"])
        del model, tokenizer, device
        release_model_memory()

    for regime, table_name in [
        ("m10_position_binned", "m10_bins"),
        ("midpoint_matched_cost", "midpoint_bins"),
        ("random_bin_assignment", "random_bin_assignment"),
    ]:
        cached_scores = read_score_cache(score_reuse_dir, regime, expected_prompt_indices=expected_prompt_indices)
        if cached_scores is not None:
            all_scores[regime] = cached_scores
            write_score_cache(run_dir, regime, all_scores[regime])
            excluded_by_regime[regime] = {"regime": regime, "table": table_name, "reused_score_cache": str(score_reuse_dir)}
            continue
        all_scores[regime], excluded_by_regime[regime] = score_dynamic_scale_segments(
            model_provenance=model_provenance,
            prompts=prompts,
            target_tokens=target_tokens,
            scale_tables=scale_tables,
            table_name=table_name,
            segments=SCORING_SEGMENTS,
            batch_size=args.batch_size,
            dtype_name=args.dtype,
            device_name=args.device,
            run_events_path=run_events_path,
            regime_name=regime,
        )
        write_score_cache(run_dir, regime, all_scores[regime])

    shared.write_json(
        run_dir / "excluded_tensors.json",
        {"schema_version": f"{SCHEMA_VERSION}_excluded_tensors", "created_at_utc": shared.utc_now(), "by_regime": excluded_by_regime},
    )

    per_trace_rows: list[dict[str, Any]] = []
    for prompt in prompts:
        index = int(prompt["index"])
        perplexities = {regime: float(scores[index]["perplexity"]) for regime, scores in all_scores.items()}
        mean_nll = {regime: float(scores[index]["mean_nll"]) for regime, scores in all_scores.items()}
        static_gap = perplexities["static_smoothquant"] - perplexities["bf16"]
        no_gap = static_gap <= 0.0
        recoveries = {}
        for regime in ["m10_position_binned", "midpoint_matched_cost", "random_bin_assignment"]:
            recoveries[regime] = None if no_gap else 1.0 - (perplexities[regime] - perplexities["bf16"]) / static_gap
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
    shared.write_json(
        run_dir / "per_trace_metrics.json",
        {"schema_version": f"{SCHEMA_VERSION}_per_trace_metrics", "created_at_utc": shared.utc_now(), "traces": per_trace_rows},
    )
    metrics, bootstrap, controls = build_metrics(
        run_dir=run_dir,
        prompt_manifest=prompt_manifest,
        model_provenance=model_provenance,
        per_trace_rows=per_trace_rows,
        activation_path=activation_path,
        bf16_trace_path=trace_path,
    )
    shared.write_json(run_dir / "metrics.json", metrics)
    shared.write_json(run_dir / "bootstrap_ci.json", bootstrap)
    shared.write_json(run_dir / "control_metrics.json", controls)
    run_events_path.open("a", encoding="utf-8").write(json.dumps({"created_at_utc": shared.utc_now(), "event": "run_completed"}, sort_keys=True) + "\n")
    print(json.dumps({"run_dir": str(run_dir), "primary_result": metrics["primary_result"]}, indent=2, sort_keys=True))
    sys.stdout.flush()
    sys.stderr.flush()
    shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
    result = checker.evaluate(run_dir)
    print(json.dumps({"checker_decision": result["decision"], "artifact_complete": result.get("artifact_complete", False)}, indent=2, sort_keys=True))
    shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
    checker.evaluate(run_dir)
    sys.excepthook = previous_excepthook
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
