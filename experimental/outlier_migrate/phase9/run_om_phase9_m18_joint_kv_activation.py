#!/usr/bin/env python3
"""Run Phase 9 M18 joint KV-cache + activation protection."""

from __future__ import annotations

import argparse
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
from experimental.outlier_migrate.phase9 import check_om_phase9_m18_joint_kv_activation as checker
from experimental.outlier_migrate.phase9 import run_om_phase9_m11_ema_drift as m11_runner
from experimental.outlier_migrate.phase9 import run_om_phase9_m2_position_conditional as m2_runner
from experimental.shared import run_phase0_branch as shared


DEFAULT_RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase9/results"
DEFAULT_PROMPT_FILE = checker.DEFAULT_PROMPT_FILE
DEFAULT_MODEL_ID = checker.MODEL_ID
SCHEMA_VERSION = checker.SCHEMA_VERSION
ACTIVATION_POSITIONS = (100, 1000, 5000, 10000)


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
    phase4_runner.release_model_memory()


def top_channels(values: list[float], count: int) -> list[int]:
    return sorted(range(len(values)), key=lambda channel: (-float(values[channel]), channel))[:count]


def filter_activation_rows(source_path: Path, output_path: Path, *, prompt_indices: set[int]) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    layer_indices: set[int] = set()
    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        for row in shared.iter_activation_rows(source_path):
            if int(row["decode_position"]) not in ACTIVATION_POSITIONS:
                continue
            if int(row.get("prompt_index", -1)) not in prompt_indices:
                continue
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            rows += 1
            layer_indices.add(int(row["layer_index"]))
    return {
        "schema_version": f"{SCHEMA_VERSION}_activation_magnitude_manifest",
        "created_at_utc": shared.utc_now(),
        "artifact": "activation_magnitudes.jsonl.gz",
        "artifact_sha256": shared.file_sha256(output_path),
        "source_artifact": str(source_path),
        "source_artifact_sha256": shared.file_sha256(source_path),
        "positions": list(ACTIVATION_POSITIONS),
        "prompt_indices": sorted(prompt_indices),
        "row_count": rows,
        "layer_count": len(layer_indices),
    }


def activation_means_by_layer_position(
    rows: list[dict[str, Any]],
) -> tuple[dict[int, dict[int, list[float]]], dict[int, str]]:
    grouped: dict[int, dict[int, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    layer_names: dict[int, str] = {}
    for row in rows:
        layer_index = int(row["layer_index"])
        position = int(row["decode_position"])
        layer_names[layer_index] = str(row["layer_name"])
        grouped[layer_index][position].append([float(value) for value in row["channel_magnitudes"]])
    means_by_layer: dict[int, dict[int, list[float]]] = {}
    for layer_index, by_position in grouped.items():
        means_by_layer[layer_index] = {}
        for position, vectors in by_position.items():
            channel_count = len(vectors[0])
            means_by_layer[layer_index][position] = [
                float(mean(vector[channel] for vector in vectors)) for channel in range(channel_count)
            ]
    return means_by_layer, layer_names


def attention_layer_indices_from_config(config: Any, total_layers: int) -> list[int]:
    for attr in ("layers_block_type", "layer_types"):
        values = getattr(config, attr, None)
        if isinstance(values, list) and values:
            return [idx for idx, value in enumerate(values) if str(value).lower() == "attention"]
    return list(range(total_layers))


def tensor_flat_channel_magnitudes(tensor: Any) -> list[float]:
    import torch

    if not torch.is_tensor(tensor) or tensor.numel() == 0:
        return []
    data = tensor.detach().float().abs()
    if data.ndim == 4:
        # Hugging Face cache convention: [batch, heads, sequence, head_dim].
        return [float(value) for value in data.mean(dim=(0, 2)).reshape(-1).cpu().tolist()]
    if data.ndim >= 2:
        reduce_dims = tuple(range(data.ndim - 1))
        return [float(value) for value in data.mean(dim=reduce_dims).reshape(-1).cpu().tolist()]
    return [float(value) for value in data.reshape(-1).cpu().tolist()]


def extract_cache_lists(cache: Any) -> tuple[list[Any], list[Any]]:
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if isinstance(key_cache, list) and isinstance(value_cache, list):
        return key_cache, value_cache
    layers = getattr(cache, "layers", None)
    if isinstance(layers, list):
        keys: list[Any] = []
        values: list[Any] = []
        for layer in layers:
            keys.append(getattr(layer, "keys", None) or getattr(layer, "key_cache", None))
            values.append(getattr(layer, "values", None) or getattr(layer, "value_cache", None))
        return keys, values
    if isinstance(cache, (list, tuple)):
        keys = []
        values = []
        for item in cache:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                keys.append(item[0])
                values.append(item[1])
        return keys, values
    return [], []


def capture_kv_cache_channel_evidence(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    prompts: list[dict[str, Any]],
    target_tokens: dict[int, list[int]],
    max_new_tokens: int,
    batch_size: int,
    run_events_path: Path,
) -> dict[str, Any]:
    import torch

    normalize_cache_inputs, output_cache = phase4_runner.make_cache_helpers(model)
    attention_indices = attention_layer_indices_from_config(model.config, int(getattr(model.config, "num_hidden_layers", 0)))
    key_vectors: dict[int, list[list[float]]] = defaultdict(list)
    value_vectors: dict[int, list[list[float]]] = defaultdict(list)
    shapes: dict[int, dict[str, Any]] = {}

    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            batch_indices = [int(item["index"]) for item in batch]
            texts = [shared.make_prompt_text(str(item["prompt"])) for item in batch]
            encoded = tokenizer(texts, padding=True, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            target_tensor = torch.tensor(
                [target_tokens[index][:max_new_tokens] for index in batch_indices],
                dtype=torch.long,
                device=device,
            )
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
                raise RuntimeError("model did not return cache for M18 KV evidence")
            for decode_position in range(1, max_new_tokens + 1):
                current_target = target_tensor[:, decode_position - 1]
                if decode_position == max_new_tokens:
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
                    raise RuntimeError(f"model dropped cache during M18 KV evidence at {decode_position}")
            key_cache, value_cache = extract_cache_lists(past_key_values)
            for layer_index in attention_indices:
                if layer_index >= len(key_cache):
                    continue
                key_tensor = key_cache[layer_index]
                value_tensor = value_cache[layer_index] if layer_index < len(value_cache) else None
                key_mag = tensor_flat_channel_magnitudes(key_tensor)
                value_mag = tensor_flat_channel_magnitudes(value_tensor)
                if key_mag:
                    key_vectors[layer_index].append(key_mag)
                    shapes.setdefault(layer_index, {})["key_shape"] = list(getattr(key_tensor, "shape", []))
                if value_mag:
                    value_vectors[layer_index].append(value_mag)
                    shapes.setdefault(layer_index, {})["value_shape"] = list(getattr(value_tensor, "shape", []))
            with run_events_path.open("a", encoding="utf-8") as event_handle:
                event_handle.write(
                    json.dumps(
                        {
                            "created_at_utc": shared.utc_now(),
                            "event": "captured_kv_evidence_batch",
                            "batch_indices": batch_indices,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

    layers: dict[str, Any] = {}
    for layer_index in attention_indices:
        key_rows = key_vectors.get(layer_index, [])
        value_rows = value_vectors.get(layer_index, [])
        key_mean: list[float] = []
        value_mean: list[float] = []
        if key_rows:
            key_mean = [float(mean(row[channel] for row in key_rows)) for channel in range(len(key_rows[0]))]
        if value_rows:
            value_mean = [float(mean(row[channel] for row in value_rows)) for channel in range(len(value_rows[0]))]
        key_count = max(1, math.ceil(len(key_mean) * 0.01)) if key_mean else 0
        value_count = max(1, math.ceil(len(value_mean) * 0.01)) if value_mean else 0
        layers[str(layer_index)] = {
            "layer_index": layer_index,
            "key_cache_accessible": bool(key_mean),
            "value_cache_accessible": bool(value_mean),
            "key_flat_channel_count": len(key_mean),
            "value_flat_channel_count": len(value_mean),
            "key_top_channels": top_channels(key_mean, key_count) if key_mean else [],
            "value_top_channels": top_channels(value_mean, value_count) if value_mean else [],
            "key_shape": shapes.get(layer_index, {}).get("key_shape"),
            "value_shape": shapes.get(layer_index, {}).get("value_shape"),
            "mapping_note": "KV flattened channel index follows [head, head_dim] flattening for 4D HF caches.",
        }
    accessible = sum(1 for layer in layers.values() if layer["key_cache_accessible"])
    coverage = accessible / len(attention_indices) if attention_indices else 0.0
    return {
        "schema_version": f"{SCHEMA_VERSION}_kv_cache_channel_evidence",
        "created_at_utc": shared.utc_now(),
        "attention_layer_indices": attention_indices,
        "attention_layer_count": len(attention_indices),
        "key_cache_accessible_attention_layer_count": accessible,
        "key_cache_attention_layer_coverage": coverage,
        "selection_position": checker.SCORING_POSITION,
        "selection_basis": "mean absolute BF16 key/value cache magnitude over deterministic traces 0-11",
        "layers": layers,
    }


def map_activation_to_kv_channels(channels: list[int], flat_count: int) -> list[int]:
    if flat_count <= 0:
        return []
    return sorted({int(channel) % flat_count for channel in channels})


def build_protected_sets(
    activation_rows: list[dict[str, Any]],
    kv_evidence: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    means_by_layer, layer_names = activation_means_by_layer_position(activation_rows)
    missing: list[str] = []
    for layer_index, by_position in means_by_layer.items():
        absent = [position for position in ACTIVATION_POSITIONS if position not in by_position]
        if absent:
            missing.append(f"layer {layer_index}: missing positions {absent}")
    if missing:
        raise RuntimeError("M18 requires activation evidence at preregistered positions; " + "; ".join(missing[:4]))

    rng = random.Random(checker.BOOTSTRAP_SEED)
    regime_layers: dict[str, dict[str, Any]] = {regime: {} for regime in checker.REGIMES if regime != "bf16"}
    kv_layers_by_regime: dict[str, dict[str, Any]] = {regime: {} for regime in checker.REGIMES if regime != "bf16"}
    kv_layers = kv_evidence.get("layers", {})

    for layer_index in sorted(means_by_layer):
        values_100 = means_by_layer[layer_index][100]
        channel_count = len(values_100)
        top1_count = max(1, math.ceil(channel_count * 0.01))
        static_channels = sorted(top_channels(values_100, top1_count))
        union_channels = sorted(
            {
                channel
                for position in ACTIVATION_POSITIONS
                for channel in top_channels(means_by_layer[layer_index][position], top1_count)
            }
        )
        random_channels = sorted(rng.sample(range(channel_count), len(union_channels)))
        empty_channels: list[int] = []
        specs = {
            "static_activation_1pct": (static_channels, "activation_top1_position_100"),
            "kivi_key_only": (empty_channels, "no_activation_protection_kivi_key_only"),
            "m18_activation_k": (union_channels, "activation_union_positions_100_1000_5000_10000"),
            "m18_activation_kv": (union_channels, "activation_union_positions_100_1000_5000_10000"),
            "random_coupled_activation_k": (random_channels, "seeded_random_matched_to_activation_union_count"),
        }
        for regime, (channels, source) in specs.items():
            regime_layers[regime][str(layer_index)] = {
                "layer_name": layer_names[layer_index],
                "channel_count": channel_count,
                "requested_top_k": top1_count,
                "protected_count": len(channels),
                "protected_channels": channels,
                "source": source,
                "activation_positions": list(ACTIVATION_POSITIONS) if "union" in source else [100],
            }

        kv_layer = kv_layers.get(str(layer_index))
        if not kv_layer:
            continue
        key_flat_count = int(kv_layer.get("key_flat_channel_count", 0))
        value_flat_count = int(kv_layer.get("value_flat_channel_count", 0))
        kivi_key = [int(channel) for channel in kv_layer.get("key_top_channels", [])]
        union_key = map_activation_to_kv_channels(union_channels, key_flat_count)
        union_value = map_activation_to_kv_channels(union_channels, value_flat_count)
        random_key = sorted(rng.sample(range(key_flat_count), len(union_key))) if key_flat_count and union_key else []
        kv_layers_by_regime["kivi_key_only"][str(layer_index)] = {
            "key_protected_channels": kivi_key,
            "value_protected_channels": [],
            "source": "key_cache_top1_by_key_magnitude_no_activation_cross_reference",
        }
        kv_layers_by_regime["m18_activation_k"][str(layer_index)] = {
            "key_protected_channels": union_key,
            "value_protected_channels": [],
            "source": "activation_union_channels_mapped_to_flat_key_cache_channels",
        }
        kv_layers_by_regime["m18_activation_kv"][str(layer_index)] = {
            "key_protected_channels": union_key,
            "value_protected_channels": union_value,
            "source": "activation_union_channels_mapped_to_flat_key_and_value_cache_channels",
        }
        kv_layers_by_regime["random_coupled_activation_k"][str(layer_index)] = {
            "key_protected_channels": random_key,
            "value_protected_channels": [],
            "source": "seeded_random_key_channels_matched_to_m18_activation_k_count",
        }

    protected_sets = {
        "schema_version": f"{SCHEMA_VERSION}_protected_sets",
        "created_at_utc": shared.utc_now(),
        "selection_basis": "mean absolute activation and KV-cache channel magnitudes over deterministic AIME-2025 traces 0-11",
        "activation_positions": list(ACTIVATION_POSITIONS),
        "kv_mapping": "hidden activation channel c maps to flattened KV cache channel c modulo flat_kv_channel_count",
        "regimes": {},
    }
    for regime in checker.REGIMES:
        if regime == "bf16":
            continue
        protected_sets["regimes"][regime] = {
            "layers": regime_layers[regime],
            "kv_layers": kv_layers_by_regime[regime],
        }

    hook_coverage = {
        "schema_version": f"{SCHEMA_VERSION}_hook_coverage",
        "created_at_utc": shared.utc_now(),
        "attention_layer_indices": kv_evidence.get("attention_layer_indices", []),
        "attention_layer_count": kv_evidence.get("attention_layer_count", 0),
        "key_cache_accessible_attention_layer_count": kv_evidence.get("key_cache_accessible_attention_layer_count", 0),
        "key_cache_attention_layer_coverage": kv_evidence.get("key_cache_attention_layer_coverage", 0.0),
        "value_cache_accessible_attention_layer_count": sum(
            1 for layer in kv_evidence.get("layers", {}).values() if layer.get("value_cache_accessible")
        ),
        "source": "past_key_values.key_cache/value_cache inspection; no model source modification",
    }
    return protected_sets, hook_coverage


def quantize_cache_tensor_inplace(tensor: Any, protected_flat_channels: set[int]) -> bool:
    import torch

    if not torch.is_tensor(tensor) or tensor.numel() == 0 or not tensor.is_floating_point():
        return False
    original_dtype = tensor.dtype
    if tensor.ndim == 4:
        batch, heads, seq, head_dim = tensor.shape
        flat = tensor.detach().float().permute(0, 2, 1, 3).reshape(batch * seq, heads * head_dim)
        restore = flat[:, sorted(protected_flat_channels)].clone() if protected_flat_channels else None
        scale = flat.abs().amax(dim=0, keepdim=True).clamp_min(1e-8) / 7.0
        quantized = torch.clamp(torch.round(flat / scale), -7, 7) * scale
        if restore is not None:
            quantized[:, sorted(protected_flat_channels)] = restore
        updated = quantized.reshape(batch, seq, heads, head_dim).permute(0, 2, 1, 3).to(dtype=original_dtype)
        tensor.copy_(updated)
        return True
    if tensor.ndim >= 2:
        flat = tensor.detach().float().reshape(-1, tensor.shape[-1])
        protected = {channel for channel in protected_flat_channels if channel < flat.shape[1]}
        restore = flat[:, sorted(protected)].clone() if protected else None
        scale = flat.abs().amax(dim=0, keepdim=True).clamp_min(1e-8) / 7.0
        quantized = torch.clamp(torch.round(flat / scale), -7, 7) * scale
        if restore is not None:
            quantized[:, sorted(protected)] = restore
        tensor.copy_(quantized.reshape(tensor.shape).to(dtype=original_dtype))
        return True
    return False


def apply_cache_policy(cache: Any, protected_sets: dict[str, Any], regime: str) -> dict[str, Any]:
    key_cache, value_cache = extract_cache_lists(cache)
    kv_layers = protected_sets["regimes"][regime].get("kv_layers", {})
    events: list[dict[str, Any]] = []
    for layer_key, spec in kv_layers.items():
        layer_index = int(layer_key)
        if layer_index < len(key_cache):
            key_channels = {int(channel) for channel in spec.get("key_protected_channels", [])}
            if quantize_cache_tensor_inplace(key_cache[layer_index], key_channels):
                events.append(
                    {
                        "layer_index": layer_index,
                        "tensor": "key_cache",
                        "protected_count": len(key_channels),
                    }
                )
        if layer_index < len(value_cache):
            value_channels = {int(channel) for channel in spec.get("value_protected_channels", [])}
            if value_channels and quantize_cache_tensor_inplace(value_cache[layer_index], value_channels):
                events.append(
                    {
                        "layer_index": layer_index,
                        "tensor": "value_cache",
                        "protected_count": len(value_channels),
                    }
                )
    return {"regime": regime, "cache_quantized_tensor_events": events}


def score_targets_m18(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    prompts: list[dict[str, Any]],
    target_tokens: dict[int, list[int]],
    protected_sets: dict[str, Any],
    regime: str,
    max_new_tokens: int,
    batch_size: int,
    use_float16_autocast: bool,
    run_events_path: Path,
) -> tuple[dict[int, dict[str, float]], dict[str, Any]]:
    import torch

    autocast_enabled = bool(use_float16_autocast and torch.cuda.is_available())
    cache_dtype = torch.float16 if autocast_enabled else None
    normalize_cache_inputs, output_cache = phase4_runner.make_cache_helpers(model, cache_dtype=cache_dtype)
    score_start = checker.SCORING_POSITION - checker.SCORING_WINDOW_TOKENS + 1
    score_end = checker.SCORING_POSITION
    results: dict[int, dict[str, float]] = {}
    previous_fast_path = phase4_runner.set_granite_fast_path_enabled(False) if autocast_enabled else None
    cache_event_count = 0
    cache_layer_events: dict[str, int] = defaultdict(int)

    def is_cuda_oom(exc: BaseException) -> bool:
        return exc.__class__.__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()

    def maybe_apply_cache(cache: Any) -> None:
        nonlocal cache_event_count
        if regime == "static_activation_1pct":
            return
        event = apply_cache_policy(cache, protected_sets, regime)
        cache_event_count += len(event["cache_quantized_tensor_events"])
        for item in event["cache_quantized_tensor_events"]:
            cache_layer_events[f"{item['tensor']}_layer_{item['layer_index']}"] += 1

    def score_batch(batch: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
        batch_indices = [int(item["index"]) for item in batch]
        texts = [shared.make_prompt_text(str(item["prompt"])) for item in batch]
        encoded = tokenizer(texts, padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        target_tensor = torch.tensor(
            [target_tokens[index][:max_new_tokens] for index in batch_indices],
            dtype=torch.long,
            device=device,
        )
        nll = torch.zeros((len(batch),), device=device, dtype=torch.float64)
        scored = 0
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=autocast_enabled):
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
                raise RuntimeError(f"model did not return cache for scoring regime {regime}")
            maybe_apply_cache(past_key_values)
            logits = outputs.logits[:, -1, :]
            for decode_position in range(1, max_new_tokens + 1):
                current_target = target_tensor[:, decode_position - 1]
                if score_start <= decode_position <= score_end:
                    log_probs = torch.log_softmax(logits.float(), dim=-1)
                    nll -= log_probs.gather(1, current_target[:, None]).squeeze(1).to(torch.float64)
                    scored += 1
                if decode_position == max_new_tokens:
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
                    raise RuntimeError(f"model dropped cache while scoring {regime} at {decode_position}")
                maybe_apply_cache(past_key_values)
                logits = outputs.logits[:, -1, :]
        batch_results: dict[int, dict[str, float]] = {}
        for offset, prompt_index in enumerate(batch_indices):
            mean_nll = float((nll[offset] / max(1, scored)).item())
            batch_results[prompt_index] = {
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
                        "event": "completed_m18_scoring_batch",
                        "regime": regime,
                        "batch_indices": batch_indices,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        return batch_results

    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            try:
                results.update(score_batch(batch))
            except BaseException as exc:
                if batch_size > 1 and is_cuda_oom(exc):
                    release_model_memory()
                    for item in batch:
                        results.update(score_batch([item]))
                else:
                    raise
    finally:
        if previous_fast_path is not None:
            phase4_runner.set_granite_fast_path_enabled(previous_fast_path)
    return results, {
        "regime": regime,
        "cache_quantized_tensor_event_count": cache_event_count,
        "cache_quantized_layer_event_counts": dict(sorted(cache_layer_events.items())),
    }


def score_regime(
    *,
    model_provenance: dict[str, Any],
    protected_sets: dict[str, Any],
    regime: str,
    prompts: list[dict[str, Any]],
    target_tokens: dict[int, list[int]],
    batch_size: int,
    dtype_name: str,
    device_name: str,
    run_events_path: Path,
) -> tuple[dict[int, dict[str, float]], dict[str, Any]]:
    model, tokenizer, device = shared.load_model_and_tokenizer(model_provenance, dtype_name=dtype_name, device_name=device_name)
    excluded = phase4_runner.apply_quantization(model, protected_sets, regime)
    cache_summary: dict[str, Any] = {"regime": regime, "cache_quantized_tensor_event_count": 0}
    if regime in {"kivi_key_only", "m18_activation_k", "m18_activation_kv", "random_coupled_activation_k"}:
        scores, cache_summary = score_targets_m18(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompts=prompts,
            target_tokens=target_tokens,
            protected_sets=protected_sets,
            regime=regime,
            max_new_tokens=checker.SCORING_POSITION,
            batch_size=batch_size,
            use_float16_autocast=True,
            run_events_path=run_events_path,
        )
    else:
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
            regime_name=regime,
        )
    del model, tokenizer, device
    release_model_memory()
    excluded["cache_policy"] = cache_summary
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
    activation_path: Path,
    bf16_trace_path: Path,
    hook_coverage: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    included = [row for row in per_trace_rows if not bool(row["no_recoverable_static_gap"])]
    summaries = {
        regime: summarize([float(row["recoveries"][regime]) for row in included])
        for regime in checker.RECOVERY_REGIMES
    }
    no_gap_count = len(per_trace_rows) - len(included)
    for item in summaries.values():
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
        "scoring_position": checker.SCORING_POSITION,
        "scoring_window_tokens": checker.SCORING_WINDOW_TOKENS,
        "metric_name": "positive-static-activation-1pct-gap per-trace recovery",
        "metric_formula": (
            "1 - (perplexity_regime - perplexity_BF16) / "
            "(perplexity_static_activation_1pct - perplexity_BF16)"
        ),
        "primary_regime": "m18_activation_k",
        "primary_result": summaries["m18_activation_k"],
        "results_by_regime": summaries,
        "thresholds": checker.THRESHOLDS,
        "hook_coverage": hook_coverage,
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
        "results_by_regime": summaries,
    }
    controls = {
        "schema_version": f"{SCHEMA_VERSION}_control_metrics",
        "created_at_utc": shared.utc_now(),
        "controls": {
            "static_activation_1pct": {"median_recovery": 0.0},
            "kivi_key_only": summaries["kivi_key_only"],
            "random_coupled_activation_k": summaries["random_coupled_activation_k"],
        },
        "m18_minus_kivi_key_only_median": (
            None
            if summaries["m18_activation_k"]["median_recovery"] is None or summaries["kivi_key_only"]["median_recovery"] is None
            else float(summaries["m18_activation_k"]["median_recovery"]) - float(summaries["kivi_key_only"]["median_recovery"])
        ),
        "m18_minus_random_coupled_median": (
            None
            if summaries["m18_activation_k"]["median_recovery"] is None
            or summaries["random_coupled_activation_k"]["median_recovery"] is None
            else float(summaries["m18_activation_k"]["median_recovery"])
            - float(summaries["random_coupled_activation_k"]["median_recovery"])
        ),
    }
    return metrics, bootstrap, controls


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


def main(argv: list[str] | None = None) -> int:
    shared.SCHEMA_VERSION = SCHEMA_VERSION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"om_phase9_m18_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
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
    args = parser.parse_args(argv)

    if args.model_id != checker.MODEL_ID:
        raise SystemExit(f"M18 primary gate requires {checker.MODEL_ID}")
    if args.prompt_file.resolve() != DEFAULT_PROMPT_FILE.resolve():
        raise SystemExit(f"M18 requires canonical prompt file {DEFAULT_PROMPT_FILE}")
    if shared.file_sha256(args.prompt_file) != checker.EXPECTED_PROMPT_FILE_SHA256:
        raise SystemExit("canonical AIME-2025 indices 0-23 prompt file hash drifted")
    if args.seed != checker.BOOTSTRAP_SEED:
        raise SystemExit("M18 preregisters bootstrap/random seed 20260527")

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

    def m18_excepthook(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        write_failure_packet(run_dir, run_events_path, exc)
        previous_excepthook(exc_type, exc, tb)

    sys.excepthook = m18_excepthook
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

    try:
        prompt_manifest, prompt_reasons = build_prompt_manifest(args.prompt_file)
        environment = shared.build_environment(schema_version=SCHEMA_VERSION)
        model_provenance = m2_runner.resolve_model_snapshot_light(args.model_id)
        model_provenance["schema_version"] = f"{SCHEMA_VERSION}_model_provenance"
        command_metadata = {
            "schema_version": f"{SCHEMA_VERSION}_command",
            "created_at_utc": shared.utc_now(),
            "argv": sys.argv if argv is None else ["run_om_phase9_m18_joint_kv_activation.py", *argv],
            "cwd": str(Path.cwd()),
            "branch": "outlier_migrate_phase9_m18_joint_kv_activation",
            "run_dir": str(run_dir),
            "batch_size": args.batch_size,
            "reuse_activation_run_dir": str(args.reuse_activation_run_dir) if args.reuse_activation_run_dir else None,
            "reuse_trace_run_dir": str(args.reuse_trace_run_dir) if args.reuse_trace_run_dir else None,
            "reuse_score_run_dir": str(args.reuse_score_run_dir) if args.reuse_score_run_dir else None,
        }
        random_seed = {
            "schema_version": f"{SCHEMA_VERSION}_random_seed",
            "seed": args.seed,
            "determinism": {"do_sample": False, "num_beams": 1, "torch_manual_seed": args.seed},
        }
        decoding_config = {
            "schema_version": f"{SCHEMA_VERSION}_decoding_config",
            "max_new_tokens_for_scoring": checker.SCORING_POSITION,
            "max_new_tokens_for_calibration": checker.SCORING_POSITION,
            "do_sample": False,
            "num_beams": 1,
            "eos_policy": "record first EOS but continue to fixed decode positions",
            "scoring_position": checker.SCORING_POSITION,
            "scoring_window_tokens": checker.SCORING_WINDOW_TOKENS,
            "target_trace": "BF16 deterministic greedy trace",
        }
        quantization_config = {
            "schema_version": f"{SCHEMA_VERSION}_quantization_config",
            "weight_bits": 4,
            "scheme": "symmetric_per_output_channel_int4",
            "signed_integer_range": [-7, 7],
            "activation_dtype": "float16",
            "protected_channel_dtype": "bfloat16",
            "kv_cache_quantization": "symmetric_int4_dequantized_per_flat_channel_with_protected_channels_restored",
            "implementation_note": (
                "Non-BF16 regimes use dequantized INT4 weights. M18 cache regimes additionally quantize "
                "key/value cache tensors in-place after each cache update, restoring protected flat channels."
            ),
        }
        shared.write_json(run_dir / "prompt_manifest.json", prompt_manifest)
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
        if model_provenance.get("hf_snapshot_commit") != checker.MODEL_SNAPSHOT:
            shared.write_json(run_dir / "infra_error.json", {"reasons": ["model snapshot commit mismatch"]})
            shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
            checker.evaluate(run_dir)
            return 1

        prompts = prompt_manifest["prompts"]
        prompt_indices = {int(row["index"]) for row in prompts}
        activation_path = run_dir / "activation_magnitudes.jsonl.gz"
        trace_path = run_dir / "bf16_traces.jsonl.gz"
        if args.reuse_activation_run_dir:
            activation_manifest = filter_activation_rows(
                args.reuse_activation_run_dir.resolve() / "activation_magnitudes.jsonl.gz",
                activation_path,
                prompt_indices=prompt_indices,
            )
            shared.write_json(run_dir / "activation_magnitude_manifest.json", activation_manifest)
        else:
            model, tokenizer, device = shared.load_model_and_tokenizer(model_provenance, dtype_name=args.dtype, device_name=args.device)
            activation_manifest = shared.capture_activation_magnitudes(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompts=prompts,
                positions=ACTIVATION_POSITIONS,
                max_new_tokens=checker.SCORING_POSITION,
                batch_size=args.batch_size,
                output_path=activation_path,
                run_events_path=run_events_path,
            )
            shared.write_json(run_dir / "activation_magnitude_manifest.json", activation_manifest)
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

        model, tokenizer, device = shared.load_model_and_tokenizer(model_provenance, dtype_name=args.dtype, device_name=args.device)
        kv_evidence = capture_kv_cache_channel_evidence(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompts=prompts,
            target_tokens=target_tokens,
            max_new_tokens=checker.SCORING_POSITION,
            batch_size=args.batch_size,
            run_events_path=run_events_path,
        )
        shared.write_json(run_dir / "kv_cache_channel_evidence.json", kv_evidence)
        del model, tokenizer, device
        release_model_memory()

        protected_sets, hook_coverage = build_protected_sets(list(shared.iter_activation_rows(activation_path)), kv_evidence)
        shared.write_json(run_dir / "protected_sets.json", protected_sets)
        shared.write_json(run_dir / "hook_coverage.json", hook_coverage)
        if hook_coverage.get("key_cache_attention_layer_coverage", 0.0) <= 0.0:
            raise RuntimeError("M18 found no accessible key-cache channel axes")

        all_scores: dict[str, dict[int, dict[str, float]]] = {}
        excluded_by_regime: dict[str, Any] = {}
        score_reuse_dir = args.reuse_score_run_dir.resolve() if args.reuse_score_run_dir else None
        cached_bf16 = m11_runner.read_score_cache_any(score_reuse_dir, "bf16", expected_prompt_indices=prompt_indices)
        if cached_bf16 is not None:
            all_scores["bf16"] = cached_bf16
            write_score_cache(run_dir, "bf16", all_scores["bf16"])
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

        cached_static = m11_runner.read_score_cache_any(score_reuse_dir, "static_1pct", expected_prompt_indices=prompt_indices)
        if cached_static is not None:
            all_scores["static_activation_1pct"] = cached_static
            write_score_cache(run_dir, "static_activation_1pct", all_scores["static_activation_1pct"])
            excluded_by_regime["static_activation_1pct"] = {
                "regime": "static_activation_1pct",
                "reused_score_cache": str(score_reuse_dir),
                "source_regime": "static_1pct",
            }
        else:
            all_scores["static_activation_1pct"], excluded_by_regime["static_activation_1pct"] = score_regime(
                model_provenance=model_provenance,
                protected_sets=protected_sets,
                regime="static_activation_1pct",
                prompts=prompts,
                target_tokens=target_tokens,
                batch_size=args.batch_size,
                dtype_name=args.dtype,
                device_name=args.device,
                run_events_path=run_events_path,
            )
            write_score_cache(run_dir, "static_activation_1pct", all_scores["static_activation_1pct"])

        for regime in ["kivi_key_only", "m18_activation_k", "m18_activation_kv", "random_coupled_activation_k"]:
            all_scores[regime], excluded_by_regime[regime] = score_regime(
                model_provenance=model_provenance,
                protected_sets=protected_sets,
                regime=regime,
                prompts=prompts,
                target_tokens=target_tokens,
                batch_size=args.batch_size,
                dtype_name=args.dtype,
                device_name=args.device,
                run_events_path=run_events_path,
            )
            write_score_cache(run_dir, regime, all_scores[regime])

        shared.write_json(
            run_dir / "excluded_tensors.json",
            {"schema_version": f"{SCHEMA_VERSION}_excluded_tensors", "created_at_utc": shared.utc_now(), "by_regime": excluded_by_regime},
        )

        per_trace_rows: list[dict[str, Any]] = []
        for prompt in prompts:
            index = int(prompt["index"])
            perplexities = {regime: float(all_scores[regime][index]["perplexity"]) for regime in checker.REGIMES}
            mean_nll = {regime: float(all_scores[regime][index]["mean_nll"]) for regime in checker.REGIMES}
            static_gap = perplexities["static_activation_1pct"] - perplexities["bf16"]
            no_gap = static_gap <= 0.0
            recoveries: dict[str, float | None] = {}
            for regime in checker.RECOVERY_REGIMES:
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
            hook_coverage=hook_coverage,
        )
        shared.write_json(run_dir / "metrics.json", metrics)
        shared.write_json(run_dir / "bootstrap_ci.json", bootstrap)
        shared.write_json(run_dir / "control_metrics.json", controls)
        run_events_path.open("a", encoding="utf-8").write(
            json.dumps({"created_at_utc": shared.utc_now(), "event": "run_completed"}, sort_keys=True) + "\n"
        )
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
    except BaseException as exc:
        write_failure_packet(run_dir, run_events_path, exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
