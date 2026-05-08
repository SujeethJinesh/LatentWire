#!/usr/bin/env python3
"""Run shared Phase 0 characterization branches."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTLIER_RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase0/results"
RM_RESULTS_DIR = ROOT / "experimental/residual_migration/phase0/results"
DEFAULT_PROMPT_FILE = ROOT / "experimental/shared/prompts/aime_2025_indices_0_11.jsonl"
DEFAULT_MODEL_ID = "ibm-granite/granite-4.0-h-tiny"
DEFAULT_POSITIONS = (100, 500, 1000, 5000, 10000)
DEFAULT_SEED = 20260508
SCHEMA_VERSION = "om_phase0_v1"
RM_SCHEMA_VERSION = "rm_phase0_v1"
RM_DEFAULT_MAX_NEW_TOKENS = 2048
RM_CLIP_QUANTILE = 0.95
EXPECTED_PROMPT_SOURCE_DATASET = "opencompass/AIME2025"
EXPECTED_PROMPT_SOURCE_FILE = "aime2025-I.jsonl"
EXPECTED_PROMPT_SOURCE_COMMIT = "a6ad95f611d72cf628a80b58bd0432ef6638f958"
THRESHOLDS = {
    "migration_fraction_threshold": 0.05,
    "rank_delta_strictly_greater_than": 2,
    "top_channel_fraction": 0.01,
    "bootstrap_samples": 1000,
    "bootstrap_ci": 0.95,
}
RM_THRESHOLDS = {
    "replicates_ci_upper_lt": 0.015,
    "hybrids_depend_ci_lower_gt": 0.03,
    "bootstrap_samples": 1000,
    "bootstrap_ci": 0.95,
}


class Tee:
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def command_output(command: list[str], *, timeout: int = 60) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=ROOT,
        )
    except Exception as exc:  # pragma: no cover - environment-specific.
        return {"command": command, "returncode": None, "stdout": "", "stderr": repr(exc)}
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def git_sha() -> str | None:
    result = command_output(["git", "rev-parse", "HEAD"], timeout=30)
    if result["returncode"] == 0:
        return str(result["stdout"]).strip()
    return None


def parse_positions(text: str) -> tuple[int, ...]:
    positions: list[int] = []
    for part in text.split(","):
        value = int(part.strip())
        if value <= 0:
            raise argparse.ArgumentTypeError("decode positions must be positive integers")
        positions.append(value)
    if tuple(positions) != DEFAULT_POSITIONS:
        raise argparse.ArgumentTypeError(
            f"OutlierMigrate Phase 0 requires positions {DEFAULT_POSITIONS}; got {tuple(positions)}"
        )
    return tuple(positions)


def parse_prompt_file(prompt_file: Path) -> tuple[list[dict[str, Any]], list[str]]:
    reasons: list[str] = []
    prompts: list[dict[str, Any]] = []
    if not prompt_file.is_file():
        return [], [f"prompt file is missing: {prompt_file}"]
    try:
        lines = [line for line in prompt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row_index, line in enumerate(lines):
            item = json.loads(line)
            if not isinstance(item, dict):
                reasons.append(f"prompt row {row_index} is not a JSON object")
                continue
            prompt = item.get("prompt") or item.get("problem") or item.get("question")
            if not isinstance(prompt, str) or not prompt.strip():
                reasons.append(f"prompt row {row_index} has no prompt/problem/question text")
                continue
            prompts.append(
                {
                    "index": int(item.get("index", row_index)),
                    "prompt_id": str(item.get("prompt_id", item.get("id", row_index))),
                    "prompt": prompt,
                    "answer": item.get("answer"),
                    "source_dataset": item.get("source_dataset"),
                    "source_file": item.get("source_file"),
                    "source_commit": item.get("source_commit"),
                }
            )
            if item.get("source_dataset") != EXPECTED_PROMPT_SOURCE_DATASET:
                reasons.append(
                    f"prompt row {row_index} source_dataset is not {EXPECTED_PROMPT_SOURCE_DATASET}"
                )
            if item.get("source_file") != EXPECTED_PROMPT_SOURCE_FILE:
                reasons.append(f"prompt row {row_index} source_file is not {EXPECTED_PROMPT_SOURCE_FILE}")
            if item.get("source_commit") != EXPECTED_PROMPT_SOURCE_COMMIT:
                reasons.append(
                    f"prompt row {row_index} source_commit is not {EXPECTED_PROMPT_SOURCE_COMMIT}"
                )
    except Exception as exc:
        return [], [f"cannot parse prompt file: {exc!r}"]
    if [row["index"] for row in prompts] != list(range(12)):
        reasons.append("prompt indices are not exactly deterministic indices 0-11")
    if len(prompts) != 12:
        reasons.append(f"prompt count {len(prompts)} is not the preregistered count 12")
    return prompts, reasons


def prompt_payload_sha256(prompts: list[dict[str, Any]]) -> str | None:
    if not prompts:
        return None
    ordered = sorted(prompts, key=lambda row: int(row["index"]))
    payload = "".join(str(row["prompt"]) for row in ordered).encode("utf-8")
    return bytes_sha256(payload)


def build_prompt_manifest(prompt_file: Path, *, schema_version: str = SCHEMA_VERSION) -> tuple[dict[str, Any], list[str]]:
    prompts, reasons = parse_prompt_file(prompt_file)
    return (
        {
            "schema_version": f"{schema_version}_prompt_manifest",
            "created_at_utc": utc_now(),
            "source": "AIME-2025",
            "selection": "deterministic_indices_0_11",
            "prompt_file": str(prompt_file),
            "prompt_file_sha256": file_sha256(prompt_file) if prompt_file.is_file() else None,
            "prompt_count": len(prompts),
            "prompt_sha256": prompt_payload_sha256(prompts),
            "prompt_sha256_semantics": "sha256 of concatenated prompt text in deterministic index order",
            "prompts": prompts,
        },
        reasons,
    )


def package_version(name: str) -> dict[str, Any]:
    try:
        module = __import__(name)
    except Exception as exc:
        return {"available": False, "import_error": repr(exc)}
    return {"available": True, "version": getattr(module, "__version__", None)}


def build_environment(*, schema_version: str = SCHEMA_VERSION) -> dict[str, Any]:
    torch_info: dict[str, Any]
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": torch.cuda.get_device_capability(index),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
        }
    except Exception as exc:  # pragma: no cover - environment-specific.
        torch_info = {"import_error": repr(exc)}
    return {
        "schema_version": f"{schema_version}_environment",
        "created_at_utc": utc_now(),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "repo": {"root": str(ROOT), "git_sha": git_sha()},
        "torch": torch_info,
        "packages": {
            name: package_version(name)
            for name in ["torch", "transformers", "accelerate", "huggingface_hub", "numpy"]
        },
        "commands": {
            "nvidia_smi": command_output(["nvidia-smi"], timeout=30),
            "pip_freeze": command_output([sys.executable, "-m", "pip", "freeze"], timeout=120),
        },
        "environment_variables": {
            key: os.environ.get(key)
            for key in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "CUDA_VISIBLE_DEVICES"]
        },
    }


def hf_cache_candidates(model_id: str) -> list[Path]:
    safe_id = "models--" + model_id.replace("/", "--")
    bases = [
        os.environ.get("HF_HUB_CACHE"),
        os.environ.get("TRANSFORMERS_CACHE"),
        str(Path(os.environ.get("HF_HOME", "")) / "hub") if os.environ.get("HF_HOME") else None,
        "/workspace/hf_cache/hub",
        "/workspace/hf_cache",
        str(Path.home() / ".cache/huggingface/hub"),
    ]
    candidates: list[Path] = []
    for base in bases:
        if base:
            candidates.append(Path(base) / safe_id)
    return candidates


def resolve_model_snapshot(model_id: str, *, schema_version: str = SCHEMA_VERSION) -> dict[str, Any]:
    checked: list[str] = []
    for repo_dir in hf_cache_candidates(model_id):
        checked.append(str(repo_dir))
        refs_main = repo_dir / "refs/main"
        snapshots_dir = repo_dir / "snapshots"
        commits: list[str] = []
        if refs_main.is_file():
            commits.append(refs_main.read_text(encoding="utf-8").strip())
        if snapshots_dir.is_dir():
            commits.extend(path.name for path in snapshots_dir.iterdir() if path.is_dir())
        for commit in dict.fromkeys(commits):
            snapshot = snapshots_dir / commit
            if (snapshot / "config.json").exists():
                files = []
                for path in sorted(snapshot.iterdir()):
                    if path.is_file() or path.is_symlink():
                        resolved = path.resolve()
                        files.append(
                            {
                                "path": path.name,
                                "resolved_path": str(resolved),
                                "bytes": resolved.stat().st_size if resolved.exists() else None,
                                "sha256": file_sha256(resolved) if resolved.is_file() else None,
                            }
                        )
                return {
                    "schema_version": f"{schema_version}_model_provenance",
                    "created_at_utc": utc_now(),
                    "model_id": model_id,
                    "local_files_only": True,
                    "hf_snapshot_commit": commit,
                    "snapshot_path": str(snapshot),
                    "cache_repo_path": str(repo_dir),
                    "files": files,
                    "checked_cache_paths": checked,
                }
    return {
        "schema_version": f"{schema_version}_model_provenance",
        "created_at_utc": utc_now(),
        "model_id": model_id,
        "local_files_only": True,
        "hf_snapshot_commit": None,
        "snapshot_path": None,
        "checked_cache_paths": checked,
        "error": "no local HuggingFace snapshot with config.json found",
    }


def discover_transformer_layers(model: Any) -> tuple[list[tuple[str, Any]], str]:
    import torch

    candidates = [
        ("model.layers", getattr(getattr(model, "model", None), "layers", None)),
        ("transformer.h", getattr(getattr(model, "transformer", None), "h", None)),
        ("gpt_neox.layers", getattr(getattr(model, "gpt_neox", None), "layers", None)),
        ("backbone.layers", getattr(getattr(model, "backbone", None), "layers", None)),
        ("decoder.layers", getattr(getattr(model, "decoder", None), "layers", None)),
        ("layers", getattr(model, "layers", None)),
    ]
    for origin, maybe_layers in candidates:
        if isinstance(maybe_layers, torch.nn.ModuleList) and len(maybe_layers) > 0:
            return [(f"{origin}.{index}", layer) for index, layer in enumerate(maybe_layers)], origin
        if isinstance(maybe_layers, (list, tuple)) and maybe_layers and all(
            isinstance(item, torch.nn.Module) for item in maybe_layers
        ):
            return [(f"{origin}.{index}", layer) for index, layer in enumerate(maybe_layers)], origin
    raise RuntimeError("could not locate transformer layers for activation hooks")


def tensor_from_hook_output(output: Any) -> Any:
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, list):
        output = output[0]
    return output


def make_prompt_text(prompt: str) -> str:
    return (
        "Solve the following AIME problem. Think step by step and give the final "
        f"answer as a three-digit integer.\n\nProblem: {prompt}\n\nSolution:"
    )


def load_model_and_tokenizer(model_provenance: dict[str, Any], *, dtype_name: str, device_name: str) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snapshot_path = model_provenance.get("snapshot_path")
    if not snapshot_path:
        raise RuntimeError("cannot load model without a resolved local snapshot_path")
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype_by_name = {
        "auto": None,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_by_name[dtype_name]
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_name == "auto"
        else device_name
    )
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(snapshot_path, **kwargs)
    model.eval()
    model.to(device)
    return model, tokenizer, device


def capture_activation_magnitudes(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    prompts: list[dict[str, Any]],
    positions: tuple[int, ...],
    max_new_tokens: int,
    batch_size: int,
    output_path: Path,
    run_events_path: Path,
) -> dict[str, Any]:
    import torch

    layers, layer_origin = discover_transformer_layers(model)
    state: dict[str, Any] = {
        "capture_enabled": False,
        "decode_position": None,
        "batch_prompt_indices": [],
        "records_by_layer": {},
    }
    handles = []

    def make_hook(layer_index: int, layer_name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if not state["capture_enabled"]:
                return
            tensor = tensor_from_hook_output(output)
            if not torch.is_tensor(tensor) or tensor.ndim < 2:
                return
            last_hidden = tensor[:, -1, :].detach().abs().to(torch.float32).cpu()
            state["records_by_layer"][layer_index] = (layer_name, last_hidden)

        return hook

    for layer_index, (layer_name, layer) in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(layer_index, layer_name)))

    total_rows = 0
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    prompt_events: list[dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with gzip.open(output_path, "wt", encoding="utf-8") as handle, torch.inference_mode():
            for start in range(0, len(prompts), batch_size):
                batch = prompts[start : start + batch_size]
                batch_indices = [int(item["index"]) for item in batch]
                texts = [make_prompt_text(str(item["prompt"])) for item in batch]
                encoded = tokenizer(texts, padding=True, return_tensors="pt")
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                state["capture_enabled"] = False
                cache_position = torch.arange(input_ids.shape[1], device=device)
                model_inputs = model.prepare_inputs_for_generation(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    use_cache=True,
                )
                outputs = model(**model_inputs)
                past_key_values = getattr(outputs, "past_key_values", None)
                if past_key_values is None:
                    raise RuntimeError(
                        "model did not return past_key_values after prepare_inputs_for_generation; "
                        "GraniteMoeHybrid requires its HybridMambaAttentionDynamicCache path"
                    )
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                eos_first_seen: dict[int, int | None] = {index: None for index in batch_indices}
                for decode_position in range(1, max_new_tokens + 1):
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)],
                        dim=1,
                    )
                    state["decode_position"] = decode_position
                    state["batch_prompt_indices"] = batch_indices
                    state["records_by_layer"] = {}
                    state["capture_enabled"] = decode_position in positions
                    cache_position = torch.tensor([attention_mask.shape[1] - 1], device=device, dtype=torch.long)
                    model_inputs = model.prepare_inputs_for_generation(
                        input_ids=next_token[:, None],
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        use_cache=True,
                    )
                    outputs = model(**model_inputs)
                    past_key_values = getattr(outputs, "past_key_values", None)
                    if past_key_values is None:
                        raise RuntimeError(
                            f"model dropped past_key_values at decode position {decode_position}"
                        )
                    if state["capture_enabled"]:
                        missing_layers = sorted(
                            set(range(len(layers))).difference(state["records_by_layer"].keys())
                        )
                        if missing_layers:
                            raise RuntimeError(
                                f"missing hook captures at decode position {decode_position}: {missing_layers[:8]}"
                            )
                        for layer_index in range(len(layers)):
                            layer_name, magnitudes = state["records_by_layer"][layer_index]
                            for batch_offset, prompt_index in enumerate(batch_indices):
                                row = {
                                    "schema_version": f"{SCHEMA_VERSION}_activation_row",
                                    "prompt_index": prompt_index,
                                    "prompt_id": batch[batch_offset]["prompt_id"],
                                    "layer_index": layer_index,
                                    "layer_name": layer_name,
                                    "decode_position": decode_position,
                                    "channel_count": int(magnitudes.shape[-1]),
                                    "channel_magnitudes": [
                                        float(value) for value in magnitudes[batch_offset].tolist()
                                    ],
                                }
                                handle.write(json.dumps(row, sort_keys=True) + "\n")
                                total_rows += 1
                    if eos_token_id is not None:
                        for batch_offset, token in enumerate(next_token.tolist()):
                            prompt_index = batch_indices[batch_offset]
                            if int(token) == int(eos_token_id) and eos_first_seen[prompt_index] is None:
                                eos_first_seen[prompt_index] = decode_position
                    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                prompt_events.extend(
                    {
                        "prompt_index": index,
                        "prompt_id": batch[offset]["prompt_id"],
                        "input_token_count": int(encoded["attention_mask"][offset].sum().item()),
                        "first_eos_decode_position": eos_first_seen[index],
                    }
                    for offset, index in enumerate(batch_indices)
                )
                with run_events_path.open("a", encoding="utf-8") as event_handle:
                    event_handle.write(
                        json.dumps(
                            {
                                "created_at_utc": utc_now(),
                                "event": "completed_prompt_batch",
                                "prompt_indices": batch_indices,
                                "max_new_tokens": max_new_tokens,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
    finally:
        for handle in handles:
            handle.remove()
    return {
        "schema_version": f"{SCHEMA_VERSION}_activation_manifest",
        "created_at_utc": utc_now(),
        "artifact": "activation_magnitudes.jsonl.gz",
        "artifact_sha256": file_sha256(output_path),
        "trace_count": len(prompts),
        "positions": list(positions),
        "layer_count": len(layers),
        "layer_origin": layer_origin,
        "layer_names": [name for name, _layer in layers],
        "row_count": total_rows,
        "prompt_events": prompt_events,
        "capture_semantics": {
            "module": "transformer_layer_forward_output",
            "token": "last generated token in manual greedy decode step",
            "value": "absolute activation magnitude per output channel",
            "decode_position_basis": "generated-token count after prompt; position 100 is the 100th generated token",
            "eos_policy": "continue decoding until max_new_tokens so every preregistered position is observable",
        },
    }


def iter_activation_rows(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def ranks_desc(values: list[float]) -> list[int]:
    ordered = sorted(range(len(values)), key=lambda index: (-float(values[index]), index))
    ranks = [0] * len(values)
    for rank, channel in enumerate(ordered):
        ranks[channel] = rank
    return ranks


def select_top_channels_by_layer(
    by_trace_layer: dict[int, dict[int, dict[int, list[float]]]], *, positions: tuple[int, ...]
) -> dict[int, list[int]]:
    top_fraction = float(THRESHOLDS["top_channel_fraction"])
    layer_indices = sorted({layer for trace in by_trace_layer.values() for layer in trace})
    selected: dict[int, list[int]] = {}
    for layer_index in layer_indices:
        base_vectors = [
            by_trace_layer[prompt_index][layer_index][positions[0]]
            for prompt_index in sorted(by_trace_layer)
            if layer_index in by_trace_layer[prompt_index]
        ]
        if not base_vectors:
            continue
        channel_count = len(base_vectors[0])
        top_k = max(1, math.ceil(channel_count * top_fraction))
        mean_magnitudes = [
            mean(float(vector[channel]) for vector in base_vectors) for channel in range(channel_count)
        ]
        selected[layer_index] = sorted(
            range(channel_count), key=lambda channel: (-mean_magnitudes[channel], channel)
        )[:top_k]
    return selected


def compute_outlier_migration_metrics(
    rows: list[dict[str, Any]], *, positions: tuple[int, ...], bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    by_trace_layer: dict[int, dict[int, dict[int, list[float]]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        by_trace_layer[int(row["prompt_index"])][int(row["layer_index"])][int(row["decode_position"])] = [
            float(value) for value in row["channel_magnitudes"]
        ]
    trace_metrics: list[dict[str, Any]] = []
    layer_metrics: dict[int, list[float]] = defaultdict(list)
    rank_delta = int(THRESHOLDS["rank_delta_strictly_greater_than"])
    top_channels_by_layer = select_top_channels_by_layer(by_trace_layer, positions=positions)
    for prompt_index in sorted(by_trace_layer):
        layer_fractions: list[float] = []
        for layer_index in sorted(by_trace_layer[prompt_index]):
            position_vectors = by_trace_layer[prompt_index][layer_index]
            base = position_vectors[positions[0]]
            final = position_vectors[positions[-1]]
            base_ranks = ranks_desc(base)
            final_ranks = ranks_desc(final)
            base_top_channels = top_channels_by_layer[layer_index]
            migrated = sum(
                1 for channel in base_top_channels if abs(final_ranks[channel] - base_ranks[channel]) > rank_delta
            )
            fraction = migrated / len(base_top_channels)
            layer_fractions.append(fraction)
            layer_metrics[layer_index].append(fraction)
        trace_metrics.append(
            {
                "prompt_index": prompt_index,
                "migration_fraction": float(mean(layer_fractions)),
                "layer_count": len(layer_fractions),
            }
        )
    trace_values = [float(row["migration_fraction"]) for row in trace_metrics]
    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(bootstrap_samples):
        sample = [trace_values[rng.randrange(len(trace_values))] for _ in trace_values]
        boot.append(float(mean(sample)))
    boot.sort()
    ci_low = boot[int(0.025 * (len(boot) - 1))]
    ci_high = boot[int(0.975 * (len(boot) - 1))]
    return {
        "schema_version": f"{SCHEMA_VERSION}_metrics",
        "metric_name": "outlier_rank_migration_fraction",
        "migration_fraction": float(mean(trace_values)),
        "bootstrap_ci95": {"ci95_low": ci_low, "ci95_high": ci_high},
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "trace_metrics": trace_metrics,
        "layer_metrics": [
            {
                "layer_index": layer_index,
                "migration_fraction_mean": float(mean(values)),
                "trace_count": len(values),
            }
            for layer_index, values in sorted(layer_metrics.items())
        ],
        "top_channels_by_layer": {
            str(layer_index): channels for layer_index, channels in sorted(top_channels_by_layer.items())
        },
        "top_channel_selection": "top 1% channels per layer by mean magnitude at decode position 100 across all 12 traces",
        "aggregation": {
            "trace_level": "mean over layers per trace",
            "gate_level": "mean over 12 trace-level migration fractions",
            "bootstrap_unit": "trace",
        },
        "thresholds": THRESHOLDS,
    }


def normalize_aime_answer(answer: Any) -> str:
    text = str(answer).strip()
    if re.fullmatch(r"\d+", text):
        return str(int(text))
    return text


def extract_aime_answer(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\{?\s*([0-9]{1,3})\s*\}?", text)
    if boxed:
        return normalize_aime_answer(boxed[-1])
    final_patterns = [
        r"(?:final answer|answer is|answer:)\s*(?:is\s*)?(?:\$?\\?boxed\{?\s*)?([0-9]{1,3})",
        r"(?:therefore|thus|so),?\s*the answer is\s*(?:\$?\\?boxed\{?\s*)?([0-9]{1,3})",
    ]
    for pattern in final_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return normalize_aime_answer(matches[-1])
    return None


def score_generation(generated_text: str, canonical_answer: Any) -> dict[str, Any]:
    expected = normalize_aime_answer(canonical_answer)
    extracted = extract_aime_answer(generated_text)
    return {
        "canonical_answer": expected,
        "extracted_answer": extracted,
        "correct": extracted == expected,
        "scoring_rule": (
            "exact match between canonical AIME integer answer and an explicitly boxed/final "
            "1-3 digit generated answer, normalized by removing leading zeros; rows with no "
            "explicit final answer are scored incorrect"
        ),
    }


def build_residual_clip_hooks(layers: list[tuple[str, Any]]) -> tuple[list[Any], dict[str, Any]]:
    import torch

    stats: dict[str, Any] = {
        "clip_quantile": RM_CLIP_QUANTILE,
        "threshold_scope": "per layer, per batch element, per forwarded token position, over hidden channels",
        "hook_type": "forward_pre_hook",
        "hook_input_token_position": (
            "local position in the hidden-state tensor presented to the layer; with cache-based "
            "generation, decode forwards usually contain a single local token position"
        ),
        "layers": {},
    }

    def make_hook(layer_index: int, layer_name: str):
        layer_key = str(layer_index)
        stats["layers"][layer_key] = {
            "layer_index": layer_index,
            "layer_name": layer_name,
            "invocations": 0,
            "total_values": 0,
            "clipped_values": 0,
            "clip_fraction": 0.0,
            "max_abs_observed": 0.0,
            "max_threshold_observed": 0.0,
            "token_positions": {},
        }

        def clip_hidden(hidden_states: Any) -> Any:
            if not torch.is_tensor(hidden_states) or hidden_states.ndim < 2:
                return hidden_states
            abs_values = hidden_states.detach().abs().to(torch.float32)
            threshold = torch.quantile(abs_values, RM_CLIP_QUANTILE, dim=-1, keepdim=True)
            mask = abs_values > threshold
            clipped_count = int(mask.sum().item())
            total_count = int(mask.numel())
            if clipped_count:
                clipped = torch.where(
                    mask,
                    torch.sign(hidden_states) * threshold.to(device=hidden_states.device, dtype=hidden_states.dtype),
                    hidden_states,
                )
            else:
                clipped = hidden_states
            layer_stats = stats["layers"][layer_key]
            layer_stats["invocations"] += 1
            layer_stats["total_values"] += total_count
            layer_stats["clipped_values"] += clipped_count
            layer_stats["clip_fraction"] = layer_stats["clipped_values"] / layer_stats["total_values"]
            layer_stats["max_abs_observed"] = max(
                float(layer_stats["max_abs_observed"]), float(abs_values.max().item())
            )
            layer_stats["max_threshold_observed"] = max(
                float(layer_stats["max_threshold_observed"]), float(threshold.max().item())
            )
            position_counts = mask.reshape(-1, mask.shape[-2], mask.shape[-1]).sum(dim=(0, 2)).tolist()
            position_totals = [mask.reshape(-1, mask.shape[-2], mask.shape[-1]).shape[0] * mask.shape[-1]] * len(
                position_counts
            )
            for position, (pos_clipped, pos_total) in enumerate(zip(position_counts, position_totals)):
                position_key = str(position)
                item = layer_stats["token_positions"].setdefault(
                    position_key, {"total_values": 0, "clipped_values": 0, "clip_fraction": 0.0}
                )
                item["total_values"] += int(pos_total)
                item["clipped_values"] += int(pos_clipped)
                item["clip_fraction"] = item["clipped_values"] / item["total_values"]
            return clipped

        def hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
            if args and torch.is_tensor(args[0]):
                updated_args = (clip_hidden(args[0]), *args[1:])
                return updated_args, kwargs
            if torch.is_tensor(kwargs.get("hidden_states")):
                updated_kwargs = dict(kwargs)
                updated_kwargs["hidden_states"] = clip_hidden(updated_kwargs["hidden_states"])
                return args, updated_kwargs
            return None

        return hook

    handles = [
        layer.register_forward_pre_hook(make_hook(layer_index, layer_name), with_kwargs=True)
        for layer_index, (layer_name, layer) in enumerate(layers)
    ]
    return handles, stats


def run_greedy_generations(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    prompts: list[dict[str, Any]],
    max_new_tokens: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    import torch

    generations: list[dict[str, Any]] = []
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            texts = [make_prompt_text(str(item["prompt"])) for item in batch]
            encoded = tokenizer(texts, padding=True, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            prompt_width = input_ids.shape[1]
            for batch_offset, item in enumerate(batch):
                generated_ids = outputs[batch_offset, prompt_width:]
                generated_text = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)
                score = score_generation(generated_text, item["answer"])
                generations.append(
                    {
                        "schema_version": f"{RM_SCHEMA_VERSION}_generation_row",
                        "prompt_index": int(item["index"]),
                        "prompt_id": str(item["prompt_id"]),
                        "input_token_count": int(encoded["attention_mask"][batch_offset].sum().item()),
                        "max_new_tokens": max_new_tokens,
                        "generated_token_count": int(generated_ids.numel()),
                        "generated_text": generated_text,
                        **score,
                    }
                )
    return sorted(generations, key=lambda row: int(row["prompt_index"]))


def compute_residual_migration_metrics(
    baseline_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    baseline_by_prompt = {int(row["prompt_index"]): row for row in baseline_rows}
    ablation_by_prompt = {int(row["prompt_index"]): row for row in ablation_rows}
    prompt_indices = sorted(baseline_by_prompt)
    per_prompt: list[dict[str, Any]] = []
    for prompt_index in prompt_indices:
        base = baseline_by_prompt[prompt_index]
        ablated = ablation_by_prompt[prompt_index]
        baseline_correct = 1.0 if bool(base["correct"]) else 0.0
        ablation_correct = 1.0 if bool(ablated["correct"]) else 0.0
        per_prompt.append(
            {
                "prompt_index": prompt_index,
                "prompt_id": base["prompt_id"],
                "canonical_answer": base["canonical_answer"],
                "baseline_extracted_answer": base["extracted_answer"],
                "ablation_extracted_answer": ablated["extracted_answer"],
                "baseline_correct": bool(base["correct"]),
                "ablation_correct": bool(ablated["correct"]),
                "drop": baseline_correct - ablation_correct,
            }
        )
    drops = [float(row["drop"]) for row in per_prompt]
    baseline_values = [1.0 if bool(row["correct"]) else 0.0 for row in baseline_rows]
    ablation_values = [1.0 if bool(row["correct"]) else 0.0 for row in ablation_rows]
    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(bootstrap_samples):
        sample = [drops[rng.randrange(len(drops))] for _ in drops]
        boot.append(float(mean(sample)))
    boot.sort()
    ci_low = boot[int(0.025 * (len(boot) - 1))]
    ci_high = boot[int(0.975 * (len(boot) - 1))]
    baseline_accuracy = float(mean(baseline_values))
    ablation_accuracy = float(mean(ablation_values))
    return {
        "schema_version": f"{RM_SCHEMA_VERSION}_metrics",
        "metric_name": "aime_accuracy_drop_after_residual_95p_clip",
        "baseline_accuracy": baseline_accuracy,
        "ablation_accuracy": ablation_accuracy,
        "accuracy_drop": baseline_accuracy - ablation_accuracy,
        "bootstrap_ci95": {"ci95_low": ci_low, "ci95_high": ci_high},
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "per_prompt": per_prompt,
        "aggregation": {
            "prompt_level": "binary exact-match correctness",
            "gate_level": "mean baseline correctness minus mean ablation correctness across 12 prompts",
            "bootstrap_unit": "prompt",
        },
        "thresholds": RM_THRESHOLDS,
    }


def write_generations(path: Path, *, baseline_rows: list[dict[str, Any]], ablation_rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for phase, rows in [("baseline", baseline_rows), ("ablation", ablation_rows)]:
            for row in rows:
                handle.write(json.dumps({"phase": phase, **row}, sort_keys=True) + "\n")


def run_residual_migration(args: argparse.Namespace, argv: list[str] | None) -> int:
    if args.model_id != DEFAULT_MODEL_ID:
        raise SystemExit(f"Residual Migration Phase 0 requires {DEFAULT_MODEL_ID}; got {args.model_id}")
    if args.prompt_file.resolve() != DEFAULT_PROMPT_FILE.resolve():
        raise SystemExit(f"Residual Migration Phase 0 requires canonical prompt file {DEFAULT_PROMPT_FILE}")
    if args.max_new_tokens != RM_DEFAULT_MAX_NEW_TOKENS:
        raise SystemExit(f"Residual Migration Phase 0 freezes --max-new-tokens at {RM_DEFAULT_MAX_NEW_TOKENS}")
    if args.bootstrap_samples != RM_THRESHOLDS["bootstrap_samples"]:
        raise SystemExit("Residual Migration Phase 0 preregisters bootstrap n=1000")

    run_dir = args.results_dir / args.run_id
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")
    (run_dir / "logs").mkdir(parents=True)
    stdout_log = (run_dir / "logs/stdout.log").open("w", encoding="utf-8", buffering=1)
    stderr_log = (run_dir / "logs/stderr.log").open("w", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, stdout_log)
    sys.stderr = Tee(sys.__stderr__, stderr_log)
    run_events_path = run_dir / "run_events.jsonl"
    run_events_path.write_text(
        json.dumps({"created_at_utc": utc_now(), "event": "run_started"}, sort_keys=True) + "\n",
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

    prompt_manifest, prompt_reasons = build_prompt_manifest(args.prompt_file, schema_version=RM_SCHEMA_VERSION)
    environment = build_environment(schema_version=RM_SCHEMA_VERSION)
    model_provenance = resolve_model_snapshot(args.model_id, schema_version=RM_SCHEMA_VERSION)
    command_metadata = {
        "schema_version": f"{RM_SCHEMA_VERSION}_command",
        "created_at_utc": utc_now(),
        "argv": sys.argv if argv is None else ["run_phase0_branch.py", *argv],
        "cwd": str(Path.cwd()),
        "branch": args.branch,
        "run_dir": str(run_dir),
        "frozen_generation_limit": {
            "max_new_tokens": RM_DEFAULT_MAX_NEW_TOKENS,
            "set_before_analysis": True,
            "reason": "pre-analysis Phase 0 cap for deterministic greedy AIME-2025 scoring",
        },
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "local_files_only": True,
            "prompt_template": "shared Phase 0 AIME solve/final-answer template",
        },
    }
    random_seed = {
        "schema_version": f"{RM_SCHEMA_VERSION}_random_seed",
        "seed": args.seed,
        "determinism": {"do_sample": False, "num_beams": 1, "torch_manual_seed": args.seed},
    }
    ablation_config = {
        "schema_version": f"{RM_SCHEMA_VERSION}_ablation_config",
        "created_at_utc": utc_now(),
        "ablation": "residual_stream_hidden_state_95p_clip",
        "clip_quantile": RM_CLIP_QUANTILE,
        "clip_rule": (
            "in every transformer-layer forward pre-hook, values with absolute magnitude above the "
            "per-layer/per-token-position 95th percentile over hidden channels are clipped to that "
            "threshold while preserving sign"
        ),
        "threshold_scope": "per layer, per batch element, per forwarded token position, over hidden channels",
    }
    write_json(run_dir / "prompt_manifest.json", prompt_manifest)
    write_json(run_dir / "environment.json", environment)
    write_json(run_dir / "model_provenance.json", model_provenance)
    write_json(run_dir / "command_metadata.json", command_metadata)
    write_json(run_dir / "random_seed.json", random_seed)
    write_json(run_dir / "ablation_config.json", ablation_config)
    if prompt_reasons:
        write_json(run_dir / "infra_error.json", {"reasons": prompt_reasons})
        print(f"prompt manifest failed preregistered invariants: {prompt_reasons}", file=sys.stderr)
        write_json(run_dir / "artifact_hashes.json", build_artifact_hashes(run_dir, schema_version=RM_SCHEMA_VERSION))
        return 1
    if not model_provenance.get("snapshot_path"):
        write_json(run_dir / "infra_error.json", {"reasons": [model_provenance.get("error")]})
        print("model snapshot could not be resolved locally; see infra_error.json", file=sys.stderr)
        write_json(run_dir / "artifact_hashes.json", build_artifact_hashes(run_dir, schema_version=RM_SCHEMA_VERSION))
        return 1

    prompts = prompt_manifest["prompts"]
    model, tokenizer, device = load_model_and_tokenizer(
        model_provenance, dtype_name=args.dtype, device_name=args.device
    )
    layers, layer_origin = discover_transformer_layers(model)
    command_metadata["layer_origin"] = layer_origin
    command_metadata["layer_count"] = len(layers)
    write_json(run_dir / "command_metadata.json", command_metadata)
    baseline_rows = run_greedy_generations(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    with run_events_path.open("a", encoding="utf-8") as event_handle:
        event_handle.write(
            json.dumps({"created_at_utc": utc_now(), "event": "baseline_completed"}, sort_keys=True) + "\n"
        )
    handles, clip_stats = build_residual_clip_hooks(layers)
    try:
        ablation_rows = run_greedy_generations(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
    finally:
        for handle in handles:
            handle.remove()
    ablation_config["layer_origin"] = layer_origin
    ablation_config["layer_count"] = len(layers)
    ablation_config["clip_stats"] = clip_stats
    write_json(run_dir / "ablation_config.json", ablation_config)
    write_generations(run_dir / "generations.jsonl", baseline_rows=baseline_rows, ablation_rows=ablation_rows)
    metrics = compute_residual_migration_metrics(
        baseline_rows,
        ablation_rows,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    metrics.update(
        {
            "created_at_utc": utc_now(),
            "branch": args.branch,
            "model_id": args.model_id,
            "model_snapshot_commit": model_provenance.get("hf_snapshot_commit"),
            "prompt_source": "AIME-2025",
            "prompt_selection": "deterministic_indices_0_11",
            "prompt_sha256": prompt_manifest["prompt_sha256"],
            "trace_count": len(prompts),
            "generation_artifact": "generations.jsonl",
            "generation_artifact_sha256": file_sha256(run_dir / "generations.jsonl"),
        }
    )
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "bootstrap_ci.json",
        {
            "schema_version": f"{RM_SCHEMA_VERSION}_bootstrap_ci",
            "metric_name": metrics["metric_name"],
            "bootstrap_samples": metrics["bootstrap_samples"],
            "bootstrap_seed": metrics["bootstrap_seed"],
            "bootstrap_ci95": metrics["bootstrap_ci95"],
            "accuracy_drop": metrics["accuracy_drop"],
        },
    )
    with run_events_path.open("a", encoding="utf-8") as event_handle:
        event_handle.write(
            json.dumps({"created_at_utc": utc_now(), "event": "ablation_completed"}, sort_keys=True) + "\n"
        )
        event_handle.write(
            json.dumps({"created_at_utc": utc_now(), "event": "run_completed"}, sort_keys=True) + "\n"
        )
    print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, indent=2, sort_keys=True))
    sys.stdout.flush()
    sys.stderr.flush()
    write_json(run_dir / "artifact_hashes.json", build_artifact_hashes(run_dir, schema_version=RM_SCHEMA_VERSION))
    return 0


def build_artifact_hashes(run_dir: Path, *, schema_version: str = SCHEMA_VERSION) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name not in {"artifact_hashes.json", "checker_result.json", "artifact_check.json"}:
            entries.append(
                {
                    "path": str(path.relative_to(run_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    return {
        "schema_version": f"{schema_version}_artifact_hashes",
        "created_at_utc": utc_now(),
        "artifacts": entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", required=True, choices=["outlier_migrate", "residual_migration"])
    parser.add_argument("--run-id")
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--positions")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--bootstrap-samples", type=int, default=THRESHOLDS["bootstrap_samples"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = parser.parse_args(argv)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.branch == "residual_migration":
        args.run_id = args.run_id or f"rm_phase0_{timestamp}"
        args.results_dir = args.results_dir or RM_RESULTS_DIR
        args.max_new_tokens = RM_DEFAULT_MAX_NEW_TOKENS if args.max_new_tokens is None else args.max_new_tokens
        if args.positions is not None:
            raise SystemExit("Residual Migration Phase 0 does not use --positions")
        return run_residual_migration(args, argv)

    args.run_id = args.run_id or f"om_phase0_{timestamp}"
    args.results_dir = args.results_dir or OUTLIER_RESULTS_DIR
    args.positions = parse_positions(args.positions or ",".join(map(str, DEFAULT_POSITIONS)))
    args.max_new_tokens = max(DEFAULT_POSITIONS) if args.max_new_tokens is None else args.max_new_tokens

    if args.model_id != DEFAULT_MODEL_ID:
        raise SystemExit(f"OutlierMigrate Phase 0 requires {DEFAULT_MODEL_ID}; got {args.model_id}")
    if args.prompt_file.resolve() != DEFAULT_PROMPT_FILE.resolve():
        raise SystemExit(f"OutlierMigrate Phase 0 requires canonical prompt file {DEFAULT_PROMPT_FILE}")
    if args.max_new_tokens < max(args.positions):
        raise SystemExit("--max-new-tokens must reach the preregistered 10000-token position")
    if args.bootstrap_samples != THRESHOLDS["bootstrap_samples"]:
        raise SystemExit("OutlierMigrate Phase 0 preregisters bootstrap n=1000")

    run_dir = args.results_dir / args.run_id
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")
    (run_dir / "logs").mkdir(parents=True)
    stdout_log = (run_dir / "logs/stdout.log").open("w", encoding="utf-8", buffering=1)
    stderr_log = (run_dir / "logs/stderr.log").open("w", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, stdout_log)
    sys.stderr = Tee(sys.__stderr__, stderr_log)
    run_events_path = run_dir / "run_events.jsonl"
    run_events_path.write_text(
        json.dumps({"created_at_utc": utc_now(), "event": "run_started"}, sort_keys=True) + "\n",
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

    prompt_manifest, prompt_reasons = build_prompt_manifest(args.prompt_file)
    environment = build_environment()
    model_provenance = resolve_model_snapshot(args.model_id)
    command_metadata = {
        "schema_version": f"{SCHEMA_VERSION}_command",
        "created_at_utc": utc_now(),
        "argv": sys.argv if argv is None else ["run_phase0_branch.py", *argv],
        "cwd": str(Path.cwd()),
        "branch": args.branch,
        "run_dir": str(run_dir),
        "assumptions": [
            "manual greedy decoding is used to expose deterministic decode-step hooks",
            "layer forward output absolute values are treated as per-channel activation magnitudes",
            "EOS is recorded but decoding continues to 10000 generated tokens to satisfy the fixed position grid",
        ],
    }
    random_seed = {
        "schema_version": f"{SCHEMA_VERSION}_random_seed",
        "seed": args.seed,
        "determinism": {"do_sample": False, "num_beams": 1, "torch_manual_seed": args.seed},
    }
    write_json(run_dir / "prompt_manifest.json", prompt_manifest)
    write_json(run_dir / "environment.json", environment)
    write_json(run_dir / "model_provenance.json", model_provenance)
    write_json(run_dir / "command_metadata.json", command_metadata)
    write_json(run_dir / "random_seed.json", random_seed)
    if prompt_reasons:
        write_json(run_dir / "infra_error.json", {"reasons": prompt_reasons})
        print(f"prompt manifest failed preregistered invariants: {prompt_reasons}", file=sys.stderr)
        write_json(run_dir / "artifact_hashes.json", build_artifact_hashes(run_dir))
        return 1
    if not model_provenance.get("snapshot_path"):
        write_json(run_dir / "infra_error.json", {"reasons": [model_provenance.get("error")]})
        print("model snapshot could not be resolved locally; see infra_error.json", file=sys.stderr)
        write_json(run_dir / "artifact_hashes.json", build_artifact_hashes(run_dir))
        return 1

    prompts = prompt_manifest["prompts"]
    model, tokenizer, device = load_model_and_tokenizer(
        model_provenance, dtype_name=args.dtype, device_name=args.device
    )
    activation_path = run_dir / "activation_magnitudes.jsonl.gz"
    activation_manifest = capture_activation_magnitudes(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompts=prompts,
        positions=args.positions,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        output_path=activation_path,
        run_events_path=run_events_path,
    )
    write_json(run_dir / "activation_magnitude_manifest.json", activation_manifest)
    rows = list(iter_activation_rows(activation_path))
    metrics = compute_outlier_migration_metrics(
        rows, positions=args.positions, bootstrap_samples=args.bootstrap_samples, seed=args.seed
    )
    metrics.update(
        {
            "created_at_utc": utc_now(),
            "branch": args.branch,
            "model_id": args.model_id,
            "model_snapshot_commit": model_provenance.get("hf_snapshot_commit"),
            "prompt_source": "AIME-2025",
            "prompt_selection": "deterministic_indices_0_11",
            "prompt_sha256": prompt_manifest["prompt_sha256"],
            "positions": list(args.positions),
            "trace_count": len(prompts),
            "layer_count": activation_manifest["layer_count"],
            "hidden_size": rows[0]["channel_count"] if rows else None,
            "activation_artifact": "activation_magnitudes.jsonl.gz",
            "activation_artifact_sha256": file_sha256(activation_path),
        }
    )
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "bootstrap_ci.json",
        {
            "schema_version": f"{SCHEMA_VERSION}_bootstrap_ci",
            "metric_name": metrics["metric_name"],
            "bootstrap_samples": metrics["bootstrap_samples"],
            "bootstrap_seed": metrics["bootstrap_seed"],
            "bootstrap_ci95": metrics["bootstrap_ci95"],
            "migration_fraction": metrics["migration_fraction"],
        },
    )
    (run_events_path).open("a", encoding="utf-8").write(
        json.dumps({"created_at_utc": utc_now(), "event": "run_completed"}, sort_keys=True) + "\n"
    )
    print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, indent=2, sort_keys=True))
    sys.stdout.flush()
    sys.stderr.flush()
    write_json(run_dir / "artifact_hashes.json", build_artifact_hashes(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
