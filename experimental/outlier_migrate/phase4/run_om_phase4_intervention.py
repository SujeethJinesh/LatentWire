#!/usr/bin/env python3
"""Run the preregistered OutlierMigrate Phase 4 intervention gate."""

from __future__ import annotations

import argparse
import gc
import gzip
import inspect
import json
import math
import random
import shutil
import sys
import traceback
import types
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.outlier_migrate.phase4 import check_om_phase4_intervention as checker
from experimental.shared import run_phase0_branch as shared


DEFAULT_RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase4/results"
DEFAULT_PROMPT_FILE = checker.DEFAULT_PROMPT_FILE
DEFAULT_MODEL_ID = checker.MODEL_ID
DEFAULT_SEED = checker.BOOTSTRAP_SEED
ALL_CALIBRATION_POSITIONS = tuple(sorted(set(checker.DENSE_GRID)))
SCHEMA_VERSION = checker.SCHEMA_VERSION


def resolve_model_snapshot_light(model_id: str) -> dict[str, Any]:
    """Resolve a local HF snapshot without hashing multi-GB weight shards."""

    safe_id = "models--" + model_id.replace("/", "--")
    cache_roots = [
        Path("/workspace/hf_cache/hub"),
        Path("/workspace/hf_cache"),
        Path.home() / ".cache/huggingface/hub",
    ]
    checked: list[str] = []
    for cache_root in cache_roots:
        repo_dir = cache_root / safe_id
        checked.append(str(repo_dir))
        refs_main = repo_dir / "refs/main"
        snapshots_dir = repo_dir / "snapshots"
        commits: list[str] = []
        if refs_main.is_file():
            commits.append(refs_main.read_text(encoding="utf-8").strip())
        if snapshots_dir.is_dir():
            commits.extend(path.name for path in snapshots_dir.iterdir() if path.is_dir())
        expected = checker.EXPECTED_MODEL_SNAPSHOT_COMMIT
        if expected in commits:
            commits = [expected, *[commit for commit in commits if commit != expected]]
        for commit in dict.fromkeys(commits):
            snapshot = snapshots_dir / commit
            if not (snapshot / "config.json").exists():
                continue
            files: list[dict[str, Any]] = []
            for path in sorted(snapshot.iterdir()):
                if not (path.is_file() or path.is_symlink()):
                    continue
                resolved = path.resolve()
                item: dict[str, Any] = {
                    "path": path.name,
                    "resolved_path": str(resolved),
                    "bytes": resolved.stat().st_size if resolved.exists() else None,
                }
                if path.suffix in {".safetensors", ".bin"}:
                    item["sha256"] = None
                    item["sha256_omitted_reason"] = "large weight shard; hf snapshot commit fixes revision"
                else:
                    item["sha256"] = shared.file_sha256(resolved) if resolved.is_file() else None
                files.append(item)
            return {
                "schema_version": f"{SCHEMA_VERSION}_model_provenance",
                "created_at_utc": shared.utc_now(),
                "model_id": model_id,
                "local_files_only": True,
                "hf_snapshot_commit": commit,
                "snapshot_path": str(snapshot),
                "cache_repo_path": str(repo_dir),
                "checked_cache_paths": checked,
                "files": files,
                "large_weight_sha_policy": "snapshot commit records exact model revision",
            }
    return {
        "schema_version": f"{SCHEMA_VERSION}_model_provenance",
        "created_at_utc": shared.utc_now(),
        "model_id": model_id,
        "local_files_only": True,
        "hf_snapshot_commit": None,
        "snapshot_path": None,
        "checked_cache_paths": checked,
        "error": "no local HuggingFace snapshot with config.json found",
    }


def parse_prompt_file(prompt_file: Path) -> tuple[list[dict[str, Any]], list[str]]:
    reasons: list[str] = []
    prompts: list[dict[str, Any]] = []
    if not prompt_file.is_file():
        return [], [f"prompt file missing: {prompt_file}"]
    for row_index, line in enumerate(prompt_file.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except Exception as exc:
            return [], [f"bad prompt JSON row {row_index}: {exc!r}"]
        index = int(item.get("index", row_index))
        prompt = item.get("prompt") or item.get("problem") or item.get("question")
        prompts.append(
            {
                "index": index,
                "prompt_id": str(item.get("prompt_id", item.get("id", row_index))),
                "prompt": str(prompt),
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
    if [row["index"] for row in prompts] != list(range(checker.TRACE_COUNT)):
        reasons.append("prompt indices are not exactly 0-23")
    if len(prompts) != checker.TRACE_COUNT:
        reasons.append("prompt count is not 24")
    return prompts, reasons


def build_prompt_manifest(prompt_file: Path) -> tuple[dict[str, Any], list[str]]:
    prompts, reasons = parse_prompt_file(prompt_file)
    return (
        {
            "schema_version": f"{SCHEMA_VERSION}_prompt_manifest",
            "created_at_utc": shared.utc_now(),
            "source": "AIME-2025",
            "selection": "deterministic_indices_0_23",
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


def write_environment_text(path: Path, environment: dict[str, Any]) -> None:
    nvidia = environment.get("commands", {}).get("nvidia_smi", {})
    pip_freeze = environment.get("commands", {}).get("pip_freeze", {})
    text = [
        f"created_at_utc: {environment.get('created_at_utc')}",
        f"python: {environment.get('python', {}).get('version')} ({environment.get('python', {}).get('executable')})",
        f"platform: {environment.get('platform', {}).get('platform')}",
        f"git_sha: {environment.get('repo', {}).get('git_sha')}",
        f"torch: {environment.get('torch', {}).get('version')}",
        f"cuda_available: {environment.get('torch', {}).get('cuda_available')}",
        "",
        "nvidia-smi:",
        str(nvidia.get("stdout", "")),
        "",
        "pip freeze:",
        str(pip_freeze.get("stdout", "")),
        "",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def write_failure_packet(run_dir: Path, run_events_path: Path, exc: BaseException | str) -> None:
    """Best-effort terminal packet for partial long runs."""

    reason = str(exc)
    payload: dict[str, Any] = {
        "schema_version": f"{SCHEMA_VERSION}_infra_error",
        "created_at_utc": shared.utc_now(),
        "decision": checker.FAIL_INFRA,
        "reason": reason,
    }
    if isinstance(exc, BaseException):
        payload["exception_type"] = type(exc).__name__
        payload["traceback"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    try:
        run_events_path.open("a", encoding="utf-8").write(
            json.dumps(
                {
                    "created_at_utc": shared.utc_now(),
                    "event": "run_failed",
                    "reason": reason,
                },
                sort_keys=True,
            )
            + "\n"
        )
    except Exception:
        pass
    try:
        shared.write_json(run_dir / "infra_error.json", payload)
        shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
    except Exception:
        return
    try:
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


def make_cache_helpers(model: Any, *, cache_dtype: Any | None = None):
    import torch

    forward_parameters = set(inspect.signature(model.forward).parameters)
    cache_input_name = "cache_params" if "cache_params" in forward_parameters else "past_key_values"

    def align_cache_state_dtypes(cache: Any) -> None:
        if cache_dtype is None:
            return
        for attr in ["conv_states", "ssm_states", "key_cache", "value_cache"]:
            states = getattr(cache, attr, None)
            if not isinstance(states, list):
                continue
            for index, state_tensor in enumerate(list(states)):
                if torch.is_tensor(state_tensor) and state_tensor.numel() > 0 and state_tensor.dtype != cache_dtype:
                    states[index] = state_tensor.to(dtype=cache_dtype)

    def patch_hybrid_cache(cache: Any) -> Any:
        align_cache_state_dtypes(cache)
        if getattr(cache, "_latentwire_cache_patch", False):
            return cache
        config = getattr(model, "config", None)
        conv_states = getattr(cache, "conv_states", None)
        ssm_states = getattr(cache, "ssm_states", None)
        if config is None or not isinstance(conv_states, list) or not isinstance(ssm_states, list):
            return cache
        conv_kernel = getattr(config, "conv_kernel", None)
        if conv_kernel is not None and not hasattr(cache, "conv_kernel_size"):
            cache.conv_kernel_size = int(conv_kernel)
        needed = [
            getattr(config, "mamba_num_heads", None),
            getattr(config, "mamba_head_dim", None),
            getattr(config, "n_groups", None),
            getattr(config, "ssm_state_size", None),
            conv_kernel,
        ]
        if all(value is not None for value in needed):
            conv_dim = (
                int(config.mamba_num_heads) * int(config.mamba_head_dim)
                + 2 * int(config.n_groups) * int(config.ssm_state_size)
            )
            for layer_idx, state_tensor in enumerate(list(conv_states)):
                if not torch.is_tensor(state_tensor) or state_tensor.numel() == 0 or state_tensor.ndim != 3:
                    continue
                if state_tensor.shape[1] == conv_dim and state_tensor.shape[2] == int(conv_kernel):
                    continue
                conv_states[layer_idx] = torch.zeros(
                    state_tensor.shape[0],
                    conv_dim,
                    int(conv_kernel),
                    device=state_tensor.device,
                    dtype=state_tensor.dtype,
                )

        def update_conv_state(self: Any, layer_idx: int, new_conv_state: Any, cache_init: bool = False) -> Any:
            target_device = self.conv_states[layer_idx].device
            target_dtype = self.conv_states[layer_idx].dtype
            if cache_init:
                self.conv_states[layer_idx] = new_conv_state.to(device=target_device, dtype=target_dtype)
            else:
                self.conv_states[layer_idx] = self.conv_states[layer_idx].roll(shifts=-1, dims=-1)
                self.conv_states[layer_idx][:, :, -1] = new_conv_state[:, 0, :].to(
                    device=target_device, dtype=target_dtype
                )
            return self.conv_states[layer_idx]

        def update_ssm_state(self: Any, layer_idx: int, new_ssm_state: Any) -> Any:
            self.ssm_states[layer_idx] = new_ssm_state.to(
                device=self.ssm_states[layer_idx].device,
                dtype=self.ssm_states[layer_idx].dtype,
            )
            return self.ssm_states[layer_idx]

        cache.update_conv_state = types.MethodType(update_conv_state, cache)
        cache.update_ssm_state = types.MethodType(update_ssm_state, cache)
        cache._latentwire_cache_patch = True
        return cache

    def normalize_cache_inputs(model_inputs: dict[str, Any]) -> dict[str, Any]:
        if cache_input_name == "cache_params" and "past_key_values" in model_inputs:
            model_inputs["cache_params"] = model_inputs.pop("past_key_values")
        if "cache_params" in model_inputs:
            model_inputs["cache_params"] = patch_hybrid_cache(model_inputs["cache_params"])
        return model_inputs

    def output_cache(outputs: Any) -> Any:
        return getattr(outputs, "past_key_values", None) or getattr(outputs, "cache_params", None)

    return normalize_cache_inputs, output_cache


def set_granite_fast_path_enabled(enabled: bool) -> Any:
    """Toggle GraniteMoeHybrid's optional fast path; return the previous value."""

    try:
        from transformers.models.granitemoehybrid import modeling_granitemoehybrid
    except Exception:
        return None
    previous = getattr(modeling_granitemoehybrid, "is_fast_path_available", None)
    if previous is not None:
        modeling_granitemoehybrid.is_fast_path_available = bool(enabled)
    return previous


def generate_bf16_traces(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    prompts: list[dict[str, Any]],
    max_new_tokens: int,
    batch_size: int,
    output_path: Path,
    run_events_path: Path,
) -> dict[str, Any]:
    import torch

    normalize_cache_inputs, output_cache = make_cache_helpers(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    prompt_events: list[dict[str, Any]] = []
    with gzip.open(output_path, "wt", encoding="utf-8") as handle, torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            batch_indices = [int(item["index"]) for item in batch]
            texts = [shared.make_prompt_text(str(item["prompt"])) for item in batch]
            encoded = tokenizer(texts, padding=True, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
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
                raise RuntimeError("model did not return cache for BF16 trace generation")
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            generated: dict[int, list[int]] = {index: [] for index in batch_indices}
            eos_first_seen: dict[int, int | None] = {index: None for index in batch_indices}
            for decode_position in range(1, max_new_tokens + 1):
                token_values = [int(value) for value in next_token.tolist()]
                for offset, prompt_index in enumerate(batch_indices):
                    generated[prompt_index].append(token_values[offset])
                    if (
                        eos_token_id is not None
                        and token_values[offset] == int(eos_token_id)
                        and eos_first_seen[prompt_index] is None
                    ):
                        eos_first_seen[prompt_index] = decode_position
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
                    input_ids=next_token[:, None],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    use_cache=True,
                )
                outputs = model(**normalize_cache_inputs(model_inputs))
                past_key_values = output_cache(outputs)
                if past_key_values is None:
                    raise RuntimeError(f"model dropped cache at decode position {decode_position}")
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            for offset, prompt_index in enumerate(batch_indices):
                row = {
                    "schema_version": f"{SCHEMA_VERSION}_bf16_trace",
                    "prompt_index": prompt_index,
                    "prompt_id": batch[offset]["prompt_id"],
                    "generated_token_count": len(generated[prompt_index]),
                    "first_eos_decode_position": eos_first_seen[prompt_index],
                    "token_ids": generated[prompt_index],
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                rows += 1
                prompt_events.append(
                    {
                        "prompt_index": prompt_index,
                        "prompt_id": batch[offset]["prompt_id"],
                        "input_token_count": int(encoded["attention_mask"][offset].sum().item()),
                        "generated_token_count": len(generated[prompt_index]),
                        "first_eos_decode_position": eos_first_seen[prompt_index],
                    }
                )
            with run_events_path.open("a", encoding="utf-8") as event_handle:
                event_handle.write(
                    json.dumps(
                        {
                            "created_at_utc": shared.utc_now(),
                            "event": "completed_bf16_trace_batch",
                            "prompt_indices": batch_indices,
                            "max_new_tokens": max_new_tokens,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    return {
        "schema_version": f"{SCHEMA_VERSION}_bf16_trace_manifest",
        "created_at_utc": shared.utc_now(),
        "artifact": "bf16_traces.jsonl.gz",
        "artifact_sha256": shared.file_sha256(output_path),
        "trace_count": rows,
        "max_new_tokens": max_new_tokens,
        "decode_policy": "manual greedy decode; EOS recorded but ignored for fixed-length traces",
        "prompt_events": prompt_events,
    }


def load_trace_tokens(path: Path) -> dict[int, list[int]]:
    traces: dict[int, list[int]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            traces[int(row["prompt_index"])] = [int(value) for value in row["token_ids"]]
    return traces


def score_targets(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    prompts: list[dict[str, Any]],
    target_tokens: dict[int, list[int]],
    max_new_tokens: int,
    batch_size: int,
    use_float16_autocast: bool,
    run_events_path: Path,
    regime_name: str,
) -> dict[int, dict[str, float]]:
    import torch

    autocast_enabled = bool(use_float16_autocast and torch.cuda.is_available())
    cache_dtype = torch.float16 if autocast_enabled else None
    normalize_cache_inputs, output_cache = make_cache_helpers(model, cache_dtype=cache_dtype)
    score_start = checker.SCORING_POSITION - checker.SCORING_WINDOW_TOKENS + 1
    score_end = checker.SCORING_POSITION
    results: dict[int, dict[str, float]] = {}
    previous_fast_path = set_granite_fast_path_enabled(False) if autocast_enabled else None
    try:
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
                nll = torch.zeros((len(batch),), device=device, dtype=torch.float64)
                scored = 0
                with torch.autocast("cuda", dtype=torch.float16, enabled=autocast_enabled):
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
                        raise RuntimeError(f"model did not return cache for scoring regime {regime_name}")
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
                            raise RuntimeError(f"model dropped cache while scoring {regime_name} at {decode_position}")
                        logits = outputs.logits[:, -1, :]
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
                                "event": "completed_scoring_batch",
                                "regime": regime_name,
                                "prompt_indices": batch_indices,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
    finally:
        if previous_fast_path is not None:
            set_granite_fast_path_enabled(bool(previous_fast_path))
    return results


def collect_activation_rows(path: Path) -> list[dict[str, Any]]:
    return list(shared.iter_activation_rows(path))


def top_channels(values: list[float], count: int) -> list[int]:
    return sorted(range(len(values)), key=lambda channel: (-float(values[channel]), channel))[:count]


def build_protected_sets(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_layer_position: dict[int, dict[int, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    layer_names: dict[int, str] = {}
    for row in rows:
        layer_index = int(row["layer_index"])
        position = int(row["decode_position"])
        layer_names[layer_index] = str(row["layer_name"])
        by_layer_position[layer_index][position].append([float(value) for value in row["channel_magnitudes"]])
    regimes = {
        "static_1pct": {"kind": "single_position", "positions": [100], "fraction": 0.01},
        "union_primary": {"kind": "union", "positions": list(checker.PRIMARY_GRID), "fraction": 0.01},
        "static_2pct": {"kind": "single_position", "positions": [100], "fraction": 0.02},
        "magnitude_average": {"kind": "mean_across_positions", "positions": list(checker.PRIMARY_GRID), "fraction": 0.01},
        "grid_sparse": {"kind": "union", "positions": list(checker.SPARSE_GRID), "fraction": 0.01},
        "grid_dense": {"kind": "union", "positions": list(checker.DENSE_GRID), "fraction": 0.01},
    }
    protected: dict[str, Any] = {
        "schema_version": f"{SCHEMA_VERSION}_protected_sets",
        "created_at_utc": shared.utc_now(),
        "selection_basis": "mean absolute layer output activation over fixed 24-trace AIME-2025 set",
        "tie_break": "lower channel index",
        "regimes": {},
    }
    for regime, spec in regimes.items():
        regime_layers: dict[str, Any] = {}
        for layer_index in sorted(by_layer_position):
            positions = [int(pos) for pos in spec["positions"]]
            first_position = positions[0]
            channel_count = len(by_layer_position[layer_index][first_position][0])
            top_k = max(1, math.ceil(channel_count * float(spec["fraction"])))
            if spec["kind"] == "union":
                selected: set[int] = set()
                for position in positions:
                    vectors = by_layer_position[layer_index][position]
                    means = [mean(vector[channel] for vector in vectors) for channel in range(channel_count)]
                    selected.update(top_channels(means, top_k))
                channels = sorted(selected)
            elif spec["kind"] == "mean_across_positions":
                per_position_means = []
                for position in positions:
                    vectors = by_layer_position[layer_index][position]
                    per_position_means.append(
                        [mean(vector[channel] for vector in vectors) for channel in range(channel_count)]
                    )
                means = [
                    mean(position_means[channel] for position_means in per_position_means)
                    for channel in range(channel_count)
                ]
                channels = sorted(top_channels(means, top_k))
            else:
                vectors = by_layer_position[layer_index][first_position]
                means = [mean(vector[channel] for vector in vectors) for channel in range(channel_count)]
                channels = sorted(top_channels(means, top_k))
            regime_layers[str(layer_index)] = {
                "layer_name": layer_names[layer_index],
                "channel_count": channel_count,
                "requested_top_k": top_k,
                "protected_count": len(channels),
                "protected_channels": channels,
            }
        protected["regimes"][regime] = {**spec, "layers": regime_layers}
    return protected


def quantize_weight_symmetric_int4_inplace(weight: Any, *, row_chunk: int = 512) -> None:
    import torch

    if weight.ndim == 2:
        flat = weight
    elif weight.ndim == 3:
        # Expert banks are represented as [expert, output_channel, input_channel].
        flat = weight.view(-1, weight.shape[-1])
    else:
        return
    original_dtype = weight.dtype
    for start in range(0, flat.shape[0], row_chunk):
        block = flat[start : start + row_chunk]
        work = block.float()
        scale = work.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 7.0
        dequant = torch.round(work / scale).clamp(-7, 7).mul_(scale).to(dtype=original_dtype)
        block.copy_(dequant)


def is_tied_lm_head(model: Any, module_name: str, module: Any) -> bool:
    if module_name != "lm_head" or not bool(getattr(model.config, "tie_word_embeddings", False)):
        return False
    try:
        input_embeddings = model.get_input_embeddings()
    except Exception:
        input_embeddings = None
    return input_embeddings is not None and getattr(module, "weight", None) is getattr(input_embeddings, "weight", None)


def quantize_module_weight(
    *,
    module_name: str,
    module: Any,
    protected: set[int],
    hidden_size: int,
    protection_source: str,
) -> dict[str, Any] | None:
    import torch

    weight = getattr(module, "weight", None)
    if not torch.is_tensor(weight):
        return None
    if weight.ndim not in {2, 3}:
        return {
            "name": module_name,
            "reason": "weight tensor is not 2D or expert-bank 3D",
            "shape": list(weight.shape),
        }
    if weight.ndim == 2:
        row_protected = weight.shape[0] == hidden_size
        col_protected = weight.shape[1] == hidden_size
        rows = torch.tensor(sorted(protected), device=weight.device, dtype=torch.long) if row_protected and protected else None
        cols = torch.tensor(sorted(protected), device=weight.device, dtype=torch.long) if col_protected and protected else None
        saved_rows = weight.index_select(0, rows).clone() if rows is not None else None
        saved_cols = weight.index_select(1, cols).clone() if cols is not None else None
        quantize_weight_symmetric_int4_inplace(weight)
        if row_protected and protected:
            weight.index_copy_(0, rows, saved_rows)
        if col_protected and protected:
            weight.index_copy_(1, cols, saved_cols)
    else:
        row_protected = weight.shape[1] == hidden_size
        col_protected = weight.shape[2] == hidden_size
        rows = torch.tensor(sorted(protected), device=weight.device, dtype=torch.long) if row_protected and protected else None
        cols = torch.tensor(sorted(protected), device=weight.device, dtype=torch.long) if col_protected and protected else None
        saved_rows = weight.index_select(1, rows).clone() if rows is not None else None
        saved_cols = weight.index_select(2, cols).clone() if cols is not None else None
        quantize_weight_symmetric_int4_inplace(weight)
        if row_protected and protected:
            weight.index_copy_(1, rows, saved_rows)
        if col_protected and protected:
            weight.index_copy_(2, cols, saved_cols)
    return {
        "name": module_name,
        "shape": list(weight.shape),
        "row_protected": bool(row_protected),
        "col_protected": bool(col_protected),
        "protected_channel_count": len(protected) if (row_protected or col_protected) else 0,
        "protection_source": protection_source,
        "weight_kind": "linear_2d" if weight.ndim == 2 else "expert_bank_3d",
    }


def apply_quantization(model: Any, protected_sets: dict[str, Any], regime: str) -> dict[str, Any]:
    import torch

    hidden_size = int(getattr(model.config, "hidden_size"))
    layers, layer_origin = shared.discover_transformer_layers(model)
    protected_by_layer = protected_sets["regimes"][regime]["layers"]
    quantized: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    processed_module_ids: set[int] = set()
    with torch.no_grad():
        for layer_index, (layer_name, layer) in enumerate(layers):
            protected = set(int(ch) for ch in protected_by_layer[str(layer_index)]["protected_channels"])
            for module_name, module in layer.named_modules():
                full_name = f"{layer_name}.{module_name}" if module_name else layer_name
                item = quantize_module_weight(
                    module_name=full_name,
                    module=module,
                    protected=protected,
                    hidden_size=hidden_size,
                    protection_source=f"layer_{layer_index}",
                )
                if item is None:
                    continue
                processed_module_ids.add(id(module))
                if "reason" in item:
                    excluded.append(item)
                else:
                    quantized.append(item)
        first_layer_protected = set(int(ch) for ch in protected_by_layer["0"]["protected_channels"])
        last_layer_index = str(max(int(key) for key in protected_by_layer))
        last_layer_protected = set(int(ch) for ch in protected_by_layer[last_layer_index]["protected_channels"])
        for name, module in model.named_modules():
            if id(module) in processed_module_ids:
                continue
            weight = getattr(module, "weight", None)
            if not torch.is_tensor(weight):
                continue
            if is_tied_lm_head(model, name, module):
                excluded.append(
                    {
                        "name": name,
                        "reason": "tied input/output embedding; quantizing it would confound prompt embeddings with layer protection",
                        "shape": list(weight.shape),
                    }
                )
                continue
            outside_protected: set[int] = set()
            protection_source = "outside_transformer_stack_no_hidden_axis"
            if weight.ndim == 2 and weight.shape[1] == hidden_size:
                outside_protected.update(last_layer_protected)
                protection_source = "outside_transformer_stack_hidden_input_uses_last_layer"
            if weight.ndim == 2 and weight.shape[0] == hidden_size:
                outside_protected.update(first_layer_protected)
                protection_source = "outside_transformer_stack_hidden_output_uses_first_layer"
            item = quantize_module_weight(
                module_name=name,
                module=module,
                protected=outside_protected,
                hidden_size=hidden_size,
                protection_source=protection_source,
            )
            if item is None:
                continue
            if "reason" in item:
                excluded.append(item)
            else:
                quantized.append(item)
    return {
        "regime": regime,
        "layer_origin": layer_origin,
        "quantized_tensor_count": len(quantized),
        "quantized_tensors": quantized,
        "excluded_tensors": excluded,
    }


def summarize(values: list[float]) -> dict[str, Any]:
    above_half_count = sum(1 for value in values if float(value) > 0.50)
    return {
        "median_recovery": float(median(values)),
        "mean_recovery": float(mean(values)),
        "bootstrap_ci95": checker.bootstrap_median(values),
        "trace_count": len(values),
        "per_trace_recovery": values,
        "recovery_gt_0_50_count": above_half_count,
        "recovery_gt_0_50_fraction": above_half_count / len(values) if values else 0.0,
    }


def build_metrics(
    *,
    run_dir: Path,
    prompt_manifest: dict[str, Any],
    model_provenance: dict[str, Any],
    per_trace_rows: list[dict[str, Any]],
    protected_sets: dict[str, Any],
    activation_path: Path,
    bf16_trace_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    recovery_by_regime: dict[str, list[float]] = defaultdict(list)
    no_gap_count = sum(1 for row in per_trace_rows if bool(row.get("no_recoverable_static_gap")))
    for row in per_trace_rows:
        for regime, recovery in row["recoveries"].items():
            recovery_by_regime[regime].append(float(recovery))
    primary = summarize(recovery_by_regime["union_primary"])
    primary["no_recoverable_static_gap_count"] = no_gap_count
    primary["no_recoverable_static_gap_fraction"] = no_gap_count / len(per_trace_rows) if per_trace_rows else 0.0
    controls = {
        "schema_version": f"{SCHEMA_VERSION}_control_metrics",
        "created_at_utc": shared.utc_now(),
        "controls": {
            "static_2pct": summarize(recovery_by_regime["static_2pct"]),
            "magnitude_average": summarize(recovery_by_regime["magnitude_average"]),
        },
        "union_primary": primary,
        "union_outperforms_both_controls": (
            primary["median_recovery"] > median(recovery_by_regime["static_2pct"])
            and primary["median_recovery"] > median(recovery_by_regime["magnitude_average"])
        ),
    }
    grid = {
        "schema_version": f"{SCHEMA_VERSION}_grid_sensitivity",
        "created_at_utc": shared.utc_now(),
        "grids": {
            "sparse": {"positions": list(checker.SPARSE_GRID), **summarize(recovery_by_regime["grid_sparse"])},
            "primary": {"positions": list(checker.PRIMARY_GRID), **primary},
            "dense": {"positions": list(checker.DENSE_GRID), **summarize(recovery_by_regime["grid_dense"])},
        },
    }
    bootstrap = {
        "schema_version": f"{SCHEMA_VERSION}_bootstrap_ci",
        "metric_name": "median_per_trace_recovery",
        "bootstrap_samples": checker.BOOTSTRAP_SAMPLES,
        "bootstrap_seed": checker.BOOTSTRAP_SEED,
        "primary_result": primary,
    }
    metrics = {
        "schema_version": f"{SCHEMA_VERSION}_metrics",
        "created_at_utc": shared.utc_now(),
        "preregistration": str(checker.PREREG_PATH.relative_to(ROOT)),
        "preregistration_sha256": shared.file_sha256(checker.PREREG_PATH),
        "model_id": checker.MODEL_ID,
        "model_snapshot_commit": model_provenance.get("hf_snapshot_commit"),
        "prompt_sha256": prompt_manifest["prompt_sha256"],
        "trace_count": checker.TRACE_COUNT,
        "no_recoverable_static_gap_count": no_gap_count,
        "no_recoverable_static_gap_fraction": no_gap_count / len(per_trace_rows) if per_trace_rows else 0.0,
        "primary_grid": list(checker.PRIMARY_GRID),
        "calibration_positions": list(ALL_CALIBRATION_POSITIONS),
        "scoring_position": checker.SCORING_POSITION,
        "scoring_window_tokens": checker.SCORING_WINDOW_TOKENS,
        "metric_name": "per_trace_recovery",
        "metric_formula": "1 - (perplexity_regime - perplexity_BF16) / (perplexity_static_1pct - perplexity_BF16)",
        "primary_result": primary,
        "thresholds": checker.THRESHOLDS,
        "artifacts": {
            "activation_magnitudes": "activation_magnitudes.jsonl.gz",
            "activation_magnitudes_sha256": shared.file_sha256(activation_path),
            "bf16_traces": "bf16_traces.jsonl.gz",
            "bf16_traces_sha256": shared.file_sha256(bf16_trace_path),
            "run_dir": str(run_dir),
        },
        "protected_set_summary": {
            regime: {
                "positions": spec["positions"],
                "kind": spec["kind"],
                "median_protected_count": float(
                    median(layer["protected_count"] for layer in spec["layers"].values())
                ),
            }
            for regime, spec in protected_sets["regimes"].items()
        },
    }
    return metrics, bootstrap, controls, grid


def main(argv: list[str] | None = None) -> int:
    shared.SCHEMA_VERSION = SCHEMA_VERSION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"om_phase4_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=checker.SCORING_POSITION)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument(
        "--reuse-prequant-run-dir",
        type=Path,
        help=(
            "Optional prior Phase 4 run dir whose activation/protected-set/BF16-trace artifacts "
            "are reused. The prior run must have failed before quantized metrics were produced."
        ),
    )
    args = parser.parse_args(argv)

    if args.model_id != DEFAULT_MODEL_ID:
        raise SystemExit(f"Phase 4 preregisters {DEFAULT_MODEL_ID}; got {args.model_id}")
    if args.prompt_file.resolve() != DEFAULT_PROMPT_FILE.resolve():
        raise SystemExit(f"Phase 4 requires canonical prompt file {DEFAULT_PROMPT_FILE}")
    if shared.file_sha256(args.prompt_file) != checker.EXPECTED_PROMPT_FILE_SHA256:
        raise SystemExit("canonical AIME-2025 indices 0-23 prompt file hash drifted")
    if args.max_new_tokens != checker.SCORING_POSITION:
        raise SystemExit("Phase 4 core gate preregisters scoring at decode position 10000")
    if args.seed != DEFAULT_SEED:
        raise SystemExit("Phase 4 preregisters bootstrap/random seed 20260510")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")

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

    def phase4_excepthook(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        write_failure_packet(run_dir, run_events_path, exc)
        previous_excepthook(exc_type, exc, tb)

    sys.excepthook = phase4_excepthook
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

    prompt_manifest, prompt_reasons = build_prompt_manifest(args.prompt_file)
    environment = shared.build_environment(schema_version=SCHEMA_VERSION)
    model_provenance = resolve_model_snapshot_light(args.model_id)
    command_metadata = {
        "schema_version": f"{SCHEMA_VERSION}_command",
        "created_at_utc": shared.utc_now(),
        "argv": sys.argv if argv is None else ["run_om_phase4_intervention.py", *argv],
        "cwd": str(Path.cwd()),
        "branch": "outlier_migrate_phase4_intervention",
        "run_dir": str(run_dir),
        "batch_size": args.batch_size,
        "reuse_prequant_run_dir": str(args.reuse_prequant_run_dir) if args.reuse_prequant_run_dir else None,
        "notes": [
            "No Phase 4 quantization was run before approved preregistration commit 80bfbc26.",
            "BF16 traces are generated deterministically, then all regimes score the same token windows.",
            "Quantized regimes use symmetric INT4-dequantized weights and float16 autocast for scoring.",
            "Default batch size is 1 to avoid expected Granite-Small OOM on a 96GB single GPU.",
        ],
    }
    random_seed = {
        "schema_version": f"{SCHEMA_VERSION}_random_seed",
        "seed": args.seed,
        "determinism": {"do_sample": False, "num_beams": 1, "torch_manual_seed": args.seed},
    }
    decoding_config = {
        "schema_version": f"{SCHEMA_VERSION}_decoding_config",
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "eos_policy": "record first EOS but continue to fixed decode position",
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
        "implementation_note": (
            "INT4 weights are represented as dequantized tensors for framework compatibility; "
            "quantization levels are exactly symmetric signed 4-bit per output channel. "
            "Protected entries are restored from the original BF16 checkpoint tensor before scoring; "
            "quantized-regime scoring uses FP16 autocast for activations."
        ),
        "forbidden_methods": [
            "AWQ-style activation-aware scaling",
            "SmoothQuant-style activation folding",
        ],
    }
    shared.write_json(run_dir / "prompt_manifest.json", prompt_manifest)
    shared.write_json(run_dir / "environment.json", environment)
    write_environment_text(run_dir / "environment.txt", environment)
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
    if model_provenance.get("hf_snapshot_commit") != checker.EXPECTED_MODEL_SNAPSHOT_COMMIT:
        shared.write_json(
            run_dir / "infra_error.json",
            {
                "reasons": [
                    "local Granite-Small snapshot does not match Phase 4 preregistered commit "
                    f"{checker.EXPECTED_MODEL_SNAPSHOT_COMMIT}",
                ],
                "observed_hf_snapshot_commit": model_provenance.get("hf_snapshot_commit"),
                "expected_hf_snapshot_commit": checker.EXPECTED_MODEL_SNAPSHOT_COMMIT,
            },
        )
        shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
        checker.evaluate(run_dir)
        return 1

    prompts = prompt_manifest["prompts"]
    activation_path = run_dir / "activation_magnitudes.jsonl.gz"
    trace_path = run_dir / "bf16_traces.jsonl.gz"
    if args.reuse_prequant_run_dir:
        reuse_dir = args.reuse_prequant_run_dir.resolve()
        required_reuse = [
            "model_provenance.json",
            "prompt_manifest.json",
            "decoding_config.json",
            "activation_magnitudes.jsonl.gz",
            "activation_magnitude_manifest.json",
            "protected_sets.json",
            "bf16_traces.jsonl.gz",
            "bf16_trace_manifest.json",
        ]
        missing_reuse = [rel for rel in required_reuse if not (reuse_dir / rel).is_file()]
        forbidden_reuse_outputs = [
            "per_trace_metrics.json",
            "metrics.json",
            "bootstrap_ci.json",
            "control_metrics.json",
            "grid_sensitivity_metrics.json",
        ]
        produced_metrics = [rel for rel in forbidden_reuse_outputs if (reuse_dir / rel).exists()]
        if missing_reuse:
            raise SystemExit(f"--reuse-prequant-run-dir missing required files: {missing_reuse}")
        if produced_metrics:
            raise SystemExit(f"--reuse-prequant-run-dir already has metric/checker outputs: {produced_metrics}")
        source_model = json.loads((reuse_dir / "model_provenance.json").read_text(encoding="utf-8"))
        source_prompt = json.loads((reuse_dir / "prompt_manifest.json").read_text(encoding="utf-8"))
        source_decoding = json.loads((reuse_dir / "decoding_config.json").read_text(encoding="utf-8"))
        source_trace = json.loads((reuse_dir / "bf16_trace_manifest.json").read_text(encoding="utf-8"))
        reuse_reasons = []
        if source_model.get("hf_snapshot_commit") != checker.EXPECTED_MODEL_SNAPSHOT_COMMIT:
            reuse_reasons.append("source model snapshot does not match Phase 4 preregistered Granite-Small commit")
        if source_prompt.get("prompt_sha256") != checker.EXPECTED_PROMPT_PAYLOAD_SHA256:
            reuse_reasons.append("source prompt SHA does not match Phase 1 fixed prompt set")
        if source_decoding.get("do_sample") is not False or source_decoding.get("num_beams") != 1:
            reuse_reasons.append("source decoding config is not deterministic greedy")
        if source_decoding.get("scoring_window_tokens") != checker.SCORING_WINDOW_TOKENS:
            reuse_reasons.append("source scoring window is not 512 tokens")
        if source_trace.get("trace_count") != checker.TRACE_COUNT or source_trace.get("max_new_tokens") != checker.SCORING_POSITION:
            reuse_reasons.append("source BF16 traces do not cover all 24 traces to decode position 10000")
        if source_trace.get("artifact_sha256") and source_trace.get("artifact_sha256") != shared.file_sha256(reuse_dir / "bf16_traces.jsonl.gz"):
            reuse_reasons.append("source BF16 trace manifest hash does not match source trace artifact")
        if reuse_reasons:
            raise SystemExit(f"--reuse-prequant-run-dir is not Phase 4 compatible: {reuse_reasons}")
        for rel in [
            "activation_magnitudes.jsonl.gz",
            "activation_magnitude_manifest.json",
            "protected_sets.json",
            "bf16_traces.jsonl.gz",
            "bf16_trace_manifest.json",
        ]:
            shutil.copy2(reuse_dir / rel, run_dir / rel)
        shared.write_json(
            run_dir / "prequant_reuse_manifest.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_prequant_reuse",
                "created_at_utc": shared.utc_now(),
                "source_run_dir": str(reuse_dir),
                "source_files": {
                    rel: {
                        "sha256": shared.file_sha256(run_dir / rel),
                        "bytes": (run_dir / rel).stat().st_size,
                    }
                    for rel in required_reuse
                },
                "reuse_reason": (
                    "source run failed before quantized metrics due FP16 autocast/cache dtype mismatch; "
                    "activation/protected-set/BF16-trace artifacts were produced before the failing stage"
                ),
            },
        )
        protected_sets = json.loads((run_dir / "protected_sets.json").read_text(encoding="utf-8"))
    else:
        model, tokenizer, device = shared.load_model_and_tokenizer(
            model_provenance, dtype_name=args.dtype, device_name=args.device
        )
        activation_manifest = shared.capture_activation_magnitudes(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompts=prompts,
            positions=ALL_CALIBRATION_POSITIONS,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            output_path=activation_path,
            run_events_path=run_events_path,
        )
        shared.write_json(run_dir / "activation_magnitude_manifest.json", activation_manifest)
        protected_sets = build_protected_sets(collect_activation_rows(activation_path))
        shared.write_json(run_dir / "protected_sets.json", protected_sets)
        trace_manifest = generate_bf16_traces(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            output_path=trace_path,
            run_events_path=run_events_path,
        )
        shared.write_json(run_dir / "bf16_trace_manifest.json", trace_manifest)
        release_model_memory(model, tokenizer)
        model = tokenizer = device = None
    target_tokens = load_trace_tokens(trace_path)
    release_model_memory()
    model, tokenizer, device = shared.load_model_and_tokenizer(
        model_provenance, dtype_name=args.dtype, device_name=args.device
    )
    all_scores: dict[str, dict[int, dict[str, float]]] = {}
    all_scores["bf16"] = score_targets(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompts=prompts,
        target_tokens=target_tokens,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        use_float16_autocast=False,
        run_events_path=run_events_path,
        regime_name="bf16",
    )
    release_model_memory(model, tokenizer)
    model = tokenizer = device = None

    excluded_by_regime: dict[str, Any] = {}
    for regime in [
        "static_1pct",
        "union_primary",
        "static_2pct",
        "magnitude_average",
        "grid_sparse",
        "grid_dense",
    ]:
        print(json.dumps({"event": "loading_quantized_regime", "regime": regime, "time": shared.utc_now()}))
        model, tokenizer, device = shared.load_model_and_tokenizer(
            model_provenance, dtype_name=args.dtype, device_name=args.device
        )
        excluded_by_regime[regime] = apply_quantization(model, protected_sets, regime)
        all_scores[regime] = score_targets(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompts=prompts,
            target_tokens=target_tokens,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            use_float16_autocast=True,
            run_events_path=run_events_path,
            regime_name=regime,
        )
        release_model_memory(model, tokenizer)
        model = tokenizer = device = None
    shared.write_json(
        run_dir / "excluded_tensors.json",
        {
            "schema_version": f"{SCHEMA_VERSION}_excluded_tensors",
            "created_at_utc": shared.utc_now(),
            "by_regime": excluded_by_regime,
        },
    )

    per_trace_rows: list[dict[str, Any]] = []
    recovery_regimes = [
        "union_primary",
        "static_2pct",
        "magnitude_average",
        "grid_sparse",
        "grid_dense",
    ]
    for prompt in prompts:
        index = int(prompt["index"])
        perplexities = {regime: float(scores[index]["perplexity"]) for regime, scores in all_scores.items()}
        mean_nll = {regime: float(scores[index]["mean_nll"]) for regime, scores in all_scores.items()}
        static_gap = perplexities["static_1pct"] - perplexities["bf16"]
        no_gap = static_gap <= 0.0
        recoveries = {}
        for regime in recovery_regimes:
            if no_gap:
                recoveries[regime] = 0.0
            else:
                recoveries[regime] = 1.0 - (perplexities[regime] - perplexities["bf16"]) / static_gap
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
        {
            "schema_version": f"{SCHEMA_VERSION}_per_trace_metrics",
            "created_at_utc": shared.utc_now(),
            "traces": per_trace_rows,
        },
    )
    metrics, bootstrap, controls, grid = build_metrics(
        run_dir=run_dir,
        prompt_manifest=prompt_manifest,
        model_provenance=model_provenance,
        per_trace_rows=per_trace_rows,
        protected_sets=protected_sets,
        activation_path=activation_path,
        bf16_trace_path=trace_path,
    )
    shared.write_json(run_dir / "metrics.json", metrics)
    shared.write_json(run_dir / "bootstrap_ci.json", bootstrap)
    shared.write_json(run_dir / "control_metrics.json", controls)
    shared.write_json(run_dir / "grid_sensitivity_metrics.json", grid)
    run_events_path.open("a", encoding="utf-8").write(
        json.dumps({"created_at_utc": shared.utc_now(), "event": "run_completed"}, sort_keys=True) + "\n"
    )
    print(json.dumps({"run_dir": str(run_dir), "primary_result": metrics["primary_result"]}, indent=2, sort_keys=True))
    sys.stdout.flush()
    sys.stderr.flush()
    shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
    checker_result = checker.evaluate(run_dir)
    print(
        json.dumps(
            {
                "checker_decision": checker_result["decision"],
                "artifact_complete": checker_result.get("artifact_complete", False),
                "control_stop_condition": checker_result.get("control_results", {}).get("control_stop_condition", False),
            },
            indent=2,
            sort_keys=True,
        )
    )
    sys.stdout.flush()
    sys.stderr.flush()
    sys.excepthook = previous_excepthook
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
