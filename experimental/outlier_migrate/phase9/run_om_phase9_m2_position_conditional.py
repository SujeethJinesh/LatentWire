#!/usr/bin/env python3
"""Run Phase 9 M2 position-conditional union."""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import math
import random
import shutil
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.outlier_migrate.phase4 import run_om_phase4_intervention as phase4_runner
from experimental.outlier_migrate.phase9 import check_om_phase9_m2_position_conditional as checker
from experimental.shared import run_phase0_branch as shared


DEFAULT_RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase9/results"
DEFAULT_PROMPT_FILE = checker.DEFAULT_PROMPT_FILE
DEFAULT_MODEL_ID = "ibm-granite/granite-4.0-h-small"
SCHEMA_VERSION = checker.SCHEMA_VERSION
SET_SPECS = {
    "static_1pct": {"kind": "single_position", "positions": [100], "fraction": 0.01},
    "static_3pct": {"kind": "single_position", "positions": [100], "fraction": 0.03},
    "m2_set_A": {"kind": "union", "positions": [100, 200, 500], "fraction": 0.01},
    "m2_set_B": {"kind": "union", "positions": [1000, 2000, 5000], "fraction": 0.01},
    "m2_set_C": {"kind": "union", "positions": [7000, 10000, 15000], "fraction": 0.01},
}
M2_SEGMENTS = [
    {"name": "early", "start": 1, "end": 799, "set": "m2_set_A"},
    {"name": "middle", "start": 800, "end": 5999, "set": "m2_set_B"},
    {"name": "late", "start": 6000, "end": checker.SCORING_POSITION, "set": "m2_set_C"},
]
RANDOM_SEGMENTS = [
    {"name": "early", "start": 1, "end": 799, "set": "m2_set_C"},
    {"name": "middle", "start": 800, "end": 5999, "set": "m2_set_B"},
    {"name": "late", "start": 6000, "end": checker.SCORING_POSITION, "set": "m2_set_A"},
]


def resolve_model_snapshot_light(model_id: str) -> dict[str, Any]:
    safe_id = "models--" + model_id.replace("/", "--")
    cache_roots = [Path("/workspace/hf_cache/hub"), Path("/workspace/hf_cache"), Path.home() / ".cache/huggingface/hub"]
    checked: list[str] = []
    expected = checker.MODEL_SNAPSHOTS.get(model_id)
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
                    item["sha256_omitted_reason"] = "large weight shard; HF snapshot commit fixes revision"
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
                "The 24-trace Granite-Small M2 packet completed BF16/static-1/static-3 scoring "
                "but OOMed entering dynamic M2; rerunning the full 24 traces would block paper "
                "progress during vacation mode. The fixed first-12 trace slice preserves the M2 "
                "scientific question with wider uncertainty."
            ),
            "invalidates_if": "Human requires the original 24-trace preregistered M2 packet before interpreting M2.",
        },
    )


def copy_filtered_jsonl_gz(source: Path, dest: Path, *, prompt_indices: set[int]) -> int:
    count = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(source, "rt", encoding="utf-8") as src, gzip.open(dest, "wt", encoding="utf-8") as out:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("prompt_index", -1)) in prompt_indices:
                out.write(json.dumps(row, sort_keys=True) + "\n")
                count += 1
    return count


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


def move_tensor_state(obj: Any, device: Any) -> Any:
    try:
        import torch
    except Exception:
        return obj
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, list):
        for index, item in enumerate(list(obj)):
            obj[index] = move_tensor_state(item, device)
        return obj
    if isinstance(obj, tuple):
        return tuple(move_tensor_state(item, device) for item in obj)
    if isinstance(obj, dict):
        for key, item in list(obj.items()):
            obj[key] = move_tensor_state(item, device)
        return obj
    for attr in ["conv_states", "ssm_states", "key_cache", "value_cache"]:
        states = getattr(obj, attr, None)
        if isinstance(states, list):
            for index, item in enumerate(list(states)):
                states[index] = move_tensor_state(item, device)
    return obj


def top_channels(values: list[float], count: int) -> list[int]:
    return sorted(range(len(values)), key=lambda channel: (-float(values[channel]), channel))[:count]


def build_protected_sets(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_layer_position: dict[int, dict[int, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    layer_names: dict[int, str] = {}
    prompt_indices = sorted({int(row["prompt_index"]) for row in rows})
    for row in rows:
        layer_index = int(row["layer_index"])
        position = int(row["decode_position"])
        layer_names[layer_index] = str(row["layer_name"])
        by_layer_position[layer_index][position].append([float(value) for value in row["channel_magnitudes"]])
    protected: dict[str, Any] = {
        "schema_version": f"{SCHEMA_VERSION}_protected_sets",
        "created_at_utc": shared.utc_now(),
        "selection_basis": "mean absolute layer output activation over fixed AIME-2025 trace set",
        "prompt_indices": prompt_indices,
        "tie_break": "lower channel index",
        "random_bin_assignment": {
            "seed": checker.BOOTSTRAP_SEED,
            "assignment": {"early": "m2_set_C", "middle": "m2_set_B", "late": "m2_set_A"},
        },
        "regimes": {},
    }
    for regime, spec in SET_SPECS.items():
        regime_layers: dict[str, Any] = {}
        for layer_index in sorted(by_layer_position):
            positions = [int(pos) for pos in spec["positions"]]
            channel_count = len(by_layer_position[layer_index][positions[0]][0])
            top_k = max(1, math.ceil(channel_count * float(spec["fraction"])))
            selected: set[int] = set()
            if spec["kind"] == "union":
                for position in positions:
                    vectors = by_layer_position[layer_index][position]
                    means = [mean(vector[channel] for vector in vectors) for channel in range(channel_count)]
                    selected.update(top_channels(means, top_k))
                channels = sorted(selected)
            else:
                vectors = by_layer_position[layer_index][positions[0]]
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


def score_dynamic_segments(
    *,
    model_provenance: dict[str, Any],
    prompts: list[dict[str, Any]],
    target_tokens: dict[int, list[int]],
    protected_sets: dict[str, Any],
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
                model, tokenizer, device = shared.load_model_and_tokenizer(
                    model_provenance, dtype_name=dtype_name, device_name=device_name
                )
                quantized = phase4_runner.apply_quantization(model, protected_sets, str(segment["set"]))
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
                        past_key_values = move_tensor_state(past_key_values, device)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                    for decode_position in range(int(segment["start"]), int(segment["end"]) + 1):
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
                    past_key_values = move_tensor_state(past_key_values, "cpu")
                if attention_mask is not None:
                    attention_mask = attention_mask.cpu()
                if logits is not None:
                    logits = logits.cpu()
                if nll is not None:
                    nll = nll.cpu()
                del target_tensor, model, tokenizer
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
                            "event": "completed_dynamic_scoring_batch",
                            "regime": regime_name,
                            "prompt_indices": batch_indices,
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
    return results, {"regime": regime_name, "segments": quantized_segments}


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
        for regime in ["m2_position_conditional", "static_3pct", "random_bin_assignment"]
    }
    primary = summarize(by_regime["m2_position_conditional"])
    static3 = summarize(by_regime["static_3pct"])
    random_bin = summarize(by_regime["random_bin_assignment"])
    no_gap_count = len(per_trace_rows) - len(included)
    for item in [primary, static3, random_bin]:
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
        "included_trace_count": len(included),
        "no_recoverable_static_gap_count": no_gap_count,
        "no_recoverable_static_gap_fraction": no_gap_count / len(per_trace_rows) if per_trace_rows else 0.0,
        "calibration_positions": list(checker.CALIBRATION_POSITIONS),
        "scoring_position": checker.SCORING_POSITION,
        "scoring_window_tokens": checker.SCORING_WINDOW_TOKENS,
        "metric_name": "positive-static-gap per-trace recovery",
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
            "static_1pct": {"median_recovery": 0.0},
            "static_3pct": static3,
            "random_bin_assignment": random_bin,
        },
        "m2_minus_static_3pct_median": (
            None
            if primary["median_recovery"] is None or static3["median_recovery"] is None
            else float(primary["median_recovery"]) - float(static3["median_recovery"])
        ),
        "m2_minus_random_bin_median": (
            None
            if primary["median_recovery"] is None or random_bin["median_recovery"] is None
            else float(primary["median_recovery"]) - float(random_bin["median_recovery"])
        ),
    }
    return metrics, bootstrap, controls


def main(argv: list[str] | None = None) -> int:
    shared.SCHEMA_VERSION = SCHEMA_VERSION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"om_phase9_m2_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=checker.BOOTSTRAP_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--reuse-activation-run-dir", type=Path)
    parser.add_argument("--reuse-trace-run-dir", type=Path)
    parser.add_argument("--trace-count", type=int, default=checker.TRACE_COUNT)
    args = parser.parse_args(argv)

    if args.model_id not in checker.MODEL_SNAPSHOTS:
        raise SystemExit(f"M2 model is not preregistered: {args.model_id}")
    if args.prompt_file.resolve() != DEFAULT_PROMPT_FILE.resolve():
        raise SystemExit(f"M2 requires canonical prompt file {DEFAULT_PROMPT_FILE}")
    if shared.file_sha256(args.prompt_file) != checker.EXPECTED_PROMPT_FILE_SHA256:
        raise SystemExit("canonical AIME-2025 indices 0-23 prompt file hash drifted")
    if args.seed != checker.BOOTSTRAP_SEED:
        raise SystemExit("M2 preregisters bootstrap/random seed 20260513")
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

    def m2_excepthook(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        write_failure_packet(run_dir, run_events_path, exc)
        previous_excepthook(exc_type, exc, tb)

    sys.excepthook = m2_excepthook
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
    model_provenance = resolve_model_snapshot_light(args.model_id)
    command_metadata = {
        "schema_version": f"{SCHEMA_VERSION}_command",
        "created_at_utc": shared.utc_now(),
        "argv": sys.argv if argv is None else ["run_om_phase9_m2_position_conditional.py", *argv],
        "cwd": str(Path.cwd()),
        "branch": "outlier_migrate_phase9_m2_position_conditional",
        "run_dir": str(run_dir),
        "batch_size": args.batch_size,
        "reuse_activation_run_dir": str(args.reuse_activation_run_dir) if args.reuse_activation_run_dir else None,
        "reuse_trace_run_dir": str(args.reuse_trace_run_dir) if args.reuse_trace_run_dir else None,
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
        "m2_segments": M2_SEGMENTS,
        "random_bin_segments": RANDOM_SEGMENTS,
    }
    quantization_config = {
        "schema_version": f"{SCHEMA_VERSION}_quantization_config",
        "weight_bits": 4,
        "scheme": "symmetric_per_output_channel_int4",
        "signed_integer_range": [-7, 7],
        "activation_dtype": "float16",
        "protected_channel_dtype": "bfloat16",
        "implementation_note": "INT4 weights are represented as dequantized tensors for framework compatibility.",
        "forbidden_methods": ["AWQ-style activation-aware scaling", "SmoothQuant-style activation folding"],
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
        copied_rows = copy_filtered_jsonl_gz(
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
        model, tokenizer, device = shared.load_model_and_tokenizer(
            model_provenance, dtype_name=args.dtype, device_name=args.device
        )
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
    protected_sets = build_protected_sets(list(shared.iter_activation_rows(activation_path)))
    shared.write_json(run_dir / "protected_sets.json", protected_sets)

    if args.reuse_trace_run_dir:
        reuse_trace_dir = args.reuse_trace_run_dir.resolve()
        copied_traces = copy_filtered_jsonl_gz(
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
        model, tokenizer, device = shared.load_model_and_tokenizer(
            model_provenance, dtype_name=args.dtype, device_name=args.device
        )
    else:
        model, tokenizer, device = shared.load_model_and_tokenizer(
            model_provenance, dtype_name=args.dtype, device_name=args.device
        )
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
    target_tokens = phase4_runner.load_trace_tokens(trace_path)

    all_scores: dict[str, dict[int, dict[str, float]]] = {}
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
    for regime in ["static_1pct", "static_3pct"]:
        print(json.dumps({"event": "loading_quantized_regime", "regime": regime, "time": shared.utc_now()}))
        model, tokenizer, device = shared.load_model_and_tokenizer(
            model_provenance, dtype_name=args.dtype, device_name=args.device
        )
        excluded_by_regime[regime] = phase4_runner.apply_quantization(model, protected_sets, regime)
        all_scores[regime] = phase4_runner.score_targets(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompts=prompts,
            target_tokens=target_tokens,
            max_new_tokens=checker.SCORING_POSITION,
            batch_size=args.batch_size,
            use_float16_autocast=True,
            run_events_path=run_events_path,
            regime_name=regime,
        )
        write_score_cache(run_dir, regime, all_scores[regime])
        del model, tokenizer, device
        release_model_memory()
    all_scores["m2_position_conditional"], excluded_by_regime["m2_position_conditional"] = score_dynamic_segments(
        model_provenance=model_provenance,
        prompts=prompts,
        target_tokens=target_tokens,
        protected_sets=protected_sets,
        segments=M2_SEGMENTS,
        batch_size=args.batch_size,
        dtype_name=args.dtype,
        device_name=args.device,
        run_events_path=run_events_path,
        regime_name="m2_position_conditional",
    )
    write_score_cache(run_dir, "m2_position_conditional", all_scores["m2_position_conditional"])
    all_scores["random_bin_assignment"], excluded_by_regime["random_bin_assignment"] = score_dynamic_segments(
        model_provenance=model_provenance,
        prompts=prompts,
        target_tokens=target_tokens,
        protected_sets=protected_sets,
        segments=RANDOM_SEGMENTS,
        batch_size=args.batch_size,
        dtype_name=args.dtype,
        device_name=args.device,
        run_events_path=run_events_path,
        regime_name="random_bin_assignment",
    )
    write_score_cache(run_dir, "random_bin_assignment", all_scores["random_bin_assignment"])
    shared.write_json(
        run_dir / "excluded_tensors.json",
        {"schema_version": f"{SCHEMA_VERSION}_excluded_tensors", "created_at_utc": shared.utc_now(), "by_regime": excluded_by_regime},
    )

    per_trace_rows: list[dict[str, Any]] = []
    for prompt in prompts:
        index = int(prompt["index"])
        perplexities = {regime: float(scores[index]["perplexity"]) for regime, scores in all_scores.items()}
        mean_nll = {regime: float(scores[index]["mean_nll"]) for regime, scores in all_scores.items()}
        static_gap = perplexities["static_1pct"] - perplexities["bf16"]
        no_gap = static_gap <= 0.0
        recoveries = {}
        for regime in ["m2_position_conditional", "static_3pct", "random_bin_assignment"]:
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
    shared.write_json(run_dir / "per_trace_metrics.json", {"schema_version": f"{SCHEMA_VERSION}_per_trace_metrics", "created_at_utc": shared.utc_now(), "traces": per_trace_rows})
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
