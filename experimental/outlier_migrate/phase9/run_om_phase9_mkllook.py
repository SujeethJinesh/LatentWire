#!/usr/bin/env python3
"""Run Phase 9 M-KLLOOK sampled forward-KL lookahead oracle."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
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
from experimental.outlier_migrate.phase9 import check_om_phase9_mkllook as checker
from experimental.outlier_migrate.phase9 import run_om_phase9_m11_ema_drift as m11_runner
from experimental.outlier_migrate.phase9 import run_om_phase9_m2_position_conditional as m2_runner
from experimental.shared import run_phase0_branch as shared


SCHEMA_VERSION = checker.SCHEMA_VERSION
DEFAULT_SOURCES = {
    "granite": ROOT
    / "experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z",
    "nemotron": ROOT
    / "experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z",
}
DEFAULT_PROMPT_FILE = ROOT / "experimental/shared/prompts/aime_2025_indices_0_23.jsonl"
EXPECTED_PROMPT_FILE_SHA256 = "sha256:ead004dae0848ad43ad102551f48fa22a0b8ed4a57efecdcf9d7ae387bb6d17a"


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
            json.dumps({"created_at_utc": shared.utc_now(), "event": "run_failed", "reason": str(exc)}, sort_keys=True)
            + "\n"
        )
        shared.write_json(run_dir / "infra_error.json", payload)
        shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
        checker.evaluate(run_dir)
    except Exception:
        pass


def release_model_memory(*objects: Any) -> None:
    phase4_runner.release_model_memory(*objects)


def copy_required_source_artifacts(source_dir: Path, run_dir: Path) -> dict[str, Any]:
    source_artifacts: dict[str, Any] = {
        "schema_version": f"{SCHEMA_VERSION}_source_artifacts",
        "created_at_utc": shared.utc_now(),
        "source_run_dir": str(source_dir),
        "artifacts": [],
    }
    for rel in [
        "activation_magnitude_manifest.json",
        "activation_magnitudes.jsonl.gz",
        "bf16_trace_manifest.json",
        "bf16_traces.jsonl.gz",
    ]:
        src = source_dir / rel
        dst = run_dir / rel
        if not src.is_file():
            raise FileNotFoundError(f"source packet missing {rel}: {source_dir}")
        shutil.copy2(src, dst)
        source_artifacts["artifacts"].append({"name": rel, "path": str(src), "sha256": shared.file_sha256(src)})
    return source_artifacts


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


def load_source_scores(source_dir: Path, run_dir: Path, prompt_indices: set[int], source_artifacts: dict[str, Any]) -> dict[str, dict[int, dict[str, float]]]:
    out: dict[str, dict[int, dict[str, float]]] = {}
    source_names = {"bf16": "bf16", "static_1pct": "static_1pct", "m11b_top10": "m11b_top10"}
    for target_regime, source_regime in source_names.items():
        cached = m11_runner.read_score_cache_any(source_dir, source_regime, expected_prompt_indices=prompt_indices)
        if cached is None:
            raise FileNotFoundError(f"source score cache missing or prompt mismatch: {source_regime}")
        out[target_regime] = cached
        write_score_cache(run_dir, target_regime, cached)
        src = source_dir / "score_cache" / f"{source_regime}.json"
        source_artifacts["artifacts"].append({"name": f"score_cache_{source_regime}", "path": str(src), "sha256": shared.file_sha256(src)})
    return out


def scoring_positions(token_count: int) -> list[int]:
    start = checker.SCORING_POSITION - checker.SCORING_WINDOW_TOKENS + 1
    end = checker.SCORING_POSITION
    if token_count >= checker.SCORING_WINDOW_TOKENS:
        return list(range(start, end + 1))
    if token_count <= 1:
        return [end]
    return sorted({int(round(start + i * (end - start) / (token_count - 1))) for i in range(token_count)})


def activation_means(rows: list[dict[str, Any]]) -> tuple[dict[int, dict[int, list[float]]], dict[int, str]]:
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
            width = len(vectors[0])
            means_by_layer[layer_index][position] = [float(mean(vector[channel] for vector in vectors)) for channel in range(width)]
    return means_by_layer, layer_names


def top_channels(values: list[float], count: int) -> list[int]:
    return sorted(range(len(values)), key=lambda channel: (-float(values[channel]), channel))[:count]


def choose_sampled_layers(layer_indices: list[int], count: int) -> list[int]:
    if count >= len(layer_indices):
        return layer_indices
    if count <= 1:
        return [layer_indices[len(layer_indices) // 2]]
    selected = {
        layer_indices[int(round(i * (len(layer_indices) - 1) / (count - 1)))]
        for i in range(count)
    }
    return sorted(selected)


def build_candidate_plan(
    *,
    activation_rows: list[dict[str, Any]],
    source_protected_sets: dict[str, Any],
    sampled_layer_count: int,
    candidates_per_layer: int,
    seed: int,
    kl_positions: list[int],
    kl_prompt_indices: list[int],
) -> tuple[dict[str, Any], dict[int, list[int]], dict[int, int], dict[int, str]]:
    means_by_layer, layer_names = activation_means(activation_rows)
    available_layers = sorted(int(key) for key in source_protected_sets["regimes"]["m11b_top10"]["layers"])
    sampled_layers = choose_sampled_layers(available_layers, sampled_layer_count)
    candidates_by_layer: dict[int, list[int]] = {}
    selected_counts: dict[int, int] = {}
    rng = random.Random(seed)
    for layer_index in sampled_layers:
        by_position = means_by_layer[layer_index]
        position = checker.SCORING_POSITION if checker.SCORING_POSITION in by_position else max(by_position)
        values = by_position[position]
        budget_count = int(source_protected_sets["regimes"]["m11b_top10"]["layers"][str(layer_index)]["protected_count"])
        top_count = min(len(values), max(candidates_per_layer, budget_count))
        candidate_pool = top_channels(values, top_count)
        if len(candidate_pool) > candidates_per_layer:
            # Always keep the strongest half and sample the rest deterministically
            # from the remaining high-magnitude pool.
            keep = candidate_pool[: max(1, candidates_per_layer // 2)]
            rest = candidate_pool[max(1, candidates_per_layer // 2) :]
            sampled_rest = rng.sample(rest, k=candidates_per_layer - len(keep))
            candidate_pool = sorted(set(keep + sampled_rest))
        candidates_by_layer[layer_index] = candidate_pool
        selected_counts[layer_index] = min(budget_count, len(candidate_pool))
    total_candidates = sum(len(channels) for channels in candidates_by_layer.values())
    config = {
        "schema_version": f"{SCHEMA_VERSION}_oracle_sampling_config",
        "created_at_utc": shared.utc_now(),
        "frozen_before_scoring": True,
        "seed": seed,
        "sampled_layer_count": len(sampled_layers),
        "sampled_layers": sampled_layers,
        "candidates_per_layer_cap": candidates_per_layer,
        "candidate_count": total_candidates,
        "candidate_selection": "top final-position activation magnitude with deterministic high-pool subsampling",
        "selected_count_by_layer": {str(layer): selected_counts[layer] for layer in sampled_layers},
        "kl_prompt_indices": kl_prompt_indices,
        "kl_positions": kl_positions,
        "kl_position_count": len(kl_positions),
        "coverage_note": "M-KLLOOK is a sampled offline oracle; it is not exhaustive over all channels.",
    }
    return config, candidates_by_layer, selected_counts, layer_names


def capture_channel_patches(model: Any, candidates_by_layer: dict[int, list[int]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    import torch

    hidden_size = int(getattr(model.config, "hidden_size"))
    layers, _origin = shared.discover_transformer_layers(model)
    out: dict[tuple[int, int], list[dict[str, Any]]] = {}
    with torch.no_grad():
        for layer_index, channels in candidates_by_layer.items():
            layer_name, layer = layers[layer_index]
            for channel in channels:
                patches: list[dict[str, Any]] = []
                for module_name, module in layer.named_modules():
                    full_name = f"{layer_name}.{module_name}" if module_name else layer_name
                    weight = getattr(module, "weight", None)
                    if not torch.is_tensor(weight) or weight.ndim not in {2, 3}:
                        continue
                    if weight.ndim == 2:
                        row = weight[channel : channel + 1, :].detach().cpu().clone() if weight.shape[0] == hidden_size else None
                        col = weight[:, channel : channel + 1].detach().cpu().clone() if weight.shape[1] == hidden_size else None
                    else:
                        row = weight[:, channel : channel + 1, :].detach().cpu().clone() if weight.shape[1] == hidden_size else None
                        col = weight[:, :, channel : channel + 1].detach().cpu().clone() if weight.shape[2] == hidden_size else None
                    if row is not None or col is not None:
                        patches.append({"name": full_name, "module": module, "channel": channel, "row": row, "col": col})
                out[(layer_index, channel)] = patches
    return out


def clone_current_patch_slices(patches: dict[tuple[int, int], list[dict[str, Any]]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    import torch

    cloned: dict[tuple[int, int], list[dict[str, Any]]] = {}
    with torch.no_grad():
        for key, rows in patches.items():
            copied: list[dict[str, Any]] = []
            for patch in rows:
                weight = patch["module"].weight
                channel = int(patch["channel"])
                if weight.ndim == 2:
                    row = weight[channel : channel + 1, :].detach().cpu().clone() if patch["row"] is not None else None
                    col = weight[:, channel : channel + 1].detach().cpu().clone() if patch["col"] is not None else None
                else:
                    row = weight[:, channel : channel + 1, :].detach().cpu().clone() if patch["row"] is not None else None
                    col = weight[:, :, channel : channel + 1].detach().cpu().clone() if patch["col"] is not None else None
                copied.append({"module": patch["module"], "channel": channel, "row": row, "col": col})
            cloned[key] = copied
    return cloned


def apply_patch_slices(rows: list[dict[str, Any]]) -> None:
    import torch

    with torch.no_grad():
        for patch in rows:
            weight = patch["module"].weight
            channel = int(patch["channel"])
            if patch["row"] is not None:
                row = patch["row"].to(device=weight.device, dtype=weight.dtype)
                if weight.ndim == 2:
                    weight[channel : channel + 1, :].copy_(row)
                else:
                    weight[:, channel : channel + 1, :].copy_(row)
            if patch["col"] is not None:
                col = patch["col"].to(device=weight.device, dtype=weight.dtype)
                if weight.ndim == 2:
                    weight[:, channel : channel + 1].copy_(col)
                else:
                    weight[:, :, channel : channel + 1].copy_(col)


def collect_log_probs(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    prompts: list[dict[str, Any]],
    target_tokens: dict[int, list[int]],
    positions: list[int],
    use_float16_autocast: bool,
    regime_name: str,
) -> dict[int, dict[int, Any]]:
    import torch

    max_position = max(positions)
    wanted = set(positions)
    autocast_enabled = bool(use_float16_autocast and torch.cuda.is_available())
    cache_dtype = torch.float16 if autocast_enabled else None
    normalize_cache_inputs, output_cache = phase4_runner.make_cache_helpers(model, cache_dtype=cache_dtype)
    previous_fast_paths = phase4_runner.set_autocast_sensitive_fast_paths(False) if autocast_enabled else {}
    out: dict[int, dict[int, Any]] = {}
    try:
        for prompt in prompts:
            prompt_index = int(prompt["index"])
            text = shared.make_prompt_text(str(prompt["prompt"]))
            encoded = tokenizer([text], padding=True, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            target_tensor = torch.tensor(target_tokens[prompt_index][:max_position], dtype=torch.long, device=device)
            out[prompt_index] = {}
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
                    raise RuntimeError(f"model did not return cache for {regime_name}")
                logits = outputs.logits[:, -1, :]
                for decode_position in range(1, max_position + 1):
                    current_target = target_tensor[decode_position - 1]
                    if decode_position in wanted:
                        out[prompt_index][decode_position] = torch.log_softmax(logits.float().squeeze(0), dim=-1).cpu()
                    if decode_position == max_position:
                        break
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)],
                        dim=1,
                    )
                    cache_position = torch.tensor([attention_mask.shape[1] - 1], device=device, dtype=torch.long)
                    model_inputs = model.prepare_inputs_for_generation(
                        input_ids=current_target.reshape(1, 1),
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        use_cache=True,
                    )
                    outputs = model(**normalize_cache_inputs(model_inputs))
                    past_key_values = output_cache(outputs)
                    if past_key_values is None:
                        raise RuntimeError(f"model dropped cache for {regime_name} at {decode_position}")
                    logits = outputs.logits[:, -1, :]
    finally:
        phase4_runner.restore_autocast_sensitive_fast_paths(previous_fast_paths)
    return out


def mean_kl_against_reference(reference: dict[int, dict[int, Any]], candidate: dict[int, dict[int, Any]]) -> float:
    values: list[float] = []
    for prompt_index, by_position in reference.items():
        for position, ref_log_probs in by_position.items():
            cand_log_probs = candidate[prompt_index][position]
            probs = ref_log_probs.exp()
            values.append(float((probs * (ref_log_probs - cand_log_probs)).sum().item()))
    return float(mean(values)) if values else float("nan")


def build_empty_protected_sets(source_protected_sets: dict[str, Any]) -> dict[str, Any]:
    layers: dict[str, Any] = {}
    for layer_key, layer in source_protected_sets["regimes"]["m11b_top10"]["layers"].items():
        layers[layer_key] = {
            "layer_name": layer["layer_name"],
            "channel_count": layer["channel_count"],
            "protected_count": 0,
            "protected_channels": [],
            "source": "no_protection_base_for_kl_lookahead",
        }
    return {
        "schema_version": f"{SCHEMA_VERSION}_empty_protected_sets",
        "created_at_utc": shared.utc_now(),
        "regimes": {"empty": {"layers": layers}},
    }


def build_oracle_sets(
    *,
    source_protected_sets: dict[str, Any],
    candidates_by_layer: dict[int, list[int]],
    selected_counts: dict[int, int],
    candidate_rows: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    by_layer_delta: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        by_layer_delta[int(row["layer_index"])].append(row)
    m11b_layers = source_protected_sets["regimes"]["m11b_top10"]["layers"]
    static_layers = source_protected_sets["regimes"]["static_1pct"]["layers"]
    oracle_layers: dict[str, Any] = {}
    random_layers: dict[str, Any] = {}
    for layer_key, base_layer in m11b_layers.items():
        layer_index = int(layer_key)
        base_channels = [int(ch) for ch in base_layer["protected_channels"]]
        target_count = int(base_layer["protected_count"])
        if layer_index in candidates_by_layer:
            ranked = sorted(by_layer_delta[layer_index], key=lambda row: (-float(row["delta_kl"]), int(row["channel_index"])))
            chosen = [int(row["channel_index"]) for row in ranked[: selected_counts[layer_index]]]
            candidate_pool = candidates_by_layer[layer_index]
            random_chosen = rng.sample(candidate_pool, k=min(selected_counts[layer_index], len(candidate_pool)))
            oracle = []
            for channel in [*chosen, *base_channels]:
                if channel not in oracle:
                    oracle.append(channel)
                if len(oracle) == target_count:
                    break
            random_set = []
            for channel in [*random_chosen, *base_channels]:
                if channel not in random_set:
                    random_set.append(channel)
                if len(random_set) == target_count:
                    break
        else:
            oracle = base_channels
            random_set = base_channels
        oracle_layers[layer_key] = {
            **base_layer,
            "protected_channels": sorted(oracle),
            "protected_count": len(oracle),
            "source": "sampled_forward_kl_lookahead",
        }
        random_layers[layer_key] = {
            **base_layer,
            "protected_channels": sorted(random_set),
            "protected_count": len(random_set),
            "source": "random_matched_sampled_candidate_control",
        }
    return {
        "schema_version": f"{SCHEMA_VERSION}_protected_sets",
        "created_at_utc": shared.utc_now(),
        "selection_basis": "sampled forward-KL lookahead candidates blended into M11b top-10 budget",
        "regimes": {
            "static_1pct": {"layers": static_layers},
            "m11b_top10": {"layers": m11b_layers},
            "mkllook_top10": {"layers": oracle_layers},
            "random_top10": {"layers": random_layers},
        },
    }


def score_quantized_regime(
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
    release_model_memory(model, tokenizer, device)
    return scores, excluded


def summarize_recovery(rows: list[dict[str, Any]], regime: str) -> dict[str, Any]:
    included = [row for row in rows if not bool(row.get("no_recoverable_static_gap"))]
    values = [float(row["recoveries"][regime]) for row in included]
    return {
        "median_recovery": float(median(values)) if values else None,
        "mean_recovery": float(mean(values)) if values else None,
        "bootstrap_ci95": checker.bootstrap_median(values) if hasattr(checker, "bootstrap_median") else bootstrap_median(values),
        "included_trace_count": len(values),
        "total_trace_count": len(rows),
        "per_trace_recovery_included": values,
        "no_recoverable_static_gap_count": len(rows) - len(values),
        "no_recoverable_static_gap_fraction": (len(rows) - len(values)) / len(rows) if rows else 0.0,
    }


def bootstrap_median(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"ci95_low": None, "ci95_high": None}
    rng = random.Random(checker.BOOTSTRAP_SEED)
    boot = [float(median([values[rng.randrange(len(values))] for _ in values])) for _ in range(checker.BOOTSTRAP_SAMPLES)]
    boot.sort()
    return {"ci95_low": boot[int(0.025 * (len(boot) - 1))], "ci95_high": boot[int(0.975 * (len(boot) - 1))]}


def build_metrics(
    *,
    run_dir: Path,
    model_key: str,
    prompt_manifest: dict[str, Any],
    model_provenance: dict[str, Any],
    per_trace_rows: list[dict[str, Any]],
    sampling_config: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    summaries = {regime: summarize_recovery(per_trace_rows, regime) for regime in checker.RECOVERY_REGIMES}
    sampled_deltas = [float(row["delta_kl"]) for row in candidate_rows]
    metrics = {
        "schema_version": f"{SCHEMA_VERSION}_metrics",
        "created_at_utc": shared.utc_now(),
        "preregistration": str(checker.PREREG_PATH.relative_to(ROOT)),
        "preregistration_sha256": shared.file_sha256(checker.PREREG_PATH),
        "model_key": model_key,
        "model_id": model_provenance.get("model_id"),
        "model_snapshot_commit": model_provenance.get("hf_snapshot_commit"),
        "prompt_sha256": prompt_manifest.get("prompt_sha256"),
        "trace_count": checker.TRACE_COUNT,
        "effective_trace_count": len(per_trace_rows),
        "scoring_position": checker.SCORING_POSITION,
        "scoring_window_tokens": checker.SCORING_WINDOW_TOKENS,
        "metric_name": "positive-static-1pct-gap per-trace recovery",
        "metric_formula": "1 - (ppl_regime - ppl_BF16) / (ppl_static_1pct - ppl_BF16)",
        "oracle_delta_metric": "mean sampled forward KL improvement over base W4A16",
        "results_by_regime": summaries,
        "sampling_summary": {
            "candidate_count": sampling_config["candidate_count"],
            "sampled_layer_count": sampling_config["sampled_layer_count"],
            "kl_prompt_indices": sampling_config["kl_prompt_indices"],
            "kl_position_count": sampling_config["kl_position_count"],
            "mean_delta_kl": float(mean(sampled_deltas)) if sampled_deltas else None,
            "max_delta_kl": float(max(sampled_deltas)) if sampled_deltas else None,
            "min_delta_kl": float(min(sampled_deltas)) if sampled_deltas else None,
        },
        "thresholds": checker.THRESHOLDS,
        "artifacts": {"run_dir": str(run_dir)},
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
            "m11b_top10": summaries["m11b_top10"],
            "random_top10": summaries["random_top10"],
            "static_1pct": {"median_recovery": 0.0},
        },
    }
    return metrics, bootstrap, controls


def prompt_payload_sha256(prompts: list[dict[str, Any]]) -> str:
    ordered = sorted(prompts, key=lambda row: int(row["index"]))
    return shared.bytes_sha256("".join(str(row["prompt"]) for row in ordered).encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    shared.SCHEMA_VERSION = SCHEMA_VERSION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"om_phase9_mkllook_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--results-dir", type=Path, default=checker.RESULTS_DIR)
    parser.add_argument("--model-key", choices=sorted(checker.MODEL_SPECS), required=True)
    parser.add_argument("--source-run-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--sampled-layer-count", type=int, default=2)
    parser.add_argument("--candidates-per-layer", type=int, default=4)
    parser.add_argument("--kl-prompt-count", type=int, default=1)
    parser.add_argument("--kl-token-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=checker.BOOTSTRAP_SEED)
    args = parser.parse_args(argv)

    source_dir = (args.source_run_dir or DEFAULT_SOURCES[args.model_key]).resolve()
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

    def mkllook_excepthook(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        write_failure_packet(run_dir, run_events_path, exc)
        previous_excepthook(exc_type, exc, tb)

    sys.excepthook = mkllook_excepthook
    run_events_path.write_text(json.dumps({"created_at_utc": shared.utc_now(), "event": "run_started"}, sort_keys=True) + "\n", encoding="utf-8")
    random.seed(args.seed)

    try:
        if not source_dir.is_dir():
            raise FileNotFoundError(f"source run dir missing: {source_dir}")
        source_prompt_manifest = json.loads((source_dir / "prompt_manifest.json").read_text(encoding="utf-8"))
        prompts = [row for row in source_prompt_manifest["prompts"] if int(row["index"]) < checker.TRACE_COUNT]
        prompt_indices = {int(row["index"]) for row in prompts}
        if prompt_indices != set(range(checker.TRACE_COUNT)):
            raise RuntimeError("source prompt manifest does not contain deterministic indices 0-11")
        prompt_manifest = {
            **source_prompt_manifest,
            "schema_version": f"{SCHEMA_VERSION}_prompt_manifest",
            "created_at_utc": shared.utc_now(),
            "prompt_count": len(prompts),
            "prompts": prompts,
            "prompt_sha256": prompt_payload_sha256(prompts),
        }
        environment = shared.build_environment(schema_version=SCHEMA_VERSION)
        model_id = checker.MODEL_SPECS[args.model_key]["model_id"]
        model_provenance = m2_runner.resolve_model_snapshot_light(model_id)
        model_provenance["schema_version"] = f"{SCHEMA_VERSION}_model_provenance"
        if model_provenance.get("hf_snapshot_commit") != checker.MODEL_SPECS[args.model_key]["snapshot"]:
            raise RuntimeError("model snapshot missing or mismatch")

        shared.write_json(run_dir / "prompt_manifest.json", prompt_manifest)
        shared.write_json(run_dir / "environment.json", environment)
        phase4_runner.write_environment_text(run_dir / "environment.txt", environment)
        shared.write_json(run_dir / "model_provenance.json", model_provenance)
        shared.write_json(
            run_dir / "command_metadata.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_command",
                "created_at_utc": shared.utc_now(),
                "argv": sys.argv if argv is None else ["run_om_phase9_mkllook.py", *argv],
                "cwd": str(Path.cwd()),
                "run_dir": str(run_dir),
                "model_key": args.model_key,
                "source_run_dir": str(source_dir),
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
                "scheme": "symmetric_per_output_channel_int4",
                "activation_dtype": "float16",
                "protected_channel_dtype": "bfloat16",
                "implementation_note": "M-KLLOOK uses sampled forward-KL channel deltas to modify M11b top-10 sets before standard W4A16 endpoint scoring.",
            },
        )

        source_artifacts = copy_required_source_artifacts(source_dir, run_dir)
        all_scores = load_source_scores(source_dir, run_dir, prompt_indices, source_artifacts)
        target_tokens = phase4_runner.load_trace_tokens(run_dir / "bf16_traces.jsonl.gz")
        source_protected_sets = json.loads((source_dir / "protected_sets.json").read_text(encoding="utf-8"))
        source_artifacts["artifacts"].append(
            {"name": "protected_sets", "path": str(source_dir / "protected_sets.json"), "sha256": shared.file_sha256(source_dir / "protected_sets.json")}
        )

        kl_positions = scoring_positions(args.kl_token_count)
        kl_prompts = prompts[: args.kl_prompt_count]
        kl_prompt_indices = [int(row["index"]) for row in kl_prompts]
        sampling_config, candidates_by_layer, selected_counts, _layer_names = build_candidate_plan(
            activation_rows=list(shared.iter_activation_rows(run_dir / "activation_magnitudes.jsonl.gz")),
            source_protected_sets=source_protected_sets,
            sampled_layer_count=args.sampled_layer_count,
            candidates_per_layer=args.candidates_per_layer,
            seed=args.seed,
            kl_positions=kl_positions,
            kl_prompt_indices=kl_prompt_indices,
        )
        shared.write_json(run_dir / "oracle_sampling_config.json", sampling_config)

        model, tokenizer, device = shared.load_model_and_tokenizer(model_provenance, dtype_name=args.dtype, device_name=args.device)
        bf16_patches = capture_channel_patches(model, candidates_by_layer)
        reference_log_probs = collect_log_probs(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompts=kl_prompts,
            target_tokens=target_tokens,
            positions=kl_positions,
            use_float16_autocast=False,
            regime_name="bf16_reference",
        )
        empty_sets = build_empty_protected_sets(source_protected_sets)
        phase4_runner.apply_quantization(model, empty_sets, "empty")
        quant_patches = clone_current_patch_slices(bf16_patches)
        base_log_probs = collect_log_probs(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompts=kl_prompts,
            target_tokens=target_tokens,
            positions=kl_positions,
            use_float16_autocast=True,
            regime_name="base_w4a16",
        )
        base_kl = mean_kl_against_reference(reference_log_probs, base_log_probs)
        candidate_rows: list[dict[str, Any]] = []
        for layer_index in sorted(candidates_by_layer):
            for channel_index in candidates_by_layer[layer_index]:
                key = (layer_index, channel_index)
                apply_patch_slices(bf16_patches[key])
                patched_log_probs = collect_log_probs(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    prompts=kl_prompts,
                    target_tokens=target_tokens,
                    positions=kl_positions,
                    use_float16_autocast=True,
                    regime_name=f"candidate_l{layer_index}_c{channel_index}",
                )
                patched_kl = mean_kl_against_reference(reference_log_probs, patched_log_probs)
                apply_patch_slices(quant_patches[key])
                candidate_rows.append(
                    {
                        "layer_index": layer_index,
                        "channel_index": channel_index,
                        "kl_without": base_kl,
                        "kl_with": patched_kl,
                        "delta_kl": base_kl - patched_kl,
                        "kl_prompt_indices": kl_prompt_indices,
                        "kl_positions": kl_positions,
                        "token_count": len(kl_prompt_indices) * len(kl_positions),
                        "selected_by_oracle": False,
                    }
                )
                run_events_path.open("a", encoding="utf-8").write(
                    json.dumps(
                        {
                            "created_at_utc": shared.utc_now(),
                            "event": "completed_candidate_kl",
                            "layer_index": layer_index,
                            "channel_index": channel_index,
                            "delta_kl": base_kl - patched_kl,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        release_model_memory(model, tokenizer, device)

        for layer_index, count in selected_counts.items():
            ranked = sorted(
                [row for row in candidate_rows if int(row["layer_index"]) == layer_index],
                key=lambda row: (-float(row["delta_kl"]), int(row["channel_index"])),
            )
            selected = {int(row["channel_index"]) for row in ranked[:count]}
            for row in candidate_rows:
                if int(row["layer_index"]) == layer_index and int(row["channel_index"]) in selected:
                    row["selected_by_oracle"] = True
        shared.write_json(
            run_dir / "candidate_deltas.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_candidate_deltas",
                "created_at_utc": shared.utc_now(),
                "base_mean_forward_kl": base_kl,
                "candidates": candidate_rows,
            },
        )

        protected_sets = build_oracle_sets(
            source_protected_sets=source_protected_sets,
            candidates_by_layer=candidates_by_layer,
            selected_counts=selected_counts,
            candidate_rows=candidate_rows,
            seed=args.seed,
        )
        shared.write_json(run_dir / "protected_sets.json", protected_sets)

        excluded_by_regime: dict[str, Any] = {
            "static_1pct": {"regime": "static_1pct", "reused_score_cache": str(source_dir)},
            "m11b_top10": {"regime": "m11b_top10", "reused_score_cache": str(source_dir)},
        }
        for regime in ["mkllook_top10", "random_top10"]:
            scores, excluded = score_quantized_regime(
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
            all_scores[regime] = scores
            excluded_by_regime[regime] = excluded
            write_score_cache(run_dir, regime, scores)
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
                regime: None if no_gap else 1.0 - (perplexities[regime] - perplexities["bf16"]) / static_gap
                for regime in checker.RECOVERY_REGIMES
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
            model_key=args.model_key,
            prompt_manifest=prompt_manifest,
            model_provenance=model_provenance,
            per_trace_rows=per_trace_rows,
            sampling_config=sampling_config,
            candidate_rows=candidate_rows,
        )
        shared.write_json(run_dir / "metrics.json", metrics)
        shared.write_json(run_dir / "bootstrap_ci.json", bootstrap)
        shared.write_json(run_dir / "control_metrics.json", controls)
        run_events_path.open("a", encoding="utf-8").write(json.dumps({"created_at_utc": shared.utc_now(), "event": "run_completed"}, sort_keys=True) + "\n")
        shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
        result = checker.evaluate(run_dir)
        print(json.dumps({"run_dir": str(run_dir), "checker_decision": result["decision"], "details": result.get("details", {})}, indent=2, sort_keys=True))
        sys.excepthook = previous_excepthook
        return 0 if result["decision"] != checker.FAIL_INFRA else 1
    except BaseException as exc:
        write_failure_packet(run_dir, run_events_path, exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
