#!/usr/bin/env python3
"""Run Stage 1 E1 cross-model KL and FFT diagnostics."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import shutil
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.outlier_migrate.phase4 import run_om_phase4_intervention as phase4_runner
from experimental.outlier_migrate.phase9 import check_om_stage1_e1_cross_model_kl_fft as checker
from experimental.shared import run_phase0_branch as shared


SCHEMA_VERSION = checker.SCHEMA_VERSION
DEFAULT_RESULTS_DIR = checker.RESULTS_DIR


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


def load_prompts(source_run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prompt_manifest = json.loads((source_run_dir / "prompt_manifest.json").read_text(encoding="utf-8"))
    prompts = [row for row in prompt_manifest.get("prompts", []) if int(row["index"]) < checker.TRACE_COUNT]
    if [int(row["index"]) for row in prompts] != list(range(checker.TRACE_COUNT)):
        raise RuntimeError(f"{source_run_dir}: prompt indices are not deterministic 0-11")
    return {**prompt_manifest, "prompt_count": checker.TRACE_COUNT, "prompts": prompts}, prompts


def top_channels(values: np.ndarray, count: int) -> list[int]:
    order = np.lexsort((np.arange(values.shape[0]), -values))
    return [int(value) for value in order[:count]]


def select_m11_channels(
    scores: np.ndarray,
    *,
    min_count: int,
    cap_count: int,
    protected_durations: dict[int, int],
) -> list[int]:
    selected = [int(channel) for channel, score in enumerate(scores.tolist()) if float(score) > 0.5]
    if len(selected) < min_count:
        selected = top_channels(scores, min_count)
    if len(selected) > cap_count:
        selected = sorted(
            selected,
            key=lambda channel: (
                int(protected_durations.get(channel, 0)),
                -float(scores[channel]),
                -channel,
            ),
        )[:cap_count]
    return sorted(selected)


def write_activation_means_npz(
    path: Path,
    *,
    means: dict[int, dict[int, np.ndarray]],
    layer_names: dict[int, str],
    positions: list[int],
) -> None:
    arrays: dict[str, np.ndarray] = {"positions": np.asarray(positions, dtype=np.int32)}
    for layer in sorted(means):
        arrays[f"layer_{layer}"] = np.stack([means[layer][position] for position in positions]).astype(np.float32)
    arrays["layer_indices"] = np.asarray(sorted(means), dtype=np.int32)
    arrays["layer_names_json"] = np.frombuffer(
        json.dumps({str(key): value for key, value in sorted(layer_names.items())}).encode("utf-8"),
        dtype=np.uint8,
    )
    np.savez_compressed(path, **arrays)


def build_protected_sets(means: dict[int, dict[int, np.ndarray]], layer_names: dict[int, str]) -> dict[str, Any]:
    update_positions = list(range(100, 10001, 100))
    missing: list[str] = []
    for layer, by_position in means.items():
        absent = [position for position in update_positions if position not in by_position]
        if absent:
            missing.append(f"layer {layer}: missing update positions {absent[:5]}")
    if missing:
        raise RuntimeError("E1 protection construction requires dense 100-token positions to 10000; " + "; ".join(missing[:4]))

    regimes: dict[str, dict[str, Any]] = {regime: {} for regime in checker.QUANTIZED_REGIMES}
    for layer in sorted(means):
        values_100 = means[layer][100]
        values_10000 = means[layer][10000]
        channel_count = int(values_100.shape[0])
        top1_count = max(1, math.ceil(channel_count * 0.01))
        cap_count = max(top1_count, math.ceil(channel_count * 0.03))
        static_top1 = sorted(top_channels(values_100, top1_count))
        decdec_top1 = sorted(top_channels(values_10000, top1_count))

        scores = np.zeros(channel_count, dtype=np.float64)
        scores[static_top1] = 1.0
        selected = static_top1
        durations = {channel: 1 for channel in static_top1}
        for position in update_positions:
            if position == 100:
                continue
            indicator = set(top_channels(means[layer][position], top1_count))
            scores = np.asarray(
                [0.5 * (1.0 if channel in indicator else 0.0) + 0.5 * float(scores[channel]) for channel in range(channel_count)],
                dtype=np.float64,
            )
            selected = select_m11_channels(
                scores,
                min_count=top1_count,
                cap_count=cap_count,
                protected_durations=durations,
            )
            selected_set = set(selected)
            for channel in range(channel_count):
                durations[channel] = int(durations.get(channel, 0)) + 1 if channel in selected_set else 0

        common = {
            "layer_name": layer_names[layer],
            "channel_count": channel_count,
            "requested_top_k": top1_count,
        }
        regimes["static_1pct"][str(layer)] = {
            **common,
            "protected_count": len(static_top1),
            "protected_channels": static_top1,
            "source": "position_100_top1",
        }
        regimes["decdec_reactive_top1_proxy"][str(layer)] = {
            **common,
            "protected_count": len(decdec_top1),
            "protected_channels": decdec_top1,
            "source": "position_10000_endpoint_oracle_top1",
        }
        regimes["m11_alpha_0_5"][str(layer)] = {
            **common,
            "cap_count": cap_count,
            "protected_count": len(selected),
            "protected_channels": selected,
            "source": "ema_alpha_0_5_final_snapshot_position_10000",
        }

    return {
        "schema_version": f"{SCHEMA_VERSION}_protected_sets",
        "created_at_utc": shared.utc_now(),
        "selection_basis": "BF16 mean absolute layer-output activation over deterministic AIME-2025 traces 0-11",
        "tie_break": "lower channel index",
        "regimes": {
            "static_1pct": {"kind": "single_position", "positions": [100], "fraction": 0.01, "layers": regimes["static_1pct"]},
            "decdec_reactive_top1_proxy": {
                "kind": "endpoint_oracle",
                "positions": [10000],
                "fraction": 0.01,
                "layers": regimes["decdec_reactive_top1_proxy"],
            },
            "m11_alpha_0_5": {
                "kind": "ema_final_snapshot",
                "positions": list(range(100, 10001, 100)),
                "alpha": 0.5,
                "absolute_cap_fraction": 0.03,
                "layers": regimes["m11_alpha_0_5"],
            },
        },
    }


def autocorr_length(values: np.ndarray, step_tokens: int) -> int | None:
    centered = values - values.mean()
    denom = float(np.dot(centered, centered))
    if denom <= 0.0:
        return None
    threshold = 1.0 / math.e
    for lag in range(1, len(centered)):
        corr = float(np.dot(centered[:-lag], centered[lag:]) / denom)
        if corr < threshold:
            return lag * step_tokens
    return None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def write_spectral_summary(
    model_dir: Path,
    means: dict[int, dict[int, np.ndarray]],
    layer_names: dict[int, str],
    positions: list[int],
) -> dict[str, Any]:
    step_tokens = positions[1] - positions[0]
    low_frequency_fractions: list[float] = []
    spectral_entropies: list[float] = []
    autocorr_lengths: list[float] = []
    sampled_channels = 0
    layer_summaries: dict[str, Any] = {}
    for layer in sorted(means):
        if any(position not in means[layer] for position in positions):
            continue
        matrix = np.stack([means[layer][position] for position in positions])
        channel_count = int(matrix.shape[1])
        top1_count = max(1, math.ceil(channel_count * 0.01))
        selected = top_channels(matrix[0], top1_count)
        layer_entropies: list[float] = []
        layer_autocorrs: list[float] = []
        for channel in selected:
            trajectory = matrix[:, int(channel)]
            centered = trajectory - trajectory.mean()
            spectrum = np.fft.rfft(centered)
            power = np.abs(spectrum) ** 2
            non_dc = power[1:]
            if float(non_dc.sum()) <= 0.0:
                continue
            low_bins = max(1, math.ceil(0.10 * len(non_dc)))
            probs = non_dc / non_dc.sum()
            entropy = float(-(probs * np.log2(probs + 1e-30)).sum() / math.log2(len(probs)))
            low_fraction = float(non_dc[:low_bins].sum() / non_dc.sum())
            acl = autocorr_length(trajectory, step_tokens)
            low_frequency_fractions.append(low_fraction)
            spectral_entropies.append(entropy)
            layer_entropies.append(entropy)
            if acl is not None:
                autocorr_lengths.append(float(acl))
                layer_autocorrs.append(float(acl))
            sampled_channels += 1
        layer_summaries[str(layer)] = {
            "layer_name": layer_names[layer],
            "channel_count": channel_count,
            "sampled_top1_channels": len(selected),
            "median_normalized_spectral_entropy": percentile(layer_entropies, 50),
            "median_autocorrelation_length_tokens": percentile(layer_autocorrs, 50),
        }

    summary = {
        "schema_version": f"{SCHEMA_VERSION}_spectral_summary",
        "created_at_utc": shared.utc_now(),
        "positions": positions,
        "position_count": len(positions),
        "step_tokens": step_tokens,
        "layer_count": len(layer_summaries),
        "sampled_top1_channels": sampled_channels,
        "median_low_frequency_power_fraction_first_10pct_bins": percentile(low_frequency_fractions, 50),
        "median_normalized_spectral_entropy": percentile(spectral_entropies, 50),
        "normalized_spectral_entropy_iqr": [percentile(spectral_entropies, 25), percentile(spectral_entropies, 75)],
        "median_autocorrelation_length_tokens": percentile(autocorr_lengths, 50),
        "autocorrelation_length_iqr_tokens": [percentile(autocorr_lengths, 25), percentile(autocorr_lengths, 75)],
        "layer_summaries": layer_summaries,
    }
    shared.write_json(model_dir / "spectral_summary.json", summary)
    return summary


def fit_line(xs: list[float], ys: list[float], transform: str) -> dict[str, float]:
    if transform == "linear":
        tx = xs
    elif transform == "sqrt":
        tx = [math.sqrt(x) for x in xs]
    else:
        power = float(transform)
        tx = [x**power for x in xs]
    x_bar = mean(tx)
    y_bar = mean(ys)
    denom = sum((x - x_bar) ** 2 for x in tx)
    slope = 0.0 if denom == 0.0 else sum((x - x_bar) * (y - y_bar) for x, y in zip(tx, ys)) / denom
    intercept = y_bar - slope * x_bar
    rss = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(tx, ys))
    return {"intercept": float(intercept), "slope": float(slope), "rss": float(rss)}


def fit_growth_models(summary_by_regime: dict[str, list[dict[str, float]]]) -> dict[str, Any]:
    regime_fits: dict[str, Any] = {}
    for regime, rows in summary_by_regime.items():
        if regime == "bf16_reference":
            continue
        xs = [float(row["decode_position"]) for row in rows]
        ys = [float(row["mean_kl"]) for row in rows]
        candidates = {
            "linear": fit_line(xs, ys, "linear"),
            "sublinear_sqrt": fit_line(xs, ys, "sqrt"),
        }
        for power in [1.1, 1.25, 1.5, 2.0]:
            candidates[f"superlinear_power_{power}"] = fit_line(xs, ys, str(power))
        best_name = min(candidates, key=lambda name: candidates[name]["rss"])
        centered = [value - mean(ys) for value in ys]
        denom = sum(value * value for value in centered[:-1])
        ar1 = None if denom == 0.0 else sum(a * b for a, b in zip(centered[:-1], centered[1:])) / denom
        best_class = "sublinear" if best_name == "sublinear_sqrt" else "linear" if best_name == "linear" else "superlinear"
        if max(ys) < 1e-8:
            best_class = "flat"
        regime_fits[regime] = {
            "best_fit": best_name,
            "best_fit_class": best_class,
            "fits": candidates,
            "ar1_decay_estimate": None if ar1 is None else float(ar1),
            "mean_kl_first_position": ys[0],
            "mean_kl_last_position": ys[-1],
            "mean_kl_median": float(median(ys)),
        }
    return {
        "schema_version": f"{SCHEMA_VERSION}_growth_model_fits",
        "created_at_utc": shared.utc_now(),
        "fit_note": "Least-squares fits over trace-mean KL curve; candidate set fixed before data inspection.",
        "regime_fits": regime_fits,
    }


def write_kl_summary(model_dir: Path, positions: list[int], rows_by_regime_position: dict[str, dict[int, list[float]]]) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_by_regime: dict[str, list[dict[str, float]]] = {}
    regime_summary: dict[str, Any] = {}
    for regime in checker.KL_REGIMES:
        curve: list[dict[str, float]] = []
        all_values: list[float] = []
        for position in positions:
            values = rows_by_regime_position[regime][position]
            curve.append(
                {
                    "decode_position": int(position),
                    "mean_kl": float(mean(values)),
                    "median_kl": float(median(values)),
                    "max_kl": float(max(values)),
                }
            )
            all_values.extend(values)
        summary_by_regime[regime] = curve
        regime_summary[regime] = {
            "mean_kl_all_positions_traces": float(mean(all_values)) if all_values else None,
            "median_kl_all_positions_traces": float(median(all_values)) if all_values else None,
            "max_kl_all_positions_traces": float(max(all_values)) if all_values else None,
            "position_count": len(positions),
            "trace_count": checker.TRACE_COUNT,
        }
    summary = {
        "schema_version": f"{SCHEMA_VERSION}_kl_summary",
        "created_at_utc": shared.utc_now(),
        "regime_summary": regime_summary,
        "trace_mean_curves": summary_by_regime,
    }
    fits = fit_growth_models(summary_by_regime)
    shared.write_json(model_dir / "kl_summary.json", summary)
    shared.write_json(model_dir / "growth_model_fits.json", fits)
    return summary, fits


def collect_bf16_reference_trace_logprobs_and_activations(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    prompts: list[dict[str, Any]],
    kl_positions: list[int],
    spectral_positions: list[int],
    max_new_tokens: int,
    trace_path: Path,
    activation_npz_path: Path,
    tmp_dir: Path,
    run_events_path: Path,
) -> dict[str, Any]:
    import torch

    layers, layer_origin = shared.discover_transformer_layers(model)
    normalize_cache_inputs, output_cache = phase4_runner.make_cache_helpers(model)
    wanted_kl = set(kl_positions)
    wanted_activation = set(spectral_positions)
    state: dict[str, Any] = {"capture_enabled": False, "records_by_layer": {}}
    handles = []

    def make_hook(layer_index: int, layer_name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if not state["capture_enabled"]:
                return
            tensor = shared.tensor_from_hook_output(output)
            if not torch.is_tensor(tensor) or tensor.ndim < 2:
                return
            state["records_by_layer"][layer_index] = (layer_name, tensor[:, -1, :].detach().abs().to(torch.float32).cpu())

        return hook

    for layer_index, (layer_name, layer) in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(layer_index, layer_name)))

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    activation_npz_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    activation_sums: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    activation_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    prompt_events: list[dict[str, Any]] = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    try:
        with gzip.open(trace_path, "wt", encoding="utf-8") as trace_handle, torch.inference_mode():
            for prompt in prompts:
                prompt_index = int(prompt["index"])
                text = shared.make_prompt_text(str(prompt["prompt"]))
                encoded = tokenizer([text], padding=True, return_tensors="pt")
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
                    raise RuntimeError("model did not return cache for BF16 E1 reference collection")
                logits = outputs.logits[:, -1, :]
                generated: list[int] = []
                log_probs_by_position: dict[int, Any] = {}
                first_eos = None
                for decode_position in range(1, max_new_tokens + 1):
                    if decode_position in wanted_kl:
                        log_probs_by_position[decode_position] = torch.log_softmax(logits.float().squeeze(0), dim=-1).cpu()
                    next_token = torch.argmax(logits, dim=-1)
                    token_value = int(next_token.item())
                    generated.append(token_value)
                    if eos_token_id is not None and token_value == int(eos_token_id) and first_eos is None:
                        first_eos = decode_position
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)],
                        dim=1,
                    )
                    state["records_by_layer"] = {}
                    state["capture_enabled"] = decode_position in wanted_activation
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
                    if state["capture_enabled"]:
                        missing = sorted(set(range(len(layers))).difference(state["records_by_layer"].keys()))
                        if missing:
                            raise RuntimeError(f"missing activation hooks at decode position {decode_position}: {missing[:8]}")
                        for layer_index in range(len(layers)):
                            _layer_name, magnitudes = state["records_by_layer"][layer_index]
                            vector = magnitudes[0].numpy().astype(np.float64, copy=False)
                            if decode_position not in activation_sums[layer_index]:
                                activation_sums[layer_index][decode_position] = np.zeros_like(vector)
                            activation_sums[layer_index][decode_position] += vector
                            activation_counts[layer_index][decode_position] += 1
                    logits = outputs.logits[:, -1, :]
                    if decode_position % 1000 == 0:
                        run_events_path.open("a", encoding="utf-8").write(
                            json.dumps(
                                {
                                    "created_at_utc": shared.utc_now(),
                                    "event": "bf16_reference_trace_progress",
                                    "prompt_index": prompt_index,
                                    "decode_position": decode_position,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                trace_handle.write(
                    json.dumps(
                        {
                            "schema_version": f"{SCHEMA_VERSION}_bf16_trace",
                            "prompt_index": prompt_index,
                            "prompt_id": prompt["prompt_id"],
                            "generated_token_count": len(generated),
                            "first_eos_decode_position": first_eos,
                            "token_ids": generated,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                torch.save({position: tensor.to(torch.float32) for position, tensor in log_probs_by_position.items()}, tmp_dir / f"bf16_trace_{prompt_index}.pt")
                prompt_events.append(
                    {
                        "prompt_index": prompt_index,
                        "prompt_id": prompt["prompt_id"],
                        "input_token_count": int(encoded["attention_mask"][0].sum().item()),
                        "generated_token_count": len(generated),
                        "first_eos_decode_position": first_eos,
                    }
                )
                run_events_path.open("a", encoding="utf-8").write(
                    json.dumps(
                        {
                            "created_at_utc": shared.utc_now(),
                            "event": "completed_bf16_reference_trace",
                            "prompt_index": prompt_index,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    finally:
        for handle in handles:
            handle.remove()
    means: dict[int, dict[int, np.ndarray]] = {}
    for layer, by_position in activation_sums.items():
        means[layer] = {}
        for position, vector in by_position.items():
            means[layer][position] = vector / float(activation_counts[layer][position])
    layer_names = {index: name for index, (name, _layer) in enumerate(layers)}
    write_activation_means_npz(activation_npz_path, means=means, layer_names=layer_names, positions=spectral_positions)
    return {
        "activation_means": means,
        "layer_names": layer_names,
        "trace_manifest": {
            "schema_version": f"{SCHEMA_VERSION}_bf16_trace_manifest",
            "created_at_utc": shared.utc_now(),
            "artifact": "bf16_traces.jsonl.gz",
            "artifact_sha256": shared.file_sha256(trace_path),
            "trace_count": len(prompts),
            "max_new_tokens": max_new_tokens,
            "decode_policy": "manual greedy BF16 decode; EOS recorded but ignored for fixed-length traces",
            "prompt_events": prompt_events,
        },
        "activation_manifest": {
            "schema_version": f"{SCHEMA_VERSION}_activation_manifest",
            "created_at_utc": shared.utc_now(),
            "artifact": "activation_means.npz",
            "artifact_sha256": shared.file_sha256(activation_npz_path),
            "trace_count": len(prompts),
            "positions": spectral_positions,
            "layer_count": len(layers),
            "layer_origin": layer_origin,
            "layer_names": [name for name, _layer in layers],
            "aggregation": "mean absolute layer-output activation over deterministic traces 0-11",
            "capture_semantics": {
                "module": "transformer_layer_forward_output",
                "token": "generated token at the current decode position",
                "value": "absolute activation magnitude per output channel",
            },
        },
    }


def collect_quantized_kl(
    *,
    model_provenance: dict[str, Any],
    protected_sets: dict[str, Any],
    regime: str,
    prompts: list[dict[str, Any]],
    target_tokens: dict[int, list[int]],
    positions: list[int],
    max_new_tokens: int,
    dtype_name: str,
    device_name: str,
    tmp_dir: Path,
    kl_handle: Any,
    rows_by_regime_position: dict[str, dict[int, list[float]]],
    model_dir: Path,
    run_events_path: Path,
) -> None:
    import torch

    model, tokenizer, device = shared.load_model_and_tokenizer(model_provenance, dtype_name=dtype_name, device_name=device_name)
    excluded = phase4_runner.apply_quantization(model, protected_sets, regime)
    shared.write_json(model_dir / f"excluded_tensors_{regime}.json", excluded)
    try:
        for prompt in prompts:
            prompt_index = int(prompt["index"])
            bf16_log_probs = torch.load(tmp_dir / f"bf16_trace_{prompt_index}.pt", map_location="cpu")
            q_log_probs = phase4_runner_kl_collect(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                target_tokens=target_tokens[prompt_index],
                positions=positions,
                max_new_tokens=max_new_tokens,
                use_float16_autocast=True,
                regime_name=regime,
            )
            for position in positions:
                ref_logp = bf16_log_probs[position]
                q_logp = q_log_probs[position].to(ref_logp.dtype)
                probs = ref_logp.exp()
                kl_value = float(torch.sum(probs * (ref_logp - q_logp)).item())
                if kl_value < 0.0 and kl_value > -1e-6:
                    kl_value = 0.0
                row = {
                    "schema_version": f"{SCHEMA_VERSION}_kl_row",
                    "prompt_index": prompt_index,
                    "prompt_id": prompt["prompt_id"],
                    "regime": regime,
                    "decode_position": position,
                    "kl_bf16_q": kl_value,
                }
                kl_handle.write(json.dumps(row, sort_keys=True) + "\n")
                rows_by_regime_position[regime][position].append(kl_value)
            run_events_path.open("a", encoding="utf-8").write(
                json.dumps(
                    {
                        "created_at_utc": shared.utc_now(),
                        "event": "completed_e1_quantized_kl_trace",
                        "model_key": model_dir.name,
                        "regime": regime,
                        "prompt_index": prompt_index,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    finally:
        del model, tokenizer, device
        release_model_memory()


def phase4_runner_kl_collect(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    prompt: dict[str, Any],
    target_tokens: list[int],
    positions: list[int],
    max_new_tokens: int,
    use_float16_autocast: bool,
    regime_name: str,
) -> dict[int, Any]:
    import torch

    autocast_enabled = bool(use_float16_autocast and torch.cuda.is_available())
    cache_dtype = torch.float16 if autocast_enabled else None
    normalize_cache_inputs, output_cache = phase4_runner.make_cache_helpers(model, cache_dtype=cache_dtype)
    previous_fast_path = phase4_runner.set_granite_fast_path_enabled(False) if autocast_enabled else None
    wanted = set(positions)
    out: dict[int, Any] = {}
    try:
        text = shared.make_prompt_text(str(prompt["prompt"]))
        encoded = tokenizer([text], padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        target_tensor = torch.tensor(target_tokens[:max_new_tokens], dtype=torch.long, device=device)
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
                raise RuntimeError(f"model did not return cache for KL regime {regime_name}")
            logits = outputs.logits[:, -1, :]
            for decode_position in range(1, max_new_tokens + 1):
                current_target = target_tensor[decode_position - 1]
                if decode_position in wanted:
                    out[decode_position] = torch.log_softmax(logits.float().squeeze(0), dim=-1).cpu()
                if decode_position == max_new_tokens:
                    break
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)],
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
                    raise RuntimeError(f"model dropped cache for {regime_name} at decode position {decode_position}")
                logits = outputs.logits[:, -1, :]
        return out
    finally:
        if previous_fast_path is not None:
            phase4_runner.set_granite_fast_path_enabled(bool(previous_fast_path))


def run_model(
    *,
    run_dir: Path,
    run_events_path: Path,
    model_key: str,
    dtype_name: str,
    device_name: str,
    resume: bool,
) -> dict[str, Any]:
    import torch

    model_config = checker.MODEL_CONFIGS[model_key]
    source_run_dir = ROOT / model_config["source_run_dir"]
    model_dir = run_dir / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    if resume and (model_dir / "spectral_summary.json").is_file() and (model_dir / "growth_model_fits.json").is_file():
        return {"model_key": model_key, "status": "reused_existing_complete_artifacts"}

    model_provenance = json.loads((source_run_dir / "model_provenance.json").read_text(encoding="utf-8"))
    if model_provenance.get("model_id") != model_config["model_id"]:
        raise RuntimeError(f"{model_key}: source model_id mismatch")
    if model_provenance.get("hf_snapshot_commit") != model_config["snapshot"]:
        raise RuntimeError(f"{model_key}: source snapshot mismatch")
    prompt_manifest, prompts = load_prompts(source_run_dir)
    shared.write_json(model_dir / "model_provenance.json", model_provenance)
    shared.write_json(model_dir / "prompt_manifest.json", prompt_manifest)

    kl_positions = checker.dense_grid_positions()
    spectral_positions = checker.spectral_positions()
    shared.write_json(
        model_dir / "kl_positions.json",
        {
            "schema_version": f"{SCHEMA_VERSION}_kl_positions",
            "created_at_utc": shared.utc_now(),
            "positions": kl_positions,
            "position_count": len(kl_positions),
            "source": "same fixed dense grid as Granite-Small KL accumulation experiment",
        },
    )

    trace_path = model_dir / "bf16_traces.jsonl.gz"
    activation_npz_path = model_dir / "activation_means.npz"
    tmp_dir = ROOT / ".debug" / "stage1_e1" / run_dir.name / model_key
    if tmp_dir.exists() and not resume:
        shutil.rmtree(tmp_dir)
    model, tokenizer, device = shared.load_model_and_tokenizer(model_provenance, dtype_name=dtype_name, device_name=device_name)
    reference_manifest = collect_bf16_reference_trace_logprobs_and_activations(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompts=prompts,
        kl_positions=kl_positions,
        spectral_positions=spectral_positions,
        max_new_tokens=checker.MAX_NEW_TOKENS,
        trace_path=trace_path,
        activation_npz_path=activation_npz_path,
        tmp_dir=tmp_dir,
        run_events_path=run_events_path,
    )
    del model, tokenizer, device
    release_model_memory()
    shared.write_json(model_dir / "bf16_trace_manifest.json", reference_manifest["trace_manifest"])
    shared.write_json(model_dir / "activation_summary_manifest.json", reference_manifest["activation_manifest"])

    target_tokens = phase4_runner.load_trace_tokens(trace_path)
    for prompt in prompts:
        index = int(prompt["index"])
        if len(target_tokens[index]) < checker.MAX_NEW_TOKENS:
            raise RuntimeError(f"{model_key}: trace {index} has only {len(target_tokens[index])} target tokens")

    means = reference_manifest["activation_means"]
    layer_names = reference_manifest["layer_names"]
    protected_sets = build_protected_sets(means, layer_names)
    shared.write_json(model_dir / "protected_sets.json", protected_sets)
    spectral = write_spectral_summary(model_dir, means, layer_names, spectral_positions)

    rows_by_regime_position: dict[str, dict[int, list[float]]] = {
        regime: defaultdict(list) for regime in checker.KL_REGIMES
    }
    kl_path = model_dir / "kl_rows.jsonl.gz"
    with gzip.open(kl_path, "wt", encoding="utf-8") as kl_handle:
        for prompt in prompts:
            index = int(prompt["index"])
            for position in kl_positions:
                row = {
                    "schema_version": f"{SCHEMA_VERSION}_kl_row",
                    "prompt_index": index,
                    "prompt_id": prompt["prompt_id"],
                    "regime": "bf16_reference",
                    "decode_position": position,
                    "kl_bf16_q": 0.0,
                }
                kl_handle.write(json.dumps(row, sort_keys=True) + "\n")
                rows_by_regime_position["bf16_reference"][position].append(0.0)
        for regime in checker.QUANTIZED_REGIMES:
            print(json.dumps({"event": "starting_e1_quantized_kl", "model_key": model_key, "regime": regime, "time": shared.utc_now()}, sort_keys=True))
            collect_quantized_kl(
                model_provenance=model_provenance,
                protected_sets=protected_sets,
                regime=regime,
                prompts=prompts,
                target_tokens=target_tokens,
                positions=kl_positions,
                max_new_tokens=checker.MAX_NEW_TOKENS,
                dtype_name=dtype_name,
                device_name=device_name,
                tmp_dir=tmp_dir,
                kl_handle=kl_handle,
                rows_by_regime_position=rows_by_regime_position,
                model_dir=model_dir,
                run_events_path=run_events_path,
            )
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    kl_summary, fits = write_kl_summary(model_dir, kl_positions, rows_by_regime_position)
    return {
        "model_key": model_key,
        "status": "completed",
        "spectral": {
            "median_normalized_spectral_entropy": spectral.get("median_normalized_spectral_entropy"),
            "median_autocorrelation_length_tokens": spectral.get("median_autocorrelation_length_tokens"),
        },
        "kl_best_fits": {
            regime: fits.get("regime_fits", {}).get(regime, {}).get("best_fit")
            for regime in checker.QUANTIZED_REGIMES
        },
        "mean_kl_by_regime": {
            regime: kl_summary.get("regime_summary", {}).get(regime, {}).get("mean_kl_all_positions_traces")
            for regime in checker.QUANTIZED_REGIMES
        },
    }


def parse_model_keys(text: str) -> list[str]:
    if text == "all":
        return list(checker.MODEL_KEYS)
    keys = [item.strip() for item in text.split(",") if item.strip()]
    unknown = [key for key in keys if key not in checker.MODEL_KEYS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown model keys: {unknown}; choices={checker.MODEL_KEYS}")
    return keys


def cap_exhausted(start_time: float, cap_hours: float) -> bool:
    return (time.monotonic() - start_time) / 3600.0 >= cap_hours


def main(argv: list[str] | None = None) -> int:
    shared.SCHEMA_VERSION = SCHEMA_VERSION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"om_stage1_e1_kl_fft_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--models", type=parse_model_keys, default=list(checker.MODEL_KEYS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--cap-hours", type=float, default=15.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    started = time.monotonic()

    run_dir = args.results_dir / args.run_id
    if run_dir.exists() and not args.resume:
        raise SystemExit(f"run directory already exists: {run_dir}")
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    stdout_log = (run_dir / "logs/stdout.log").open("a", encoding="utf-8", buffering=1)
    stderr_log = (run_dir / "logs/stderr.log").open("a", encoding="utf-8", buffering=1)
    sys.stdout = shared.Tee(sys.__stdout__, stdout_log)
    sys.stderr = shared.Tee(sys.__stderr__, stderr_log)
    run_events_path = run_dir / "run_events.jsonl"
    if not run_events_path.exists():
        run_events_path.write_text(json.dumps({"created_at_utc": shared.utc_now(), "event": "run_started"}, sort_keys=True) + "\n", encoding="utf-8")

    try:
        import torch

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        environment = shared.build_environment(schema_version=SCHEMA_VERSION)
        shared.write_json(run_dir / "environment.json", environment)
        phase4_runner.write_environment_text(run_dir / "environment.txt", environment)
        shared.write_json(
            run_dir / "command_metadata.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_command",
                "created_at_utc": shared.utc_now(),
                "argv": sys.argv if argv is None else ["run_om_stage1_e1_cross_model_kl_fft.py", *argv],
                "cwd": str(ROOT),
                "run_dir": str(run_dir),
                "model_keys": args.models,
                "cap_hours": args.cap_hours,
                "nemotron_deferred_reason": checker.MODEL_CONFIGS["nemotron3_nano"].get("deferred_reason"),
                "preregistration": str(checker.PREREG_PATH.relative_to(ROOT)),
                "preregistration_sha256": shared.file_sha256(checker.PREREG_PATH),
            },
        )
        shared.write_json(
            run_dir / "random_seed.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_random_seed",
                "seed": args.seed,
                "determinism": {"do_sample": False, "num_beams": 1, "torch_manual_seed": args.seed},
            },
        )
        shared.write_json(
            run_dir / "source_artifacts.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_source_artifacts",
                "created_at_utc": shared.utc_now(),
                "sources": {
                    key: {
                        **checker.MODEL_CONFIGS[key],
                        "source_run_dir_sha256s": [
                            {"path": str(path.relative_to(ROOT)), "sha256": shared.file_sha256(path), "bytes": path.stat().st_size}
                            for path in sorted((ROOT / checker.MODEL_CONFIGS[key]["source_run_dir"]).glob("*.json"))
                        ],
                    }
                    for key in args.models
                },
            },
        )
        shared.write_json(
            run_dir / "quantization_config.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_quantization_config",
                "weight_quantization": "simple symmetric per-channel INT4 represented as dequantized tensors",
                "activation_dtype": "float16 autocast for quantized KL regimes",
                "protected_channel_dtype": "bfloat16",
                "regimes": checker.KL_REGIMES,
            },
        )

        model_summaries = []
        for model_key in args.models:
            if cap_exhausted(started, args.cap_hours):
                run_events_path.open("a", encoding="utf-8").write(
                    json.dumps(
                        {
                            "created_at_utc": shared.utc_now(),
                            "event": "cap_exhausted_before_model",
                            "model_key": model_key,
                            "cap_hours": args.cap_hours,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                break
            print(json.dumps({"event": "starting_e1_model", "model_key": model_key, "time": shared.utc_now()}, sort_keys=True))
            model_summaries.append(
                run_model(
                    run_dir=run_dir,
                    run_events_path=run_events_path,
                    model_key=model_key,
                    dtype_name=args.dtype,
                    device_name=args.device,
                    resume=args.resume,
                )
            )
            shared.write_json(
                run_dir / "stage1_e1_summary.json",
                {
                    "schema_version": f"{SCHEMA_VERSION}_summary",
                    "created_at_utc": shared.utc_now(),
                    "model_summaries": model_summaries,
                    "completed_model_count": len(model_summaries),
                    "requested_model_count": len(args.models),
                },
            )
            shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))

        run_events_path.open("a", encoding="utf-8").write(json.dumps({"created_at_utc": shared.utc_now(), "event": "run_completed"}, sort_keys=True) + "\n")
        shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
        result = checker.evaluate(run_dir)
        print(json.dumps({"checker_decision": result["decision"], "artifact_complete": result["artifact_complete"]}, indent=2, sort_keys=True))
        shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
        checker.evaluate(run_dir)
        return 0
    except BaseException as exc:
        write_failure_packet(run_dir, run_events_path, exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
