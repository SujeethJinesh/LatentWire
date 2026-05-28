#!/usr/bin/env python3
"""Run Phase 9 LAMBDA/HYST positive-method funnel smoke tests."""

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

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.outlier_migrate.phase4 import check_om_phase4_intervention as phase4_checker
from experimental.outlier_migrate.phase4 import run_om_phase4_intervention as phase4_runner
from experimental.outlier_migrate.phase9 import check_om_phase9_funnel_smoke as checker
from experimental.outlier_migrate.phase9 import run_om_phase9_m11_ema_drift as m11_runner
from experimental.outlier_migrate.phase9 import run_om_phase9_m11b_budget_scaling as m11b_runner
from experimental.outlier_migrate.phase9 import run_om_phase9_m2_position_conditional as m2_runner
from experimental.shared import run_phase0_branch as shared


SOURCE_RUNS = {
    "deepseek": ROOT / "experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z",
    "falcon": ROOT / "experimental/outlier_migrate/phase9/results/om_v2_m11b_falcon_20260527T1438Z",
}
BASELINE_REGIMES = ["bf16", "static_1pct", "m11b_top10"]
METHOD_REGIMES_BY_NAME = {
    "lambda": ("mlambda_top10_smoke", "random_lambda_top10"),
    "hyst": ("hyst_top10_smoke", "random_hyst_top10"),
}
DEFAULT_METHODS = ("lambda", "hyst")
UPDATE_POSITIONS = tuple(range(100, checker.SCORING_POSITION + 1, 100))
SMOKE_SELECTION_SOURCE = "docs/smoke_trace_selection.md"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_failure_packet(run_dir: Path, run_events_path: Path, exc: BaseException | str) -> None:
    payload: dict[str, Any] = {
        "schema_version": f"{checker.SCHEMA_VERSION}_infra_error",
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
        shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=checker.SCHEMA_VERSION))
        checker.evaluate(run_dir)
    except Exception:
        pass


def ensure_source(source_dir: Path) -> None:
    required = [
        "activation_magnitudes.jsonl.gz",
        "activation_magnitude_manifest.json",
        "bf16_traces.jsonl.gz",
        "bf16_trace_manifest.json",
        "decoding_config.json",
        "environment.json",
        "environment.txt",
        "model_provenance.json",
        "prompt_manifest.json",
        "protected_sets.json",
        "quantization_config.json",
    ]
    missing = [rel for rel in required if not (source_dir / rel).is_file()]
    missing.extend(f"score_cache/{regime}.json" for regime in BASELINE_REGIMES if not (source_dir / "score_cache" / f"{regime}.json").is_file())
    if missing:
        raise FileNotFoundError(f"source packet missing required artifacts: {missing}")


def copy_packet_inputs(source_dir: Path, run_dir: Path, prompt_indices: set[int]) -> None:
    for rel in [
        "activation_magnitudes.jsonl.gz",
        "activation_magnitude_manifest.json",
        "decoding_config.json",
        "environment.json",
        "environment.txt",
        "model_provenance.json",
        "quantization_config.json",
    ]:
        shutil.copy2(source_dir / rel, run_dir / rel)
    m2_runner.copy_filtered_jsonl_gz(source_dir / "bf16_traces.jsonl.gz", run_dir / "bf16_traces.jsonl.gz", prompt_indices=prompt_indices)
    trace_manifest = load_json(source_dir / "bf16_trace_manifest.json")
    trace_manifest["created_at_utc"] = shared.utc_now()
    trace_manifest["source_run_dir"] = str(source_dir)
    trace_manifest["prompt_indices"] = sorted(prompt_indices)
    trace_manifest["artifact"] = "bf16_traces.jsonl.gz"
    trace_manifest["artifact_sha256"] = shared.file_sha256(run_dir / "bf16_traces.jsonl.gz")
    shared.write_json(run_dir / "bf16_trace_manifest.json", trace_manifest)


def filtered_prompt_manifest(source_dir: Path, prompt_indices: list[int]) -> dict[str, Any]:
    source = load_json(source_dir / "prompt_manifest.json")
    by_index = {int(row["index"]): row for row in source["prompts"]}
    prompts = [by_index[index] for index in prompt_indices]
    manifest = dict(source)
    manifest["created_at_utc"] = shared.utc_now()
    manifest["selection"] = "positive_method_funnel_smoke_stratified"
    manifest["prompt_count"] = len(prompts)
    manifest["prompts"] = prompts
    manifest["prompt_sha256"] = phase4_checker.prompt_payload_sha256(prompts)
    manifest["source_prompt_manifest"] = str(source_dir / "prompt_manifest.json")
    return manifest


def top_channels(values: np.ndarray, count: int) -> list[int]:
    if count <= 0:
        return []
    count = min(count, int(values.shape[0]))
    candidates = np.argpartition(-values, count - 1)[:count]
    return sorted((int(channel) for channel in candidates), key=lambda channel: (-float(values[channel]), channel))[:count]


def activation_means(rows: list[dict[str, Any]]) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, str]]:
    grouped: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    layer_names: dict[int, str] = {}
    for row in rows:
        layer_index = int(row["layer_index"])
        position = int(row["decode_position"])
        layer_names[layer_index] = str(row["layer_name"])
        values = np.asarray(row["channel_magnitudes"], dtype=np.float64)
        entry = grouped[layer_index].get(position)
        if entry is None:
            entry = {"sum": np.zeros_like(values), "count": 0}
            grouped[layer_index][position] = entry
        entry["sum"] += values
        entry["count"] += 1
    means_by_layer: dict[int, dict[int, np.ndarray]] = {}
    for layer_index, by_position in grouped.items():
        means_by_layer[layer_index] = {
            position: entry["sum"] / max(1, int(entry["count"])) for position, entry in by_position.items()
        }
    missing: list[str] = []
    for layer_index, by_position in means_by_layer.items():
        absent = [position for position in UPDATE_POSITIONS if position not in by_position]
        if absent:
            missing.append(f"layer {layer_index}: missing {absent[:5]}")
    if missing:
        raise RuntimeError("funnel smoke requires 100-position cached activations; " + "; ".join(missing[:4]))
    return means_by_layer, layer_names


def final_ema_scores(means_by_position: dict[int, np.ndarray], alpha: float) -> np.ndarray:
    scores = np.square(means_by_position[100])
    for position in UPDATE_POSITIONS:
        scores = alpha * np.square(means_by_position[position]) + (1.0 - alpha) * scores
    return scores


def lambda_layers(
    means_by_layer: dict[int, dict[int, np.ndarray]],
    source_layers: dict[str, Any],
    layer_names: dict[int, str],
    alpha: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    total_budget = sum(int(layer["protected_count"]) for layer in source_layers.values())
    scored: list[tuple[float, int, int]] = []
    channel_counts: dict[int, int] = {}
    for layer_index in sorted(means_by_layer):
        scores = final_ema_scores(means_by_layer[layer_index], alpha)
        channel_counts[layer_index] = int(scores.shape[0])
        scored.extend((float(score), layer_index, channel) for channel, score in enumerate(scores))
    selected = sorted(scored, key=lambda item: (-item[0], item[1], item[2]))[:total_budget]
    by_layer: dict[int, list[int]] = defaultdict(list)
    for _score, layer_index, channel in selected:
        by_layer[layer_index].append(channel)
    layers: dict[str, Any] = {}
    allocation: dict[str, Any] = {}
    for layer_index in sorted(means_by_layer):
        channels = sorted(by_layer.get(layer_index, []))
        layers[str(layer_index)] = {
            "layer_name": layer_names[layer_index],
            "channel_count": channel_counts[layer_index],
            "protected_count": len(channels),
            "protected_channels": channels,
            "source": "global_ema_squared_magnitude_waterfill_smoke",
        }
        allocation[str(layer_index)] = {
            "layer_name": layer_names[layer_index],
            "protected_count": len(channels),
            "flat_m11b_count": int(source_layers[str(layer_index)]["protected_count"]),
        }
    return layers, allocation


def hysteresis_layers(
    means_by_layer: dict[int, dict[int, np.ndarray]],
    source_layers: dict[str, Any],
    layer_names: dict[int, str],
    exit_margin_pct_points: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    layers: dict[str, Any] = {}
    trajectories: dict[str, Any] = {}
    for layer_index in sorted(means_by_layer):
        budget = int(source_layers[str(layer_index)]["protected_count"])
        channel_count = int(means_by_layer[layer_index][100].shape[0])
        exit_count = min(channel_count, budget + int(round(channel_count * exit_margin_pct_points / 100.0)))
        selected = set(top_channels(means_by_layer[layer_index][100], budget))
        steps: list[dict[str, Any]] = []
        for position in UPDATE_POSITIONS:
            values = means_by_layer[layer_index][position]
            enter = set(top_channels(values, budget))
            local_pool = set(top_channels(values, exit_count))
            selected = {channel for channel in selected if channel in local_pool}
            selected.update(enter)
            if len(selected) > budget:
                ordered = sorted(selected, key=lambda channel: (-float(values[channel]), channel))
                selected = set(ordered[:budget])
            elif len(selected) < budget:
                for channel in top_channels(values, budget):
                    selected.add(channel)
                    if len(selected) == budget:
                        break
            if position % 1000 == 0 or position in {100, checker.SCORING_POSITION}:
                steps.append({"position": position, "protected_count": len(selected), "protected_channels": sorted(selected)})
        layers[str(layer_index)] = {
            "layer_name": layer_names[layer_index],
            "channel_count": channel_count,
            "protected_count": len(selected),
            "protected_channels": sorted(selected),
            "source": f"hysteresis_enter_top_k_exit_top_k_plus_{exit_margin_pct_points:g}pct_points_smoke",
            "exit_margin_pct_points": exit_margin_pct_points,
            "exit_rank_count": exit_count,
        }
        trajectories[str(layer_index)] = {
            "layer_name": layer_names[layer_index],
            "channel_count": channel_count,
            "budget": budget,
            "exit_margin_pct_points": exit_margin_pct_points,
            "exit_rank_count": exit_count,
            "steps_recorded": steps,
            "final_protected_channels": sorted(selected),
        }
    return layers, trajectories


def random_layers(reference_layers: dict[str, Any], seed: int, source: str) -> dict[str, Any]:
    rng = random.Random(seed)
    layers: dict[str, Any] = {}
    for layer_key, layer in sorted(reference_layers.items(), key=lambda item: int(item[0])):
        channel_count = int(layer["channel_count"])
        count = int(layer["protected_count"])
        channels = sorted(rng.sample(range(channel_count), count)) if count > 0 else []
        layers[layer_key] = {
            "layer_name": layer["layer_name"],
            "channel_count": channel_count,
            "protected_count": count,
            "protected_channels": channels,
            "source": source,
        }
    return layers


def build_protected_sets(source_dir: Path, seed: int, hyst_exit_margin_pct_points: float) -> tuple[dict[str, Any], dict[str, Any]]:
    source_sets = load_json(source_dir / "protected_sets.json")
    rows = list(shared.iter_activation_rows(source_dir / "activation_magnitudes.jsonl.gz"))
    means_by_layer, layer_names = activation_means(rows)
    alpha = float(source_sets.get("alpha", 0.3))
    static_layers = source_sets["regimes"]["static_1pct"]["layers"]
    m11b_layers = source_sets["regimes"]["m11b_top10"]["layers"]
    mlambda, lambda_allocation = lambda_layers(means_by_layer, m11b_layers, layer_names, alpha)
    hyst, hyst_trajectories = hysteresis_layers(
        means_by_layer,
        m11b_layers,
        layer_names,
        exit_margin_pct_points=hyst_exit_margin_pct_points,
    )
    random_lambda = random_layers(mlambda, seed + 17, "random_matched_lambda_layer_allocation")
    random_hyst = random_layers(m11b_layers, seed + 29, "random_matched_flat_hyst_budget")
    protected_sets = {
        "schema_version": f"{checker.SCHEMA_VERSION}_protected_sets",
        "created_at_utc": shared.utc_now(),
        "selection_basis": "CPU-gated LAMBDA/HYST smoke from cached deterministic AIME-2025 activation magnitudes",
        "alpha": alpha,
        "hysteresis_exit_margin_pct_points": hyst_exit_margin_pct_points,
        "regimes": {
            "static_1pct": {"layers": static_layers},
            "m11b_top10": {"layers": m11b_layers},
            "mlambda_top10_smoke": {"layers": mlambda},
            "hyst_top10_smoke": {"layers": hyst},
            "random_lambda_top10": {"layers": random_lambda},
            "random_hyst_top10": {"layers": random_hyst},
        },
    }
    trajectories = {
        "schema_version": f"{checker.SCHEMA_VERSION}_protected_trajectories",
        "created_at_utc": shared.utc_now(),
        "update_positions": list(UPDATE_POSITIONS),
        "alpha": alpha,
        "hysteresis_exit_margin_pct_points": hyst_exit_margin_pct_points,
        "by_regime": {
            "mlambda_top10_smoke": {"layer_allocation": lambda_allocation},
            "hyst_top10_smoke": {"layers": hyst_trajectories},
        },
        "protected_set_count_stats": m11b_runner.summarize_protected_counts(protected_sets),
    }
    return protected_sets, trajectories


def read_score_cache_subset(source_dir: Path, regime: str, prompt_indices: set[int]) -> dict[int, dict[str, float]]:
    payload = load_json(source_dir / "score_cache" / f"{regime}.json")
    scores = {int(index): row for index, row in payload.get("scores", {}).items()}
    missing = prompt_indices - set(scores)
    if missing:
        raise KeyError(f"{regime} score cache missing prompt indices {sorted(missing)}")
    return {index: scores[index] for index in sorted(prompt_indices)}


def write_score_cache(run_dir: Path, regime: str, scores: dict[int, dict[str, float]]) -> None:
    shared.write_json(
        run_dir / "score_cache" / f"{regime}.json",
        {
            "schema_version": f"{checker.SCHEMA_VERSION}_score_cache",
            "created_at_utc": shared.utc_now(),
            "regime": regime,
            "scores": {str(index): row for index, row in sorted(scores.items())},
        },
    )


def build_per_trace_rows(
    prompts: list[dict[str, Any]],
    all_scores: dict[str, dict[int, dict[str, float]]],
    score_regimes: list[str],
    recovery_regimes: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        index = int(prompt["index"])
        perplexities = {regime: float(all_scores[regime][index]["perplexity"]) for regime in score_regimes}
        mean_nll = {regime: float(all_scores[regime][index]["mean_nll"]) for regime in score_regimes}
        static_gap = perplexities["static_1pct"] - perplexities["bf16"]
        no_gap = static_gap <= 0.0
        recoveries = {
            regime: None if no_gap else 1.0 - (perplexities[regime] - perplexities["bf16"]) / static_gap
            for regime in checker.RECOVERY_REGIMES
        }
        rows.append(
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
    return rows


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "median_recovery": float(median(values)) if values else None,
        "mean_recovery": float(mean(values)) if values else None,
        "bootstrap_ci95": phase4_checker.bootstrap_median(values),
        "included_trace_count": len(values),
        "per_trace_recovery_included": [float(value) for value in values],
    }


def build_metrics(
    run_dir: Path,
    model_key: str,
    prompt_manifest: dict[str, Any],
    model_provenance: dict[str, Any],
    per_trace_rows: list[dict[str, Any]],
    protected_trajectories: dict[str, Any],
    active_methods: list[str],
    method_regimes: list[str],
    control_regimes: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    included = [row for row in per_trace_rows if not bool(row["no_recoverable_static_gap"])]
    recovery_regimes = ["m11b_top10", *method_regimes, *control_regimes]
    summaries = {
        regime: summarize([float(row["recoveries"][regime]) for row in included])
        for regime in recovery_regimes
    }
    no_gap_count = len(per_trace_rows) - len(included)
    for summary in summaries.values():
        summary["total_trace_count"] = len(per_trace_rows)
        summary["no_recoverable_static_gap_count"] = no_gap_count
        summary["no_recoverable_static_gap_fraction"] = no_gap_count / len(per_trace_rows) if per_trace_rows else 0.0
    metrics = {
        "schema_version": f"{checker.SCHEMA_VERSION}_metrics",
        "created_at_utc": shared.utc_now(),
        "preregistration": str(checker.PREREG_PATH.relative_to(ROOT)),
        "preregistration_sha256": shared.file_sha256(checker.PREREG_PATH),
        "model_key": model_key,
        "model_id": model_provenance.get("model_id"),
        "model_snapshot_commit": model_provenance.get("hf_snapshot_commit"),
        "prompt_sha256": prompt_manifest.get("prompt_sha256"),
        "trace_count": checker.TRACE_COUNT,
        "included_trace_count": len(included),
        "no_recoverable_static_gap_count": no_gap_count,
        "no_recoverable_static_gap_fraction": no_gap_count / len(per_trace_rows) if per_trace_rows else 0.0,
        "scoring_position": checker.SCORING_POSITION,
        "scoring_window_tokens": checker.SCORING_WINDOW_TOKENS,
        "metric_name": "positive-static-1pct-gap per-trace recovery",
        "metric_formula": "1 - (perplexity_regime - perplexity_BF16) / (perplexity_static_1pct - perplexity_BF16)",
        "active_methods": active_methods,
        "active_method_regimes": method_regimes,
        "active_control_regimes": control_regimes,
        "results_by_regime": summaries,
        "thresholds": checker.THRESHOLDS,
        "protected_set_count_stats": protected_trajectories.get("protected_set_count_stats", {}),
        "artifacts": {"run_dir": str(run_dir)},
    }
    controls = {
        "schema_version": f"{checker.SCHEMA_VERSION}_control_metrics",
        "created_at_utc": shared.utc_now(),
        "controls": {regime: summaries[regime] for regime in control_regimes},
        "method_minus_m11b": {
            regime: (
                None
                if summaries[regime]["median_recovery"] is None or summaries["m11b_top10"]["median_recovery"] is None
                else float(summaries[regime]["median_recovery"]) - float(summaries["m11b_top10"]["median_recovery"])
            )
            for regime in method_regimes
        },
    }
    return metrics, controls


def source_artifacts(source_dir: Path) -> dict[str, Any]:
    rels = [
        "activation_magnitudes.jsonl.gz",
        "activation_magnitude_manifest.json",
        "bf16_traces.jsonl.gz",
        "bf16_trace_manifest.json",
        "protected_sets.json",
        "prompt_manifest.json",
        *[f"score_cache/{regime}.json" for regime in BASELINE_REGIMES],
        "artifacts/funnel_prefilters/decision.json",
        "artifacts/funnel_prefilters/smoke_traces.json",
        "artifacts/wjac_prefilter/decision.json",
    ]
    artifacts: list[dict[str, Any]] = []
    for rel in rels:
        path = (ROOT / rel) if rel.startswith("artifacts/") else (source_dir / rel)
        if path.is_file():
            artifacts.append({"name": rel.replace("/", "_"), "path": str(path), "sha256": shared.file_sha256(path)})
    return {"schema_version": f"{checker.SCHEMA_VERSION}_source_artifacts", "created_at_utc": shared.utc_now(), "artifacts": artifacts}


def write_run_contract(
    run_dir: Path,
    args: argparse.Namespace,
    model_key: str,
    prompt_indices: list[int],
    active_methods: list[str],
    score_regimes: list[str],
    new_regimes: list[str],
) -> None:
    command = " ".join(sys.argv)
    shared.write_json(
        run_dir / "config.json",
        {
            "schema_version": f"{checker.SCHEMA_VERSION}_config",
            "created_at_utc": shared.utc_now(),
            "model_key": model_key,
            "source_run_dir": str(args.source_run_dir or SOURCE_RUNS[model_key]),
            "prompt_indices": prompt_indices,
            "active_methods": active_methods,
            "regimes": score_regimes,
            "new_regimes": new_regimes,
            "batch_size": args.batch_size,
            "dtype": args.dtype,
            "device": args.device,
            "hyst_exit_margin_pct_points": args.hyst_exit_margin_pct_points,
        },
    )
    (run_dir / "command.sh").write_text(f"#!/usr/bin/env bash\nset -euo pipefail\n{command}\n", encoding="utf-8")
    shared.write_json(
        run_dir / "traces_used.json",
        {
            "schema_version": f"{checker.SCHEMA_VERSION}_traces_used",
            "created_at_utc": shared.utc_now(),
            "selection": "stratified_positive_method_funnel_smoke",
            "model_key": model_key,
            "prompt_indices": prompt_indices,
            "selection_source": SMOKE_SELECTION_SOURCE,
        },
    )


def main(argv: list[str] | None = None) -> int:
    shared.SCHEMA_VERSION = checker.SCHEMA_VERSION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=sorted(checker.MODEL_SPECS), required=True)
    parser.add_argument("--source-run-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--results-dir", type=Path, default=checker.RESULTS_DIR)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated smoke methods to score. Choices: lambda,hyst. Default preserves the original all-method packet.",
    )
    parser.add_argument(
        "--hyst-exit-margin-pct-points",
        type=float,
        default=10.0,
        help="HYST exit margin in percentage points above the enter top-k percent; 10 preserves the original top-2k policy.",
    )
    args = parser.parse_args(argv)
    active_methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    invalid_methods = sorted(set(active_methods) - set(METHOD_REGIMES_BY_NAME))
    if invalid_methods:
        raise SystemExit(f"unknown smoke methods: {invalid_methods}")
    if not active_methods:
        raise SystemExit("--methods must include at least one smoke method")
    method_regimes = [METHOD_REGIMES_BY_NAME[method][0] for method in active_methods]
    control_regimes = [METHOD_REGIMES_BY_NAME[method][1] for method in active_methods]
    new_regimes = [regime for pair in (METHOD_REGIMES_BY_NAME[method] for method in active_methods) for regime in pair]
    score_regimes = [*BASELINE_REGIMES, *new_regimes]

    model_key = args.model_key
    spec = checker.MODEL_SPECS[model_key]
    prompt_indices = [int(index) for index in spec["smoke_indices"]]
    prompt_index_set = set(prompt_indices)
    source_dir = (args.source_run_dir or SOURCE_RUNS[model_key]).resolve()
    ensure_source(source_dir)
    run_id = args.run_id or f"om_phase9_funnel_smoke_{model_key}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = args.results_dir / run_id
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")
    (run_dir / "logs").mkdir(parents=True)
    stdout_log = (run_dir / "logs/stdout.log").open("w", encoding="utf-8", buffering=1)
    stderr_log = (run_dir / "logs/stderr.log").open("w", encoding="utf-8", buffering=1)
    sys.stdout = shared.Tee(sys.__stdout__, stdout_log)
    sys.stderr = shared.Tee(sys.__stderr__, stderr_log)
    run_events_path = run_dir / "run_events.jsonl"
    previous_excepthook = sys.excepthook

    def funnel_excepthook(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        write_failure_packet(run_dir, run_events_path, exc)
        previous_excepthook(exc_type, exc, tb)

    sys.excepthook = funnel_excepthook
    run_events_path.write_text(
        json.dumps({"created_at_utc": shared.utc_now(), "event": "funnel_smoke_started", "model_key": model_key}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    random.seed(args.seed)

    try:
        write_run_contract(run_dir, args, model_key, prompt_indices, active_methods, score_regimes, new_regimes)
        copy_packet_inputs(source_dir, run_dir, prompt_index_set)
        prompt_manifest = filtered_prompt_manifest(source_dir, prompt_indices)
        model_provenance = load_json(run_dir / "model_provenance.json")
        if model_provenance.get("model_id") != spec["model_id"]:
            raise RuntimeError(f"source model_id mismatch: {model_provenance.get('model_id')}")
        if model_provenance.get("hf_snapshot_commit") != spec["snapshot"]:
            raise RuntimeError("source model snapshot mismatch")
        shared.write_json(run_dir / "prompt_manifest.json", prompt_manifest)
        shared.write_json(
            run_dir / "command_metadata.json",
            {
                "schema_version": f"{checker.SCHEMA_VERSION}_command",
                "created_at_utc": shared.utc_now(),
                "argv": sys.argv if argv is None else ["run_om_phase9_funnel_smoke.py", *argv],
                "cwd": str(Path.cwd()),
                "branch": "positive_method_funnel_smoke",
                "run_dir": str(run_dir),
                "source_run_dir": str(source_dir),
                "model_key": model_key,
                "active_methods": active_methods,
                "batch_size": args.batch_size,
            },
        )
        shared.write_json(
            run_dir / "random_seed.json",
            {"schema_version": f"{checker.SCHEMA_VERSION}_random_seed", "seed": args.seed, "determinism": {"do_sample": False, "num_beams": 1}},
        )
        shared.write_json(
            run_dir / "smoke_trace_selection.json",
            {
                "schema_version": f"{checker.SCHEMA_VERSION}_smoke_trace_selection",
                "created_at_utc": shared.utc_now(),
                "selection_source": SMOKE_SELECTION_SOURCE,
                "artifactized_selection_source": "artifacts/funnel_prefilters/smoke_traces.json",
                "smoke_indices": prompt_indices,
                "model_key": model_key,
                "note": "docs path retained for checker compatibility with the frozen preregistration; artifactized CPU source is also recorded.",
            },
        )
        protected_sets, protected_trajectories = build_protected_sets(
            source_dir,
            args.seed,
            hyst_exit_margin_pct_points=args.hyst_exit_margin_pct_points,
        )
        shared.write_json(run_dir / "protected_sets.json", protected_sets)
        shared.write_json(run_dir / "protected_trajectories.json", protected_trajectories)
        shared.write_json(run_dir / "source_artifacts.json", source_artifacts(source_dir))

        prompts = prompt_manifest["prompts"]
        target_tokens = phase4_runner.load_trace_tokens(run_dir / "bf16_traces.jsonl.gz")
        all_scores: dict[str, dict[int, dict[str, float]]] = {}
        excluded_by_regime: dict[str, Any] = {}
        for regime in BASELINE_REGIMES:
            all_scores[regime] = read_score_cache_subset(source_dir, regime, prompt_index_set)
            write_score_cache(run_dir, regime, all_scores[regime])
            if regime != "bf16":
                excluded_by_regime[regime] = {"regime": regime, "reused_score_cache": str(source_dir)}
        for regime in new_regimes:
            run_events_path.open("a", encoding="utf-8").write(
                json.dumps({"created_at_utc": shared.utc_now(), "event": "score_regime_started", "regime": regime}, sort_keys=True) + "\n"
            )
            all_scores[regime], excluded_by_regime[regime] = m11b_runner.score_regime(
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
            {"schema_version": f"{checker.SCHEMA_VERSION}_excluded_tensors", "created_at_utc": shared.utc_now(), "by_regime": excluded_by_regime},
        )
        per_trace_rows = build_per_trace_rows(prompts, all_scores, score_regimes, ["m11b_top10", *method_regimes, *control_regimes])
        shared.write_json(
            run_dir / "per_trace_metrics.json",
            {"schema_version": f"{checker.SCHEMA_VERSION}_per_trace_metrics", "created_at_utc": shared.utc_now(), "traces": per_trace_rows},
        )
        metrics, controls = build_metrics(
            run_dir,
            model_key,
            prompt_manifest,
            model_provenance,
            per_trace_rows,
            protected_trajectories,
            active_methods,
            method_regimes,
            control_regimes,
        )
        shared.write_json(run_dir / "metrics.json", metrics)
        shared.write_json(run_dir / "control_metrics.json", controls)
        run_events_path.open("a", encoding="utf-8").write(
            json.dumps({"created_at_utc": shared.utc_now(), "event": "funnel_smoke_completed", "model_key": model_key}, sort_keys=True) + "\n"
        )
        shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=checker.SCHEMA_VERSION))
        result = checker.evaluate(run_dir)
        print(json.dumps({"run_dir": str(run_dir), "checker_decision": result["decision"], "reasons": result["reasons"]}, indent=2, sort_keys=True))
        shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=checker.SCHEMA_VERSION))
        checker.evaluate(run_dir)
        sys.excepthook = previous_excepthook
        return 0 if result["decision"] != checker.FAIL_INFRA else 1
    except BaseException as exc:
        write_failure_packet(run_dir, run_events_path, exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
