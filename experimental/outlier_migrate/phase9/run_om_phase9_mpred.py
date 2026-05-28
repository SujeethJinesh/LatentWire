#!/usr/bin/env python3
"""Run Phase 9 M-PRED predictive channel tracker."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import traceback
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.outlier_migrate.phase4 import run_om_phase4_intervention as phase4_runner
from experimental.outlier_migrate.phase9 import check_om_phase9_mpred as checker
from experimental.outlier_migrate.phase9 import run_om_phase9_m11_ema_drift as m11_runner
from experimental.outlier_migrate.phase9 import run_om_phase9_m11b_budget_scaling as m11b_runner
from experimental.shared import run_phase0_branch as shared


DEFAULT_SOURCES = {
    "granite": ROOT
    / "experimental/outlier_migrate/phase9/results/om_phase9_m11b_granite_small_vac12_reuse_20260518T030300Z",
    "nemotron": ROOT
    / "experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z",
    "deepseek": ROOT
    / "experimental/outlier_migrate/phase9/results/om_v2_m11b_deepseek_20260527T1210Z",
    "falcon": ROOT
    / "experimental/outlier_migrate/phase9/results/om_v2_m11b_falcon_20260527T1438Z",
}
BASELINE_REGIMES = ["bf16", "static_1pct", "m11b_top5", "m11b_top10", "static_top10"]
NEW_REGIMES = [
    "mpred_top5_alpha_0_5",
    "mpred_top5_alpha_0_8",
    "mpred_top5_alpha_0_95",
    "mpred_top5_alpha_0_99",
    "mpred_top10_alpha_0_95",
    "random_walk_top5",
    "mpred_random_alpha_top5",
]
MPRED_ALPHA_BY_REGIME = {
    "mpred_top5_alpha_0_5": 0.5,
    "mpred_top5_alpha_0_8": 0.8,
    "mpred_top5_alpha_0_95": 0.95,
    "mpred_top5_alpha_0_99": 0.99,
    "mpred_top10_alpha_0_95": 0.95,
}
MPRED_BUDGET_BY_REGIME = {
    "mpred_top5_alpha_0_5": 0.05,
    "mpred_top5_alpha_0_8": 0.05,
    "mpred_top5_alpha_0_95": 0.05,
    "mpred_top5_alpha_0_99": 0.05,
    "mpred_top10_alpha_0_95": 0.10,
}
UPDATE_POSITIONS = tuple(range(checker.UPDATE_CADENCE, checker.SCORING_POSITION + 1, checker.UPDATE_CADENCE))


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
        "activation_magnitude_manifest.json",
        "activation_magnitudes.jsonl.gz",
        "bf16_trace_manifest.json",
        "bf16_traces.jsonl.gz",
        "decoding_config.json",
        "model_provenance.json",
        "prompt_manifest.json",
        "protected_sets.json",
        "quantization_config.json",
    ]
    missing = [rel for rel in required if not (source_dir / rel).is_file()]
    missing += [f"score_cache/{regime}.json" for regime in BASELINE_REGIMES if not (source_dir / "score_cache" / f"{regime}.json").is_file()]
    if missing:
        raise FileNotFoundError(f"source packet missing required artifacts: {missing}")


def copy_source_artifacts(source_dir: Path, run_dir: Path) -> None:
    for rel in [
        "activation_magnitude_manifest.json",
        "activation_magnitudes.jsonl.gz",
        "bf16_trace_manifest.json",
        "bf16_traces.jsonl.gz",
        "decoding_config.json",
        "model_provenance.json",
        "prompt_manifest.json",
        "quantization_config.json",
    ]:
        dest = run_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_dir / rel, dest)


def top_channels(values: Any, count: int) -> list[int]:
    if isinstance(values, np.ndarray):
        if count >= len(values):
            candidates = np.arange(len(values))
        else:
            candidates = np.argpartition(-values, count - 1)[:count]
        return sorted((int(channel) for channel in candidates), key=lambda channel: (-float(values[channel]), channel))[:count]
    return sorted(range(len(values)), key=lambda channel: (-float(values[channel]), channel))[:count]


def variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return float(sum((value - center) ** 2 for value in values) / len(values))


def activation_statistics(
    rows: Iterable[dict[str, Any]],
) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, np.ndarray], dict[int, str]]:
    grouped: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    layer_names: dict[int, str] = {}
    for row in rows:
        layer_index = int(row["layer_index"])
        position = int(row["decode_position"])
        layer_names[layer_index] = str(row["layer_name"])
        values = np.asarray(row["channel_magnitudes"], dtype=np.float64)
        entry = grouped[layer_index].get(position)
        if entry is None:
            entry = {"sum": np.zeros_like(values), "sumsq": np.zeros_like(values), "count": 0}
            grouped[layer_index][position] = entry
        entry["sum"] += values
        entry["sumsq"] += values * values
        entry["count"] += 1

    means_by_layer: dict[int, dict[int, np.ndarray]] = {}
    beta_by_layer: dict[int, np.ndarray] = {}
    for layer_index, by_position in grouped.items():
        means_by_layer[layer_index] = {}
        measurement_var_sum: np.ndarray | None = None
        measurement_var_count = 0
        for position, entry in by_position.items():
            count = int(entry["count"])
            position_mean = entry["sum"] / count
            position_var = np.maximum((entry["sumsq"] / count) - (position_mean * position_mean), 0.0)
            means_by_layer[layer_index][position] = position_mean
            measurement_var_sum = position_var if measurement_var_sum is None else measurement_var_sum + position_var
            measurement_var_count += 1

        positions = sorted(means_by_layer[layer_index])
        measurement_var = measurement_var_sum / max(1, measurement_var_count)
        diffs = np.stack(
            [means_by_layer[layer_index][right] - means_by_layer[layer_index][left] for left, right in zip(positions, positions[1:])],
            axis=0,
        )
        process_var = np.var(diffs, axis=0)
        denom = measurement_var + process_var
        beta_by_layer[layer_index] = np.divide(measurement_var, denom, out=np.zeros_like(measurement_var), where=denom > 0.0)
    return means_by_layer, beta_by_layer, layer_names


def validate_positions(means_by_layer: dict[int, dict[int, np.ndarray]]) -> None:
    missing: list[str] = []
    for layer_index, by_position in means_by_layer.items():
        absent = [position for position in UPDATE_POSITIONS if position not in by_position]
        if absent:
            missing.append(f"layer {layer_index}: missing update positions {absent[:8]}{'...' if len(absent) > 8 else ''}")
    if missing:
        raise RuntimeError("M-PRED requires 100-position activation evidence; " + "; ".join(missing[:4]))


def mpred_selection(
    means_by_position: dict[int, np.ndarray],
    beta: np.ndarray,
    *,
    alpha: float,
    budget_fraction: float,
) -> tuple[list[int], list[dict[str, Any]]]:
    channel_count = len(means_by_position[100])
    budget_count = max(1, math.ceil(channel_count * budget_fraction))
    state = means_by_position[100].copy()
    previous_state = state.copy()
    steps: list[dict[str, Any]] = []
    selected = sorted(top_channels(state, budget_count))
    for position in UPDATE_POSITIONS:
        observed = means_by_position[position]
        innovation = observed - previous_state
        predicted = alpha * state + (1.0 - alpha) * observed + beta * innovation
        previous_state = state
        state = predicted
        selected = sorted(top_channels(state, budget_count))
        if position % 1000 == 0 or position in {100, checker.SCORING_POSITION}:
            steps.append(
                {
                    "position": position,
                    "protected_count": len(selected),
                    "protected_channels": selected,
                    "mean_score": float(np.mean(state)),
                    "mean_beta": float(np.mean(beta)),
                }
            )
    return selected, steps


def random_walk_selection(channel_count: int, budget_count: int, rng: random.Random) -> tuple[list[int], list[dict[str, Any]]]:
    steps: list[dict[str, Any]] = []
    selected = sorted(rng.sample(range(channel_count), budget_count))
    for position in UPDATE_POSITIONS:
        selected = sorted(rng.sample(range(channel_count), budget_count))
        if position % 1000 == 0 or position in {100, checker.SCORING_POSITION}:
            steps.append({"position": position, "protected_count": len(selected), "protected_channels": selected})
    return selected, steps


def random_alpha_selection(
    means_by_position: dict[int, np.ndarray],
    beta: np.ndarray,
    rng: random.Random,
) -> tuple[list[int], list[dict[str, Any]], dict[str, float]]:
    channel_count = len(means_by_position[100])
    alphas = np.asarray([rng.uniform(0.5, 0.99) for _ in range(channel_count)], dtype=np.float64)
    budget_count = max(1, math.ceil(channel_count * 0.05))
    state = means_by_position[100].copy()
    previous_state = state.copy()
    steps: list[dict[str, Any]] = []
    selected = sorted(top_channels(state, budget_count))
    for position in UPDATE_POSITIONS:
        observed = means_by_position[position]
        innovation = observed - previous_state
        predicted = alphas * state + (1.0 - alphas) * observed + beta * innovation
        previous_state = state
        state = predicted
        selected = sorted(top_channels(state, budget_count))
        if position % 1000 == 0 or position in {100, checker.SCORING_POSITION}:
            steps.append({"position": position, "protected_count": len(selected), "protected_channels": selected})
    return selected, steps, {"mean_alpha": float(np.mean(alphas)), "min_alpha": float(np.min(alphas)), "max_alpha": float(np.max(alphas))}


def build_protected_sets(source_sets: dict[str, Any], rows: list[dict[str, Any]], seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    means_by_layer, beta_by_layer, layer_names = activation_statistics(rows)
    validate_positions(means_by_layer)
    protected_layers: dict[str, dict[str, Any]] = {}
    trajectory_layers: dict[str, dict[str, Any]] = {}
    for regime in ["static_1pct", "m11b_top5", "m11b_top10", "static_top10"]:
        protected_layers[regime] = source_sets["regimes"][regime]["layers"]
    rng = random.Random(seed)
    beta_summary: dict[str, Any] = {}
    for layer_index in sorted(means_by_layer):
        channel_count = len(means_by_layer[layer_index][100])
        beta = beta_by_layer[layer_index]
        beta_summary[str(layer_index)] = {
            "layer_name": layer_names[layer_index],
            "mean_beta": float(np.mean(beta)),
            "min_beta": float(np.min(beta)),
            "max_beta": float(np.max(beta)),
        }
        for regime, alpha in MPRED_ALPHA_BY_REGIME.items():
            protected_layers.setdefault(regime, {})
            trajectory_layers.setdefault(regime, {})
            fraction = MPRED_BUDGET_BY_REGIME[regime]
            selected, steps = mpred_selection(means_by_layer[layer_index], beta, alpha=alpha, budget_fraction=fraction)
            protected_layers[regime][str(layer_index)] = {
                "layer_name": layer_names[layer_index],
                "channel_count": channel_count,
                "budget_fraction": fraction,
                "protected_count": len(selected),
                "protected_channels": selected,
                "alpha": alpha,
                "source": f"mpred_endpoint_alpha_{alpha}_top_{int(fraction * 100)}pct",
            }
            trajectory_layers[regime][str(layer_index)] = {
                "layer_name": layer_names[layer_index],
                "channel_count": channel_count,
                "budget_fraction": fraction,
                "alpha": alpha,
                "steps_recorded": steps,
                "final_protected_channels": selected,
            }

        random_count = max(1, math.ceil(channel_count * 0.05))
        selected, steps = random_walk_selection(channel_count, random_count, rng)
        protected_layers.setdefault("random_walk_top5", {})[str(layer_index)] = {
            "layer_name": layer_names[layer_index],
            "channel_count": channel_count,
            "budget_fraction": 0.05,
            "protected_count": len(selected),
            "protected_channels": selected,
            "source": "seeded_random_walk_matched_top5_final_snapshot",
        }
        trajectory_layers.setdefault("random_walk_top5", {})[str(layer_index)] = {
            "layer_name": layer_names[layer_index],
            "channel_count": channel_count,
            "budget_fraction": 0.05,
            "steps_recorded": steps,
            "final_protected_channels": selected,
        }

        selected, steps, alpha_summary = random_alpha_selection(means_by_layer[layer_index], beta, rng)
        protected_layers.setdefault("mpred_random_alpha_top5", {})[str(layer_index)] = {
            "layer_name": layer_names[layer_index],
            "channel_count": channel_count,
            "budget_fraction": 0.05,
            "protected_count": len(selected),
            "protected_channels": selected,
            "alpha_source": "uniform_0_5_0_99_per_channel",
            **alpha_summary,
            "source": "mpred_random_alpha_top5_final_snapshot",
        }
        trajectory_layers.setdefault("mpred_random_alpha_top5", {})[str(layer_index)] = {
            "layer_name": layer_names[layer_index],
            "channel_count": channel_count,
            "budget_fraction": 0.05,
            "steps_recorded": steps,
            "final_protected_channels": selected,
            **alpha_summary,
        }

    protected_sets = {
        "schema_version": f"{checker.SCHEMA_VERSION}_protected_sets",
        "created_at_utc": shared.utc_now(),
        "selection_basis": "M-PRED endpoint protected sets from cached deterministic AIME-2025 activation magnitudes",
        "mpred_alphas": sorted(set(MPRED_ALPHA_BY_REGIME.values())),
        "beta_formula": "sigma_meas^2 / (sigma_meas^2 + sigma_proc^2)",
        "regimes": {regime: {"layers": protected_layers[regime]} for regime in checker.REGIMES if regime != "bf16"},
    }
    trajectories = {
        "schema_version": f"{checker.SCHEMA_VERSION}_protected_trajectories",
        "created_at_utc": shared.utc_now(),
        "update_cadence": checker.UPDATE_CADENCE,
        "update_positions": list(UPDATE_POSITIONS),
        "beta_summary_by_layer": beta_summary,
        "by_regime": {regime: {"layers": trajectory_layers.get(regime, {})} for regime in checker.RECOVERY_REGIMES},
        "protected_set_count_stats": m11b_runner.summarize_protected_counts(protected_sets),
    }
    return protected_sets, trajectories


def read_score_cache(run_dir: Path, regime: str, prompt_indices: set[int]) -> dict[int, dict[str, float]]:
    cached = m11_runner.read_score_cache_any(run_dir, regime, expected_prompt_indices=prompt_indices)
    if cached is None:
        raise FileNotFoundError(f"missing score cache for {regime} in {run_dir}")
    return cached


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


def build_per_trace_rows(prompts: list[dict[str, Any]], all_scores: dict[str, dict[int, dict[str, float]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    completed_regimes = list(all_scores)
    recovery_regimes = [regime for regime in completed_regimes if regime not in {"bf16", "static_1pct"}]
    for prompt in prompts:
        index = int(prompt["index"])
        perplexities = {regime: float(all_scores[regime][index]["perplexity"]) for regime in completed_regimes}
        mean_nll = {regime: float(all_scores[regime][index]["mean_nll"]) for regime in completed_regimes}
        static_gap = perplexities["static_1pct"] - perplexities["bf16"]
        no_gap = static_gap <= 0.0
        recoveries = {
            regime: None if no_gap else 1.0 - (perplexities[regime] - perplexities["bf16"]) / static_gap
            for regime in recovery_regimes
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
        "bootstrap_ci95": checker.bootstrap_median(values),
        "included_trace_count": len(values),
    }


def build_metrics(
    run_dir: Path,
    prompt_manifest: dict[str, Any],
    model_provenance: dict[str, Any],
    rows: list[dict[str, Any]],
    protected_trajectories: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    included = [row for row in rows if not bool(row["no_recoverable_static_gap"])]
    completed_regimes = list(rows[0]["perplexities"]) if rows else []
    recovery_regimes = [regime for regime in completed_regimes if regime not in {"bf16", "static_1pct"}]
    summaries = {
        regime: summarize([float(row["recoveries"][regime]) for row in included])
        for regime in recovery_regimes
    }
    no_gap_count = len(rows) - len(included)
    for summary in summaries.values():
        summary["total_trace_count"] = len(rows)
        summary["no_recoverable_static_gap_count"] = no_gap_count
        summary["no_recoverable_static_gap_fraction"] = no_gap_count / len(rows) if rows else 0.0
    model_key = checker.infer_model_key(str(model_provenance.get("model_id")))
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
        "no_recoverable_static_gap_fraction": no_gap_count / len(rows) if rows else 0.0,
        "scoring_position": checker.SCORING_POSITION,
        "scoring_window_tokens": checker.SCORING_WINDOW_TOKENS,
        "metric_name": "positive-static-1pct-gap per-trace recovery",
        "metric_formula": "1 - (perplexity_regime - perplexity_BF16) / (perplexity_static_1pct - perplexity_BF16)",
        "completed_regimes": completed_regimes,
        "partial_packet": set(completed_regimes) != set(checker.REGIMES),
        "results_by_regime": summaries,
        "thresholds": checker.THRESHOLDS,
        "protected_set_count_stats": protected_trajectories.get("protected_set_count_stats", {}),
        "artifacts": {"run_dir": str(run_dir)},
    }
    bootstrap = {
        "schema_version": f"{checker.SCHEMA_VERSION}_bootstrap_ci",
        "metric_name": metrics["metric_name"],
        "bootstrap_samples": checker.BOOTSTRAP_SAMPLES,
        "bootstrap_seed": checker.BOOTSTRAP_SEED,
        "results_by_regime": summaries,
    }
    best_mpred_name, best_mpred = checker.best_regime(summaries, checker.MPRED_REGIMES)
    best_m11b_name, best_m11b = checker.best_regime(summaries, ["m11b_top5", "m11b_top10"])
    controls = {
        "schema_version": f"{checker.SCHEMA_VERSION}_control_metrics",
        "created_at_utc": shared.utc_now(),
        "controls": {
            "static_1pct": {"median_recovery": 0.0},
            "static_top10": summaries.get("static_top10"),
            "random_walk_top5": summaries.get("random_walk_top5"),
            "mpred_random_alpha_top5": summaries.get("mpred_random_alpha_top5"),
        },
        "best_mpred_regime": best_mpred_name,
        "best_m11b_regime": best_m11b_name,
        "best_mpred_minus_best_m11b": (
            None
            if best_mpred is None or best_m11b is None
            else float(best_mpred["median_recovery"]) - float(best_m11b["median_recovery"])
        ),
    }
    return metrics, bootstrap, controls


def main(argv: list[str] | None = None) -> int:
    shared.SCHEMA_VERSION = checker.SCHEMA_VERSION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=sorted(DEFAULT_SOURCES), default="granite")
    parser.add_argument("--source-run-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--results-dir", type=Path, default=checker.RESULTS_DIR)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--new-regimes", nargs="+", choices=NEW_REGIMES, default=NEW_REGIMES)
    parser.add_argument("--seed", type=int, default=checker.BOOTSTRAP_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = parser.parse_args(argv)

    source_dir = (args.source_run_dir or DEFAULT_SOURCES[args.model_key]).resolve()
    run_id = args.run_id or f"om_phase9_mpred_{args.model_key}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = args.results_dir / run_id
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")
    if args.seed != checker.BOOTSTRAP_SEED:
        raise SystemExit(f"M-PRED preregisters bootstrap/random seed {checker.BOOTSTRAP_SEED}")
    ensure_source(source_dir)

    (run_dir / "logs").mkdir(parents=True)
    stdout_log = (run_dir / "logs/stdout.log").open("w", encoding="utf-8", buffering=1)
    stderr_log = (run_dir / "logs/stderr.log").open("w", encoding="utf-8", buffering=1)
    sys.stdout = shared.Tee(sys.__stdout__, stdout_log)
    sys.stderr = shared.Tee(sys.__stderr__, stderr_log)
    run_events_path = run_dir / "run_events.jsonl"
    previous_excepthook = sys.excepthook

    def mpred_excepthook(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        write_failure_packet(run_dir, run_events_path, exc)
        previous_excepthook(exc_type, exc, tb)

    sys.excepthook = mpred_excepthook
    random.seed(args.seed)
    run_events_path.write_text(
        json.dumps({"created_at_utc": shared.utc_now(), "event": "mpred_run_started", "model_key": args.model_key}, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    try:
        copy_source_artifacts(source_dir, run_dir)
        environment = shared.build_environment(schema_version=checker.SCHEMA_VERSION)
        phase4_runner.write_environment_text(run_dir / "environment.txt", environment)
        shared.write_json(run_dir / "environment.json", environment)
        prompt_manifest = load_json(run_dir / "prompt_manifest.json")
        model_provenance = load_json(run_dir / "model_provenance.json")
        expected = checker.MODEL_SPECS[args.model_key]
        if model_provenance.get("model_id") != expected["model_id"]:
            raise RuntimeError(f"source model mismatch: expected {expected['model_id']}")
        if model_provenance.get("hf_snapshot_commit") != expected["snapshot"]:
            raise RuntimeError("source model snapshot mismatch")
        shared.write_json(
            run_dir / "command_metadata.json",
            {
                "schema_version": f"{checker.SCHEMA_VERSION}_command",
                "created_at_utc": shared.utc_now(),
                "argv": sys.argv if argv is None else ["run_om_phase9_mpred.py", *argv],
                "cwd": str(Path.cwd()),
                "branch": "outlier_migrate_phase9_mpred",
                "run_dir": str(run_dir),
                "source_run_dir": str(source_dir),
                "model_key": args.model_key,
                "batch_size": args.batch_size,
                "new_regimes": args.new_regimes,
                "scope_reduction": None if list(args.new_regimes) == NEW_REGIMES else "bounded_subset_due_endpoint_scoring_throughput",
            },
        )
        shared.write_json(
            run_dir / "random_seed.json",
            {"schema_version": f"{checker.SCHEMA_VERSION}_random_seed", "seed": args.seed, "determinism": {"do_sample": False, "num_beams": 1}},
        )

        source_sets = load_json(source_dir / "protected_sets.json")
        protected_sets, protected_trajectories = build_protected_sets(
            source_sets,
            shared.iter_activation_rows(run_dir / "activation_magnitudes.jsonl.gz"),
            args.seed,
        )
        shared.write_json(run_dir / "protected_sets.json", protected_sets)
        shared.write_json(run_dir / "protected_trajectories.json", protected_trajectories)

        prompts = prompt_manifest["prompts"]
        prompt_indices = {int(row["index"]) for row in prompts}
        target_tokens = phase4_runner.load_trace_tokens(run_dir / "bf16_traces.jsonl.gz")
        all_scores: dict[str, dict[int, dict[str, float]]] = {}
        excluded_by_regime: dict[str, Any] = {}
        for regime in BASELINE_REGIMES:
            all_scores[regime] = read_score_cache(source_dir, regime, prompt_indices)
            write_score_cache(run_dir, regime, all_scores[regime])
            if regime != "bf16":
                excluded_by_regime[regime] = {"regime": regime, "reused_score_cache": str(source_dir)}

        for regime in args.new_regimes:
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
        per_trace_rows = build_per_trace_rows(prompts, all_scores)
        shared.write_json(
            run_dir / "per_trace_metrics.json",
            {"schema_version": f"{checker.SCHEMA_VERSION}_per_trace_metrics", "created_at_utc": shared.utc_now(), "traces": per_trace_rows},
        )
        metrics, bootstrap, controls = build_metrics(run_dir, prompt_manifest, model_provenance, per_trace_rows, protected_trajectories)
        shared.write_json(run_dir / "metrics.json", metrics)
        shared.write_json(run_dir / "bootstrap_ci.json", bootstrap)
        shared.write_json(run_dir / "control_metrics.json", controls)
        run_events_path.open("a", encoding="utf-8").write(json.dumps({"created_at_utc": shared.utc_now(), "event": "mpred_run_completed"}, sort_keys=True) + "\n")
        shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=checker.SCHEMA_VERSION))
        result = checker.evaluate(run_dir)
        print(json.dumps({"run_dir": str(run_dir), "checker_decision": result["decision"], "details": result.get("decision_details")}, indent=2, sort_keys=True))
        shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=checker.SCHEMA_VERSION))
        checker.evaluate(run_dir)
        sys.excepthook = previous_excepthook
        return 0 if result["decision"] != checker.FAIL_INFRA else 1
    except BaseException as exc:
        write_failure_packet(run_dir, run_events_path, exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
