#!/usr/bin/env python3
"""Run Phase 9 M11b budget scaling."""

from __future__ import annotations

import argparse
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
from experimental.outlier_migrate.phase9 import check_om_phase9_m11b_budget_scaling as checker
from experimental.outlier_migrate.phase9 import run_om_phase9_m11_ema_drift as m11_runner
from experimental.outlier_migrate.phase9 import run_om_phase9_m2_position_conditional as m2_runner
from experimental.shared import run_phase0_branch as shared


DEFAULT_RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase9/results"
DEFAULT_PROMPT_FILE = checker.DEFAULT_PROMPT_FILE
DEFAULT_MODEL_ID = checker.MODEL_ID
SCHEMA_VERSION = checker.SCHEMA_VERSION
UPDATE_POSITIONS = tuple(range(checker.UPDATE_CADENCE, checker.SCORING_POSITION + 1, checker.UPDATE_CADENCE))


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


def build_budget_trajectories(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    means_by_layer, layer_names = activation_means_by_layer_position(rows)
    missing: list[str] = []
    for layer_index, by_position in means_by_layer.items():
        absent = [position for position in UPDATE_POSITIONS if position not in by_position]
        if absent:
            missing.append(f"layer {layer_index}: missing update positions {absent[:8]}{'...' if len(absent) > 8 else ''}")
    if missing:
        raise RuntimeError("M11b requires 100-position activation evidence; " + "; ".join(missing[:4]))

    protected_sets: dict[str, Any] = {
        "schema_version": f"{SCHEMA_VERSION}_protected_sets",
        "created_at_utc": shared.utc_now(),
        "selection_basis": "EMA-smoothed mean absolute layer output activation over deterministic AIME-2025 traces 0-11",
        "alpha": checker.ALPHA,
        "regimes": {},
    }
    trajectories: dict[str, Any] = {
        "schema_version": f"{SCHEMA_VERSION}_protected_trajectories",
        "created_at_utc": shared.utc_now(),
        "alpha": checker.ALPHA,
        "budget_fractions": list(checker.BUDGET_FRACTIONS),
        "update_cadence": checker.UPDATE_CADENCE,
        "update_positions": list(UPDATE_POSITIONS),
        "by_regime": {},
    }
    layers_by_regime: dict[str, dict[str, Any]] = {regime: {} for regime in checker.REGIMES if regime != "bf16"}
    trajectories_by_regime: dict[str, dict[str, Any]] = {
        regime: {} for regime in ["m11b_top1", "m11b_top5", "m11b_top10"]
    }

    for layer_index in sorted(means_by_layer):
        values_100 = means_by_layer[layer_index][100]
        channel_count = len(values_100)
        top1_count = max(1, math.ceil(channel_count * 0.01))
        top10_count = max(1, math.ceil(channel_count * 0.10))
        static_1 = sorted(top_channels(values_100, top1_count))
        static_10 = sorted(top_channels(values_100, top10_count))
        layers_by_regime["static_1pct"][str(layer_index)] = {
            "layer_name": layer_names[layer_index],
            "channel_count": channel_count,
            "protected_count": len(static_1),
            "protected_channels": static_1,
            "source": "position_100_top1",
        }
        layers_by_regime["static_top10"][str(layer_index)] = {
            "layer_name": layer_names[layer_index],
            "channel_count": channel_count,
            "protected_count": len(static_10),
            "protected_channels": static_10,
            "source": "position_100_top10",
        }
        for fraction, regime in [(0.01, "m11b_top1"), (0.05, "m11b_top5"), (0.10, "m11b_top10")]:
            budget_count = max(1, math.ceil(channel_count * fraction))
            scores = [0.0] * channel_count
            for channel in top_channels(values_100, budget_count):
                scores[channel] = 1.0
            steps: list[dict[str, Any]] = []
            selected = sorted(top_channels(scores, budget_count))
            for position in UPDATE_POSITIONS:
                indicator_channels = set(top_channels(means_by_layer[layer_index][position], budget_count))
                scores = [
                    checker.ALPHA * (1.0 if channel in indicator_channels else 0.0)
                    + (1.0 - checker.ALPHA) * float(scores[channel])
                    for channel in range(channel_count)
                ]
                selected = sorted(top_channels(scores, budget_count))
                if position % 1000 == 0 or position in {100, checker.SCORING_POSITION}:
                    steps.append(
                        {
                            "position": position,
                            "protected_count": len(selected),
                            "protected_channels": selected,
                            "mean_score": float(mean(scores)),
                        }
                    )
            layers_by_regime[regime][str(layer_index)] = {
                "layer_name": layer_names[layer_index],
                "channel_count": channel_count,
                "budget_fraction": fraction,
                "protected_count": len(selected),
                "protected_channels": selected,
                "source": f"ema_alpha_0_3_top_{int(fraction * 100)}pct_final_snapshot",
            }
            trajectories_by_regime[regime][str(layer_index)] = {
                "layer_name": layer_names[layer_index],
                "channel_count": channel_count,
                "budget_fraction": fraction,
                "steps_recorded": steps,
                "final_protected_channels": selected,
            }

    protected_sets["regimes"] = {regime: {"layers": layers} for regime, layers in layers_by_regime.items()}
    trajectories["by_regime"] = {regime: {"layers": layers} for regime, layers in trajectories_by_regime.items()}
    trajectories["protected_set_count_stats"] = summarize_protected_counts(protected_sets)
    return protected_sets, trajectories


def summarize_protected_counts(protected_sets: dict[str, Any]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for regime, spec in protected_sets["regimes"].items():
        counts = [int(layer["protected_count"]) for layer in spec["layers"].values()]
        stats[regime] = {
            "min": min(counts),
            "median": float(median(counts)),
            "max": max(counts),
            "mean": float(mean(counts)),
        }
    return stats


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
    protected_trajectories: dict[str, Any],
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
        "alpha": checker.ALPHA,
        "budget_fractions": list(checker.BUDGET_FRACTIONS),
        "metric_name": "positive-static-1pct-gap per-trace recovery",
        "metric_formula": "1 - (perplexity_regime - perplexity_BF16) / (perplexity_static_1pct - perplexity_BF16)",
        "results_by_regime": summaries,
        "thresholds": checker.THRESHOLDS,
        "protected_set_count_stats": protected_trajectories.get("protected_set_count_stats", {}),
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
            "static_1pct": {"median_recovery": 0.0},
            "static_top10": summaries["static_top10"],
        },
        "m11b_top5_minus_top1_median": (
            None
            if summaries["m11b_top5"]["median_recovery"] is None or summaries["m11b_top1"]["median_recovery"] is None
            else float(summaries["m11b_top5"]["median_recovery"]) - float(summaries["m11b_top1"]["median_recovery"])
        ),
        "m11b_top10_minus_top1_median": (
            None
            if summaries["m11b_top10"]["median_recovery"] is None or summaries["m11b_top1"]["median_recovery"] is None
            else float(summaries["m11b_top10"]["median_recovery"]) - float(summaries["m11b_top1"]["median_recovery"])
        ),
        "m11b_top10_minus_static_top10_median": (
            None
            if summaries["m11b_top10"]["median_recovery"] is None or summaries["static_top10"]["median_recovery"] is None
            else float(summaries["m11b_top10"]["median_recovery"]) - float(summaries["static_top10"]["median_recovery"])
        ),
    }
    return metrics, bootstrap, controls


def main(argv: list[str] | None = None) -> int:
    shared.SCHEMA_VERSION = SCHEMA_VERSION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"om_phase9_m11b_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "experimental/outlier_migrate/phase9/results")
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
        raise SystemExit(f"M11b primary gate requires {checker.MODEL_ID}")
    if args.prompt_file.resolve() != DEFAULT_PROMPT_FILE.resolve():
        raise SystemExit(f"M11b requires canonical prompt file {DEFAULT_PROMPT_FILE}")
    if shared.file_sha256(args.prompt_file) != checker.EXPECTED_PROMPT_FILE_SHA256:
        raise SystemExit("canonical AIME-2025 indices 0-23 prompt file hash drifted")
    if args.seed != checker.BOOTSTRAP_SEED:
        raise SystemExit("M11b preregisters bootstrap/random seed 20260527")

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

    def m11b_excepthook(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        write_failure_packet(run_dir, run_events_path, exc)
        previous_excepthook(exc_type, exc, tb)

    sys.excepthook = m11b_excepthook
    run_events_path.write_text(
        json.dumps({"created_at_utc": shared.utc_now(), "event": "run_started"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    random.seed(args.seed)

    try:
        prompt_manifest, prompt_reasons = build_prompt_manifest(args.prompt_file)
        environment = shared.build_environment(schema_version=SCHEMA_VERSION)
        model_provenance = m2_runner.resolve_model_snapshot_light(args.model_id)
        model_provenance["schema_version"] = f"{SCHEMA_VERSION}_model_provenance"
        shared.write_json(run_dir / "prompt_manifest.json", prompt_manifest)
        shared.write_json(run_dir / "environment.json", environment)
        phase4_runner.write_environment_text(run_dir / "environment.txt", environment)
        shared.write_json(run_dir / "model_provenance.json", model_provenance)
        shared.write_json(
            run_dir / "command_metadata.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_command",
                "created_at_utc": shared.utc_now(),
                "argv": sys.argv if argv is None else ["run_om_phase9_m11b_budget_scaling.py", *argv],
                "cwd": str(Path.cwd()),
                "branch": "outlier_migrate_phase9_m11b_budget_scaling",
                "run_dir": str(run_dir),
                "batch_size": args.batch_size,
                "reuse_activation_run_dir": str(args.reuse_activation_run_dir) if args.reuse_activation_run_dir else None,
                "reuse_trace_run_dir": str(args.reuse_trace_run_dir) if args.reuse_trace_run_dir else None,
                "reuse_score_run_dir": str(args.reuse_score_run_dir) if args.reuse_score_run_dir else None,
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
                "max_new_tokens_for_calibration": checker.SCORING_POSITION,
                "do_sample": False,
                "num_beams": 1,
                "scoring_position": checker.SCORING_POSITION,
                "scoring_window_tokens": checker.SCORING_WINDOW_TOKENS,
                "update_cadence": checker.UPDATE_CADENCE,
                "update_positions": list(UPDATE_POSITIONS),
            },
        )
        shared.write_json(
            run_dir / "quantization_config.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_quantization_config",
                "weight_bits": 4,
                "scheme": "symmetric_per_output_channel_int4",
                "signed_integer_range": [-7, 7],
                "activation_dtype": "float16",
                "protected_channel_dtype": "bfloat16",
                "implementation_note": "M11b scores final EMA protected snapshots at position 10000 for budget levels 1%, 5%, and 10%.",
            },
        )
        if prompt_reasons:
            shared.write_json(run_dir / "infra_error.json", {"reasons": prompt_reasons})
            shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
            checker.evaluate(run_dir)
            return 1
        if not model_provenance.get("snapshot_path") or model_provenance.get("hf_snapshot_commit") != checker.MODEL_SNAPSHOT:
            shared.write_json(run_dir / "infra_error.json", {"reasons": ["model snapshot missing or mismatch"]})
            shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
            checker.evaluate(run_dir)
            return 1

        prompts = prompt_manifest["prompts"]
        prompt_indices = {int(row["index"]) for row in prompts}
        activation_path = run_dir / "activation_magnitudes.jsonl.gz"
        trace_path = run_dir / "bf16_traces.jsonl.gz"
        if args.reuse_activation_run_dir:
            m11_runner.filter_activation_rows = getattr(m11_runner, "filter_activation_rows", None)
            # Reuse the full M11 activation artifact because it already contains
            # every 100-position update required by M11b.
            source = args.reuse_activation_run_dir.resolve() / "activation_magnitudes.jsonl.gz"
            activation_path.write_bytes(source.read_bytes())
            shared.write_json(
                run_dir / "activation_magnitude_manifest.json",
                {
                    "schema_version": f"{SCHEMA_VERSION}_activation_magnitude_manifest",
                    "created_at_utc": shared.utc_now(),
                    "artifact": "activation_magnitudes.jsonl.gz",
                    "artifact_sha256": shared.file_sha256(activation_path),
                    "source_artifact": str(source),
                    "source_artifact_sha256": shared.file_sha256(source),
                    "positions": list(UPDATE_POSITIONS),
                    "prompt_indices": sorted(prompt_indices),
                },
            )
        else:
            model, tokenizer, device = shared.load_model_and_tokenizer(model_provenance, dtype_name=args.dtype, device_name=args.device)
            activation_manifest = shared.capture_activation_magnitudes(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompts=prompts,
                positions=UPDATE_POSITIONS,
                max_new_tokens=checker.SCORING_POSITION,
                batch_size=args.batch_size,
                output_path=activation_path,
                run_events_path=run_events_path,
            )
            shared.write_json(run_dir / "activation_magnitude_manifest.json", activation_manifest)
            del model, tokenizer, device
            release_model_memory()

        protected_sets, protected_trajectories = build_budget_trajectories(list(shared.iter_activation_rows(activation_path)))
        shared.write_json(run_dir / "protected_sets.json", protected_sets)
        shared.write_json(run_dir / "protected_trajectories.json", protected_trajectories)

        if args.reuse_trace_run_dir:
            reuse_trace_dir = args.reuse_trace_run_dir.resolve()
            m2_runner.copy_filtered_jsonl_gz(reuse_trace_dir / "bf16_traces.jsonl.gz", trace_path, prompt_indices=prompt_indices)
            trace_manifest = json.loads((reuse_trace_dir / "bf16_trace_manifest.json").read_text(encoding="utf-8"))
            trace_manifest["created_at_utc"] = shared.utc_now()
            trace_manifest["source_run_dir"] = str(reuse_trace_dir)
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
        excluded_by_regime: dict[str, Any] = {}
        score_reuse_dir = args.reuse_score_run_dir.resolve() if args.reuse_score_run_dir else None
        for regime, source_regime in [("bf16", "bf16"), ("static_1pct", "static_1pct")]:
            cached = m11_runner.read_score_cache_any(score_reuse_dir, source_regime, expected_prompt_indices=prompt_indices)
            if cached is not None:
                all_scores[regime] = cached
                write_score_cache(run_dir, regime, all_scores[regime])
                if regime != "bf16":
                    excluded_by_regime[regime] = {"regime": regime, "reused_score_cache": str(score_reuse_dir)}
                continue
            model, tokenizer, device = shared.load_model_and_tokenizer(model_provenance, dtype_name=args.dtype, device_name=args.device)
            all_scores[regime] = phase4_runner.score_targets(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompts=prompts,
                target_tokens=target_tokens,
                max_new_tokens=checker.SCORING_POSITION,
                batch_size=args.batch_size,
                use_float16_autocast=regime != "bf16",
                run_events_path=run_events_path,
                regime_name=regime,
            )
            write_score_cache(run_dir, regime, all_scores[regime])
            del model, tokenizer, device
            release_model_memory()

        for regime in ["m11b_top1", "m11b_top5", "m11b_top10", "static_top10"]:
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
            prompt_manifest=prompt_manifest,
            model_provenance=model_provenance,
            per_trace_rows=per_trace_rows,
            activation_path=activation_path,
            bf16_trace_path=trace_path,
            protected_trajectories=protected_trajectories,
        )
        shared.write_json(run_dir / "metrics.json", metrics)
        shared.write_json(run_dir / "bootstrap_ci.json", bootstrap)
        shared.write_json(run_dir / "control_metrics.json", controls)
        run_events_path.open("a", encoding="utf-8").write(json.dumps({"created_at_utc": shared.utc_now(), "event": "run_completed"}, sort_keys=True) + "\n")
        print(json.dumps({"run_dir": str(run_dir), "results_by_regime": metrics["results_by_regime"]}, indent=2, sort_keys=True))
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
