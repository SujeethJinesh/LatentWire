#!/usr/bin/env python3
"""Run Phase 9 M26 stable-core protection."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.outlier_migrate.phase4 import run_om_phase4_intervention as phase4_runner
from experimental.outlier_migrate.phase9 import check_om_phase9_m26_stable_core as checker
from experimental.outlier_migrate.phase9 import run_om_phase9_m11_ema_drift as m11_runner
from experimental.outlier_migrate.phase9 import run_om_phase9_m11b_budget_scaling as m11b_runner
from experimental.outlier_migrate.phase9 import run_om_phase9_m2_position_conditional as m2_runner
from experimental.shared import run_phase0_branch as shared


SCHEMA_VERSION = checker.SCHEMA_VERSION
DEFAULT_RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase9/results"
DEFAULT_PROMPT_FILE = checker.DEFAULT_PROMPT_FILE
DEFAULT_STABLE_CORE_SOURCE = (
    ROOT / "experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z/activation_magnitudes.jsonl.gz"
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


def build_stable_core_sets(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    means_by_layer, layer_names = m11b_runner.activation_means_by_layer_position(rows)
    rng = random.Random(checker.BOOTSTRAP_SEED)
    required_positions = [100, 500, 1000, 5000, 10000, 20000]
    protected_layers = {regime: {} for regime in checker.REGIMES if regime != "bf16"}
    core_layers: dict[str, Any] = {}

    for layer_index in sorted(means_by_layer):
        by_position = means_by_layer[layer_index]
        absent = [position for position in required_positions if position not in by_position]
        if absent:
            raise RuntimeError(f"layer {layer_index} missing stable-core positions {absent}")
        channel_count = len(by_position[100])
        top1_count = max(1, math.ceil(channel_count * 0.01))
        top_sets = [
            set(m11b_runner.top_channels(by_position[position], top1_count))
            for position in required_positions
        ]
        core = sorted(set.intersection(*top_sets))
        current_top1 = sorted(m11b_runner.top_channels(by_position[checker.SCORING_POSITION], top1_count))
        union = sorted(set(core) | set(current_top1))
        random_channels = sorted(rng.sample(range(channel_count), k=len(core))) if core else []
        static_1 = sorted(m11b_runner.top_channels(by_position[100], top1_count))
        layer_payload = {
            "layer_name": layer_names[layer_index],
            "channel_count": channel_count,
            "top1_count": top1_count,
        }
        protected_layers["static_1pct"][str(layer_index)] = {
            **layer_payload,
            "protected_count": len(static_1),
            "protected_channels": static_1,
            "source": "position_100_top1",
        }
        protected_layers["m26_core"][str(layer_index)] = {
            **layer_payload,
            "protected_count": len(core),
            "protected_channels": core,
            "source": "intersection_top1_positions_100_500_1000_5000_10000_20000",
        }
        protected_layers["core_current_top1_union"][str(layer_index)] = {
            **layer_payload,
            "protected_count": len(union),
            "protected_channels": union,
            "source": "stable_core_union_position_10000_top1",
        }
        protected_layers["random_matched_core"][str(layer_index)] = {
            **layer_payload,
            "protected_count": len(random_channels),
            "protected_channels": random_channels,
            "source": "random_matched_stable_core_size_seed_20260530",
        }
        core_layers[str(layer_index)] = {
            **layer_payload,
            "stable_core_count": len(core),
            "stable_core_fraction_of_channels": len(core) / channel_count if channel_count else 0.0,
            "stable_core_channels": core,
            "current_top1_position_10000_channels": current_top1,
            "core_current_top1_union_count": len(union),
        }

    protected_sets = {
        "schema_version": f"{SCHEMA_VERSION}_protected_sets",
        "created_at_utc": shared.utc_now(),
        "selection_basis": "stable-core intersection of prompt-averaged top-1% sets across Phase 1 Granite-Small positions",
        "regimes": {regime: {"layers": layers} for regime, layers in protected_layers.items()},
    }
    core_total = sum(int(item["stable_core_count"]) for item in core_layers.values())
    channel_total = sum(int(item["channel_count"]) for item in core_layers.values())
    stable_core = {
        "schema_version": f"{SCHEMA_VERSION}_stable_core",
        "created_at_utc": shared.utc_now(),
        "source_activation_artifact": str(DEFAULT_STABLE_CORE_SOURCE),
        "positions": required_positions,
        "layers": core_layers,
        "total_channels": channel_total,
        "stable_core_channels": core_total,
        "stable_core_fraction_of_channels": core_total / channel_total if channel_total else 0.0,
        "stable_core_percent_of_channels": 100.0 * core_total / channel_total if channel_total else 0.0,
    }
    return protected_sets, stable_core


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
    stable_core: dict[str, Any],
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
        "included_trace_count": len(included),
        "no_recoverable_static_gap_count": no_gap_count,
        "no_recoverable_static_gap_fraction": no_gap_count / len(per_trace_rows) if per_trace_rows else 0.0,
        "scoring_position": checker.SCORING_POSITION,
        "scoring_window_tokens": checker.SCORING_WINDOW_TOKENS,
        "metric_name": "positive-static-1pct-gap per-trace recovery",
        "metric_formula": "1 - (perplexity_regime - perplexity_BF16) / (perplexity_static_1pct - perplexity_BF16)",
        "results_by_regime": summaries,
        "thresholds": checker.THRESHOLDS,
        "stable_core": {
            "total_channels": stable_core["total_channels"],
            "stable_core_channels": stable_core["stable_core_channels"],
            "stable_core_percent_of_channels": stable_core["stable_core_percent_of_channels"],
        },
    }
    bootstrap = {
        "schema_version": f"{SCHEMA_VERSION}_bootstrap_ci",
        "bootstrap_samples": checker.BOOTSTRAP_SAMPLES,
        "bootstrap_seed": checker.BOOTSTRAP_SEED,
        "results_by_regime": summaries,
    }
    controls = {
        "schema_version": f"{SCHEMA_VERSION}_control_metrics",
        "created_at_utc": shared.utc_now(),
        "controls": {
            "static_1pct": {"median_recovery": 0.0},
            "random_matched_core": summaries["random_matched_core"],
        },
        "m26_core_minus_static_1pct_median": summaries["m26_core"]["median_recovery"],
        "m26_core_minus_random_matched_median": (
            None
            if summaries["m26_core"]["median_recovery"] is None or summaries["random_matched_core"]["median_recovery"] is None
            else float(summaries["m26_core"]["median_recovery"]) - float(summaries["random_matched_core"]["median_recovery"])
        ),
        "core_union_minus_core_median": (
            None
            if summaries["core_current_top1_union"]["median_recovery"] is None or summaries["m26_core"]["median_recovery"] is None
            else float(summaries["core_current_top1_union"]["median_recovery"]) - float(summaries["m26_core"]["median_recovery"])
        ),
    }
    return metrics, bootstrap, controls


def main(argv: list[str] | None = None) -> int:
    shared.SCHEMA_VERSION = SCHEMA_VERSION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"om_phase9_m26_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--model-id", default=checker.MODEL_ID)
    parser.add_argument("--stable-core-source", type=Path, default=DEFAULT_STABLE_CORE_SOURCE)
    parser.add_argument("--reuse-trace-run-dir", type=Path)
    parser.add_argument("--reuse-score-run-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=checker.BOOTSTRAP_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = parser.parse_args(argv)

    if args.model_id != checker.MODEL_ID:
        raise SystemExit(f"M26 primary gate requires {checker.MODEL_ID}")
    if shared.file_sha256(args.prompt_file) != checker.EXPECTED_PROMPT_FILE_SHA256:
        raise SystemExit("canonical AIME-2025 prompt file hash drifted")
    if args.seed != checker.BOOTSTRAP_SEED:
        raise SystemExit("M26 preregisters bootstrap/random seed 20260530")

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

    def excepthook(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        write_failure_packet(run_dir, run_events_path, exc)
        previous_excepthook(exc_type, exc, tb)

    sys.excepthook = excepthook
    run_events_path.write_text(
        json.dumps({"created_at_utc": shared.utc_now(), "event": "run_started"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    random.seed(args.seed)

    try:
        prompt_manifest, prompt_reasons = m11b_runner.build_prompt_manifest(args.prompt_file)
        prompt_manifest["schema_version"] = f"{SCHEMA_VERSION}_prompt_manifest"
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
                "argv": sys.argv if argv is None else ["run_om_phase9_m26_stable_core.py", *argv],
                "cwd": str(Path.cwd()),
                "branch": "outlier_migrate_phase9_m26_stable_core",
                "run_dir": str(run_dir),
                "batch_size": args.batch_size,
                "reuse_trace_run_dir": str(args.reuse_trace_run_dir) if args.reuse_trace_run_dir else None,
                "reuse_score_run_dir": str(args.reuse_score_run_dir) if args.reuse_score_run_dir else None,
                "stable_core_source": str(args.stable_core_source),
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
                "signed_integer_range": [-7, 7],
                "activation_dtype": "float16",
                "protected_channel_dtype": "bfloat16",
                "implementation_note": "M26 protects stable-core channels from a precomputed calibration intersection.",
            },
        )
        if prompt_reasons:
            raise RuntimeError("; ".join(prompt_reasons))
        if model_provenance.get("hf_snapshot_commit") != checker.MODEL_SNAPSHOT:
            raise RuntimeError("model snapshot missing or mismatch")

        prompts = prompt_manifest["prompts"]
        prompt_indices = {int(row["index"]) for row in prompts}
        protected_sets, stable_core = build_stable_core_sets(list(shared.iter_activation_rows(args.stable_core_source)))
        stable_core["source_activation_artifact"] = str(args.stable_core_source)
        stable_core["source_activation_sha256"] = shared.file_sha256(args.stable_core_source)
        shared.write_json(run_dir / "protected_sets.json", protected_sets)
        shared.write_json(run_dir / "stable_core.json", stable_core)
        shared.write_json(
            run_dir / "activation_magnitude_manifest.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_activation_magnitude_manifest",
                "created_at_utc": shared.utc_now(),
                "source_artifact": str(args.stable_core_source),
                "source_artifact_sha256": shared.file_sha256(args.stable_core_source),
                "artifact_not_copied_reason": "source activation artifact is large; M26 packet records stable_core.json and source hash",
            },
        )

        trace_path = run_dir / "bf16_traces.jsonl.gz"
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
            release_model_memory(model, tokenizer, device)
        target_tokens = phase4_runner.load_trace_tokens(trace_path)

        all_scores: dict[str, dict[int, dict[str, float]]] = {}
        excluded_by_regime: dict[str, Any] = {}
        score_reuse_dir = args.reuse_score_run_dir.resolve() if args.reuse_score_run_dir else None
        for regime in ["bf16", "static_1pct"]:
            cached = m11_runner.read_score_cache_any(score_reuse_dir, regime, expected_prompt_indices=prompt_indices)
            if cached is not None:
                all_scores[regime] = cached
                write_score_cache(run_dir, regime, cached)
                if regime != "bf16":
                    excluded_by_regime[regime] = {"regime": regime, "reused_score_cache": str(score_reuse_dir)}
                continue
            if regime == "bf16":
                model, tokenizer, device = shared.load_model_and_tokenizer(model_provenance, dtype_name=args.dtype, device_name=args.device)
                all_scores[regime] = phase4_runner.score_targets(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    prompts=prompts,
                    target_tokens=target_tokens,
                    max_new_tokens=checker.SCORING_POSITION,
                    batch_size=args.batch_size,
                    use_float16_autocast=False,
                    run_events_path=run_events_path,
                    regime_name=regime,
                )
                release_model_memory(model, tokenizer, device)
            else:
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

        for regime in ["m26_core", "core_current_top1_union", "random_matched_core"]:
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
        shared.write_json(
            run_dir / "per_trace_metrics.json",
            {"schema_version": f"{SCHEMA_VERSION}_per_trace_metrics", "created_at_utc": shared.utc_now(), "traces": per_trace_rows},
        )
        metrics, bootstrap, controls = build_metrics(
            run_dir=run_dir,
            prompt_manifest=prompt_manifest,
            model_provenance=model_provenance,
            per_trace_rows=per_trace_rows,
            stable_core=stable_core,
        )
        shared.write_json(run_dir / "metrics.json", metrics)
        shared.write_json(run_dir / "bootstrap_ci.json", bootstrap)
        shared.write_json(run_dir / "control_metrics.json", controls)
        run_events_path.open("a", encoding="utf-8").write(
            json.dumps({"created_at_utc": shared.utc_now(), "event": "run_completed"}, sort_keys=True) + "\n"
        )
        print(json.dumps({"run_dir": str(run_dir), "results_by_regime": metrics["results_by_regime"]}, indent=2, sort_keys=True))
        sys.stdout.flush()
        sys.stderr.flush()
        shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
        result = checker.evaluate(run_dir)
        print(json.dumps({"checker_decision": result["decision"], "artifact_complete": result.get("artifact_complete", False)}, indent=2, sort_keys=True))
        sys.excepthook = previous_excepthook
        return 0
    except BaseException as exc:
        write_failure_packet(run_dir, run_events_path, exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
