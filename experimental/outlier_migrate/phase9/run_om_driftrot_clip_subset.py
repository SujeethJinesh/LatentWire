#!/usr/bin/env python3
"""Run a preregistered Granite DriftRot clip/CVaR subset screen."""

from __future__ import annotations

import argparse
import json
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
from experimental.outlier_migrate.phase9 import check_om_paroquant_baseline as checker
from experimental.outlier_migrate.phase9 import run_om_paroquant_baseline as paro_runner
from experimental.outlier_migrate.phase9 import run_om_phase9_m11_ema_drift as m11_runner
from experimental.outlier_migrate.phase9 import run_om_phase9_m2_position_conditional as m2_runner
from experimental.shared import run_phase0_branch as shared


SCHEMA_VERSION = "om_driftrot_clip_subset_v1"
PREREG_PATH = ROOT / "experimental/outlier_migrate/phase9/preregister_om_driftrot_granite_clip_cvar.md"
DEFAULT_BASE_RUN_DIR = ROOT / "experimental/outlier_migrate/phase9/results/om_paroquant_granite_small_20260520T1555Z"
DEFAULT_RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase9/results"


def parse_indices(raw: str) -> list[int]:
    indices = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not indices:
        raise argparse.ArgumentTypeError("at least one prompt index is required")
    if len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("prompt indices must be unique")
    return indices


def write_failure_packet(run_dir: Path, run_events_path: Path, exc: BaseException | str) -> None:
    payload: dict[str, Any] = {
        "schema_version": f"{SCHEMA_VERSION}_infra_error",
        "created_at_utc": shared.utc_now(),
        "decision": "FAIL_INFRA_DRIFTROT_CLIP_SUBSET",
        "reason": str(exc),
    }
    if isinstance(exc, BaseException):
        payload["exception_type"] = type(exc).__name__
        payload["traceback"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    run_events_path.open("a", encoding="utf-8").write(
        json.dumps({"created_at_utc": shared.utc_now(), "event": "run_failed", "reason": str(exc)}, sort_keys=True) + "\n"
    )
    shared.write_json(run_dir / "infra_error.json", payload)
    shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "median_recovery": float(median(values)) if values else None,
        "mean_recovery": float(mean(values)) if values else None,
        "bootstrap_ci95": checker.bootstrap_median(values),
        "included_trace_count": len(values),
        "per_trace_recovery_included": values,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"om_driftrot_clip_subset_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--prompt-indices", type=parse_indices, required=True)
    parser.add_argument("--split-name", choices=["calibration", "confirmation", "diagnostic"], required=True)
    parser.add_argument("--base-run-dir", type=Path, default=DEFAULT_BASE_RUN_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=checker.BOOTSTRAP_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--num-rotations", type=int, default=8)
    parser.add_argument("--row-chunk", type=int, default=256)
    parser.add_argument("--scale-clip-min", type=float, required=True)
    parser.add_argument("--scale-clip-max", type=float, required=True)
    args = parser.parse_args(argv)

    shared.SCHEMA_VERSION = SCHEMA_VERSION
    random.seed(args.seed)
    run_dir = args.results_dir / args.run_id
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")
    (run_dir / "logs").mkdir(parents=True)
    stdout_log = (run_dir / "logs/stdout.log").open("w", encoding="utf-8", buffering=1)
    stderr_log = (run_dir / "logs/stderr.log").open("w", encoding="utf-8", buffering=1)
    sys.stdout = shared.Tee(sys.__stdout__, stdout_log)
    sys.stderr = shared.Tee(sys.__stderr__, stderr_log)
    run_events_path = run_dir / "run_events.jsonl"
    run_events_path.write_text(
        json.dumps({"created_at_utc": shared.utc_now(), "event": "run_started"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    try:
        base_run_dir = args.base_run_dir.resolve()
        prompt_manifest, prompt_reasons = paro_runner.build_prompt_manifest(checker.DEFAULT_PROMPT_FILE)
        if prompt_reasons:
            raise RuntimeError(f"canonical prompt manifest failed validation: {prompt_reasons}")
        selected_indices = set(args.prompt_indices)
        selected_prompts = [row for row in prompt_manifest["prompts"] if int(row["index"]) in selected_indices]
        if [int(row["index"]) for row in selected_prompts] != args.prompt_indices:
            raise RuntimeError("selected prompt indices are not present in canonical order")
        prompt_manifest["prompts"] = selected_prompts
        prompt_manifest["prompt_count"] = len(selected_prompts)
        prompt_manifest["selection"] = f"driftrot_clip_{args.split_name}_subset"
        prompt_manifest["selected_prompt_indices"] = args.prompt_indices
        prompt_manifest["full_prompt_file_sha256"] = prompt_manifest["prompt_file_sha256"]
        prompt_manifest["prompt_sha256"] = checker.prompt_payload_sha256(selected_prompts)

        environment = shared.build_environment(schema_version=SCHEMA_VERSION)
        model_provenance = m2_runner.resolve_model_snapshot_light(checker.MODEL_ID)
        model_provenance["schema_version"] = f"{SCHEMA_VERSION}_model_provenance"
        if model_provenance.get("hf_snapshot_commit") != checker.MODEL_SNAPSHOT:
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
                "argv": sys.argv if argv is None else ["run_om_driftrot_clip_subset.py", *argv],
                "cwd": str(Path.cwd()),
                "run_dir": str(run_dir),
                "base_run_dir": str(base_run_dir),
            },
        )
        shared.write_json(run_dir / "random_seed.json", {"schema_version": f"{SCHEMA_VERSION}_random_seed", "seed": args.seed})
        shared.write_json(
            run_dir / "decoding_config.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_decoding_config",
                "scoring_position": checker.SCORING_POSITION,
                "scoring_window_tokens": checker.SCORING_WINDOW_TOKENS,
                "selected_prompt_indices": args.prompt_indices,
            },
        )
        scale_clip = (float(args.scale_clip_min), float(args.scale_clip_max))
        shared.write_json(
            run_dir / "config.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_config",
                "experiment_id": "driftrot_granite_clip_cvar",
                "candidate_id": args.candidate_id,
                "split_name": args.split_name,
                "prompt_indices": args.prompt_indices,
                "preregistration": str(PREREG_PATH.relative_to(ROOT)),
                "base_run_dir": str(base_run_dir),
                "group_size": args.group_size,
                "num_rotations": args.num_rotations,
                "scale_clip": list(scale_clip),
            },
        )
        command = " ".join(sys.argv)
        (run_dir / "command.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n", encoding="utf-8")
        (run_dir / "command.sh").chmod(0o755)
        shared.write_json(
            run_dir / "traces_used.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_traces_used",
                "split_name": args.split_name,
                "prompt_indices": args.prompt_indices,
                "prompt_ids": [row["prompt_id"] for row in selected_prompts],
            },
        )

        target_tokens_all = phase4_runner.load_trace_tokens(base_run_dir / "bf16_traces.jsonl.gz")
        target_tokens = {index: target_tokens_all[index] for index in args.prompt_indices}
        all_scores: dict[str, dict[int, dict[str, float]]] = {}
        for regime in ["bf16", "static_1pct"]:
            cached = m11_runner.read_score_cache_any(base_run_dir, regime, expected_prompt_indices=selected_indices)
            if cached is None:
                raise RuntimeError(f"missing reusable {regime} score cache in {base_run_dir}")
            all_scores[regime] = {index: cached[index] for index in args.prompt_indices}
            paro_runner.write_score_cache(run_dir, regime, all_scores[regime])

        all_scores["paroquant_w4a16"], excluded = paro_runner.score_paroquant_regime(
            model_provenance=model_provenance,
            prompts=selected_prompts,
            target_tokens=target_tokens,
            batch_size=args.batch_size,
            dtype_name=args.dtype,
            device_name=args.device,
            run_events_path=run_events_path,
            group_size=args.group_size,
            num_rotations=args.num_rotations,
            row_chunk=args.row_chunk,
            scale_clip=scale_clip,
        )
        paro_runner.write_score_cache(run_dir, "paroquant_w4a16", all_scores["paroquant_w4a16"])
        shared.write_json(run_dir / "excluded_tensors.json", {"schema_version": f"{SCHEMA_VERSION}_excluded_tensors", "by_regime": {"paroquant_w4a16": excluded}})

        rows: list[dict[str, Any]] = []
        for prompt in selected_prompts:
            index = int(prompt["index"])
            perplexities = {regime: float(all_scores[regime][index]["perplexity"]) for regime in checker.REGIMES}
            mean_nll = {regime: float(all_scores[regime][index]["mean_nll"]) for regime in checker.REGIMES}
            static_gap = perplexities["static_1pct"] - perplexities["bf16"]
            no_gap = static_gap <= 0.0
            recovery = None if no_gap else 1.0 - (perplexities["paroquant_w4a16"] - perplexities["bf16"]) / static_gap
            rows.append(
                {
                    "prompt_index": index,
                    "prompt_id": prompt["prompt_id"],
                    "split_name": args.split_name,
                    "perplexities": perplexities,
                    "mean_nll": mean_nll,
                    "static_gap": float(static_gap),
                    "no_recoverable_static_gap": bool(no_gap),
                    "recoveries": {"paroquant_w4a16": recovery},
                    "scored_tokens": int(all_scores["bf16"][index]["scored_tokens"]),
                    "score_start": int(all_scores["bf16"][index]["score_start"]),
                    "score_end": int(all_scores["bf16"][index]["score_end"]),
                }
            )
        included = [row for row in rows if not row["no_recoverable_static_gap"]]
        values = [float(row["recoveries"]["paroquant_w4a16"]) for row in included]
        summary = summarize(values)
        summary.update(
            {
                "total_trace_count": len(rows),
                "no_recoverable_static_gap_count": len(rows) - len(included),
                "no_recoverable_static_gap_fraction": (len(rows) - len(included)) / len(rows) if rows else 0.0,
            }
        )
        metrics = {
            "schema_version": f"{SCHEMA_VERSION}_metrics",
            "created_at_utc": shared.utc_now(),
            "candidate_id": args.candidate_id,
            "split_name": args.split_name,
            "prompt_indices": args.prompt_indices,
            "model_id": model_provenance.get("model_id"),
            "model_snapshot_commit": model_provenance.get("hf_snapshot_commit"),
            "metric_name": "positive-static-1pct-gap per-trace recovery",
            "results_by_regime": {"paroquant_w4a16": summary},
        }
        shared.write_json(run_dir / "per_trace_metrics.json", {"schema_version": f"{SCHEMA_VERSION}_per_trace_metrics", "created_at_utc": shared.utc_now(), "traces": rows})
        shared.write_json(run_dir / "metrics.json", metrics)
        shared.write_json(run_dir / "decision.json", {"schema_version": f"{SCHEMA_VERSION}_decision", "decision": "SUBSET_SCORED_PENDING_SPLIT_ANALYSIS", "candidate_id": args.candidate_id})
        run_events_path.open("a", encoding="utf-8").write(json.dumps({"created_at_utc": shared.utc_now(), "event": "run_completed"}, sort_keys=True) + "\n")
        shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
        print(json.dumps({"run_dir": str(run_dir), "results_by_regime": metrics["results_by_regime"]}, indent=2, sort_keys=True))
        return 0
    except BaseException as exc:
        write_failure_packet(run_dir, run_events_path, exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
