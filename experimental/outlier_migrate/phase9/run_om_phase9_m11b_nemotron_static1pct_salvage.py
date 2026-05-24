#!/usr/bin/env python3
"""Salvage Nemotron M11b by rerunning only corrected static_1pct."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.outlier_migrate.phase4 import run_om_phase4_intervention as phase4_runner
from experimental.outlier_migrate.phase9 import check_om_phase9_m11b_nemotron_replication as checker
from experimental.outlier_migrate.phase9 import run_om_phase9_m11b_budget_scaling as base
from experimental.outlier_migrate.phase9 import run_om_phase9_m11b_nemotron_replication as nemotron_runner
from experimental.shared import run_phase0_branch as shared


SOURCE_DEFAULT = (
    ROOT
    / "experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_20260520T2013Z"
)
COPY_ARTIFACTS = [
    "activation_magnitude_manifest.json",
    "activation_magnitudes.jsonl.gz",
    "bf16_trace_manifest.json",
    "bf16_traces.jsonl.gz",
    "decoding_config.json",
    "model_provenance.json",
    "prompt_manifest.json",
    "protected_sets.json",
    "protected_trajectories.json",
    "quantization_config.json",
]
REUSED_SCORE_REGIMES = ["bf16", "m11b_top1", "m11b_top5", "m11b_top10", "static_top10"]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_source(source_run_dir: Path) -> None:
    missing = [rel for rel in COPY_ARTIFACTS if not (source_run_dir / rel).is_file()]
    missing += [
        f"score_cache/{regime}.json"
        for regime in REUSED_SCORE_REGIMES
        if not (source_run_dir / "score_cache" / f"{regime}.json").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"source packet is missing required artifacts: {missing}")


def copy_source_artifacts(source_run_dir: Path, run_dir: Path) -> None:
    for rel in COPY_ARTIFACTS:
        dest = run_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_run_dir / rel, dest)
    (run_dir / "score_cache").mkdir(parents=True, exist_ok=True)
    for regime in REUSED_SCORE_REGIMES:
        shutil.copy2(source_run_dir / "score_cache" / f"{regime}.json", run_dir / "score_cache" / f"{regime}.json")


def read_score_cache(run_dir: Path, regime: str) -> dict[int, dict[str, float]]:
    payload = load_json(run_dir / "score_cache" / f"{regime}.json")
    return {int(index): {key: float(value) for key, value in row.items()} for index, row in payload["scores"].items()}


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


def build_per_trace_rows(prompts: list[dict[str, Any]], all_scores: dict[str, dict[int, dict[str, float]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def main(argv: list[str] | None = None) -> int:
    nemotron_runner._patch_base()
    shared.SCHEMA_VERSION = checker.SCHEMA_VERSION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--run-id", default=f"om_phase9_m11b_nemotron_static1pct_salvage_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--results-dir", type=Path, default=checker.RESULTS_DIR)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=checker.BOOTSTRAP_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = parser.parse_args(argv)

    source_run_dir = args.source_run_dir.resolve()
    run_dir = args.results_dir / args.run_id
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")
    if args.seed != checker.BOOTSTRAP_SEED:
        raise SystemExit(f"Nemotron M11b preregisters bootstrap/random seed {checker.BOOTSTRAP_SEED}")
    ensure_source(source_run_dir)

    (run_dir / "logs").mkdir(parents=True)
    stdout_log = (run_dir / "logs/stdout.log").open("w", encoding="utf-8", buffering=1)
    stderr_log = (run_dir / "logs/stderr.log").open("w", encoding="utf-8", buffering=1)
    sys.stdout = shared.Tee(sys.__stdout__, stdout_log)
    sys.stderr = shared.Tee(sys.__stderr__, stderr_log)
    run_events_path = run_dir / "run_events.jsonl"
    previous_excepthook = sys.excepthook

    def salvage_excepthook(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        write_failure_packet(run_dir, run_events_path, exc)
        previous_excepthook(exc_type, exc, tb)

    sys.excepthook = salvage_excepthook
    random.seed(args.seed)
    run_events_path.write_text(
        json.dumps({"created_at_utc": shared.utc_now(), "event": "static1pct_salvage_started"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    try:
        copy_source_artifacts(source_run_dir, run_dir)
        environment = shared.build_environment(schema_version=checker.SCHEMA_VERSION)
        shared.write_json(run_dir / "environment.json", environment)
        phase4_runner.write_environment_text(run_dir / "environment.txt", environment)
        shared.write_json(
            run_dir / "command_metadata.json",
            {
                "schema_version": f"{checker.SCHEMA_VERSION}_command",
                "created_at_utc": shared.utc_now(),
                "argv": sys.argv if argv is None else ["run_om_phase9_m11b_nemotron_static1pct_salvage.py", *argv],
                "cwd": str(Path.cwd()),
                "branch": "outlier_migrate_phase9_m11b_nemotron_static1pct_salvage",
                "run_dir": str(run_dir),
                "source_run_dir": str(source_run_dir),
                "batch_size": args.batch_size,
                "reused_score_regimes": REUSED_SCORE_REGIMES,
                "rerun_regime": "static_1pct",
            },
        )
        shared.write_json(
            run_dir / "random_seed.json",
            {"schema_version": f"{checker.SCHEMA_VERSION}_random_seed", "seed": args.seed, "determinism": {"do_sample": False, "num_beams": 1}},
        )

        prompt_manifest = load_json(run_dir / "prompt_manifest.json")
        model_provenance = load_json(run_dir / "model_provenance.json")
        protected_sets = load_json(run_dir / "protected_sets.json")
        protected_trajectories = load_json(run_dir / "protected_trajectories.json")
        trace_path = run_dir / "bf16_traces.jsonl.gz"
        activation_path = run_dir / "activation_magnitudes.jsonl.gz"
        prompts = prompt_manifest["prompts"]
        target_tokens = phase4_runner.load_trace_tokens(trace_path)

        all_scores = {regime: read_score_cache(run_dir, regime) for regime in REUSED_SCORE_REGIMES}
        all_scores["static_1pct"], static_excluded = base.score_regime(
            model_provenance=model_provenance,
            protected_sets=protected_sets,
            regime="static_1pct",
            prompts=prompts,
            target_tokens=target_tokens,
            batch_size=args.batch_size,
            dtype_name=args.dtype,
            device_name=args.device,
            run_events_path=run_events_path,
        )
        base.write_score_cache(run_dir, "static_1pct", all_scores["static_1pct"])

        source_excluded = load_json(source_run_dir / "excluded_tensors.json").get("by_regime", {})
        excluded_by_regime = {"static_1pct": static_excluded}
        for regime in ["m11b_top1", "m11b_top5", "m11b_top10", "static_top10"]:
            excluded_by_regime[regime] = source_excluded[regime]
        shared.write_json(
            run_dir / "excluded_tensors.json",
            {"schema_version": f"{checker.SCHEMA_VERSION}_excluded_tensors", "created_at_utc": shared.utc_now(), "by_regime": excluded_by_regime},
        )

        per_trace_rows = build_per_trace_rows(prompts, all_scores)
        shared.write_json(
            run_dir / "per_trace_metrics.json",
            {"schema_version": f"{checker.SCHEMA_VERSION}_per_trace_metrics", "created_at_utc": shared.utc_now(), "traces": per_trace_rows},
        )
        metrics, bootstrap, controls = base.build_metrics(
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
        run_events_path.open("a", encoding="utf-8").write(json.dumps({"created_at_utc": shared.utc_now(), "event": "static1pct_salvage_completed"}, sort_keys=True) + "\n")
        print(json.dumps({"run_dir": str(run_dir), "results_by_regime": metrics["results_by_regime"]}, indent=2, sort_keys=True))
        shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=checker.SCHEMA_VERSION))
        result = checker.evaluate(run_dir)
        print(json.dumps({"checker_decision": result["decision"], "artifact_complete": result.get("artifact_complete", False)}, indent=2, sort_keys=True))
        shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=checker.SCHEMA_VERSION))
        checker.evaluate(run_dir)
        sys.excepthook = previous_excepthook
        return 0 if result["decision"] != checker.FAIL_INFRA else 1
    except BaseException as exc:
        write_failure_packet(run_dir, run_events_path, exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
