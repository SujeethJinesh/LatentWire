#!/usr/bin/env python3
"""Run Stage 1 E5 Qwen3-8B partial scale-up."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.outlier_migrate.phase4 import run_om_phase4_intervention as phase4_runner
from experimental.outlier_migrate.phase9 import check_om_stage1_e5_qwen3_scaleup as checker
from experimental.outlier_migrate.phase9 import run_om_phase9_m11b_budget_scaling as m11b_runner
from experimental.shared import run_phase0_branch as shared


SCHEMA_VERSION = checker.SCHEMA_VERSION
UPDATE_POSITIONS = tuple(range(checker.UPDATE_CADENCE, checker.SCORING_POSITION + 1, checker.UPDATE_CADENCE))
CAPTURE_POSITIONS = tuple(sorted(set(UPDATE_POSITIONS + checker.LEAVING_POSITIONS)))


def resolve_snapshot_light(model_id: str) -> dict[str, Any]:
    safe_id = "models--" + model_id.replace("/", "--")
    bases = [os.environ.get("HF_HUB_CACHE"), os.environ.get("TRANSFORMERS_CACHE"), "/workspace/hf_cache/hub", "/workspace/hf_cache", str(Path.home() / ".cache/huggingface/hub")]
    checked: list[str] = []
    for base in [item for item in bases if item]:
        repo_dir = Path(base) / safe_id
        checked.append(str(repo_dir))
        refs_main = repo_dir / "refs/main"
        snapshots_dir = repo_dir / "snapshots"
        commits: list[str] = []
        if refs_main.is_file():
            commits.append(refs_main.read_text(encoding="utf-8").strip())
        if snapshots_dir.is_dir():
            commits.extend(path.name for path in snapshots_dir.iterdir() if path.is_dir())
        for commit in dict.fromkeys(commits):
            snapshot = snapshots_dir / commit
            if (snapshot / "config.json").is_file():
                return {"schema_version": f"{SCHEMA_VERSION}_model_provenance", "created_at_utc": shared.utc_now(), "model_id": model_id, "local_files_only": True, "hf_snapshot_commit": commit, "snapshot_path": str(snapshot), "cache_repo_path": str(repo_dir), "checked_cache_paths": checked}
    return {"schema_version": f"{SCHEMA_VERSION}_model_provenance", "created_at_utc": shared.utc_now(), "model_id": model_id, "local_files_only": True, "hf_snapshot_commit": None, "snapshot_path": None, "checked_cache_paths": checked, "error": "no local snapshot with config.json found"}


def parse_prompts(path: Path) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for row_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        item = json.loads(line)
        prompts.append({"index": int(item.get("index", row_index)), "prompt_id": str(item.get("prompt_id", item.get("id", row_index))), "prompt": str(item.get("prompt") or item.get("problem") or item.get("question")), "answer": item.get("answer"), "source_dataset": item.get("source_dataset"), "source_file": item.get("source_file"), "source_commit": item.get("source_commit")})
    if [int(row["index"]) for row in prompts] != list(range(checker.TRACE_COUNT)):
        raise ValueError(f"prompt file must contain deterministic indices 0-{checker.TRACE_COUNT - 1}")
    return prompts


def prompt_manifest(path: Path) -> dict[str, Any]:
    prompts = parse_prompts(path)
    payload = "".join(str(row["prompt"]) for row in prompts).encode("utf-8")
    return {"schema_version": f"{SCHEMA_VERSION}_prompt_manifest", "created_at_utc": shared.utc_now(), "source": "AIME-2025", "selection": "deterministic_indices_0_11", "prompt_file": str(path), "prompt_file_sha256": shared.file_sha256(path), "prompt_count": len(prompts), "prompt_sha256": shared.bytes_sha256(payload), "prompts": prompts}


def top_channels(values: list[float], count: int) -> list[int]:
    return sorted(range(len(values)), key=lambda channel: (-float(values[channel]), channel))[:count]


def build_protected_sets(rows: list[dict[str, Any]]) -> dict[str, Any]:
    means_by_layer, layer_names = m11b_runner.activation_means_by_layer_position(rows)
    missing = [f"layer {layer}: missing update positions" for layer, by_position in means_by_layer.items() if any(pos not in by_position for pos in UPDATE_POSITIONS)]
    if missing:
        raise RuntimeError("; ".join(missing[:4]))
    layers_by_regime: dict[str, dict[str, Any]] = {"static_1pct": {}, "m11b_top10": {}}
    for layer_index in sorted(means_by_layer):
        values_100 = means_by_layer[layer_index][100]
        channel_count = len(values_100)
        top1_count = max(1, math.ceil(channel_count * checker.TOP_FRACTION))
        top10_count = max(1, math.ceil(channel_count * checker.BUDGET_FRACTION))
        layers_by_regime["static_1pct"][str(layer_index)] = {"layer_name": layer_names[layer_index], "channel_count": channel_count, "protected_count": top1_count, "protected_channels": sorted(top_channels(values_100, top1_count)), "source": "position_100_top1"}
        scores = [0.0] * channel_count
        for channel in top_channels(values_100, top10_count):
            scores[channel] = 1.0
        selected = sorted(top_channels(scores, top10_count))
        for position in UPDATE_POSITIONS:
            indicator = set(top_channels(means_by_layer[layer_index][position], top10_count))
            scores = [checker.ALPHA * (1.0 if channel in indicator else 0.0) + (1.0 - checker.ALPHA) * float(scores[channel]) for channel in range(channel_count)]
            selected = sorted(top_channels(scores, top10_count))
        layers_by_regime["m11b_top10"][str(layer_index)] = {"layer_name": layer_names[layer_index], "channel_count": channel_count, "budget_fraction": checker.BUDGET_FRACTION, "protected_count": len(selected), "protected_channels": selected, "source": "ema_alpha_0_3_top_10pct_final_snapshot"}
    return {"schema_version": f"{SCHEMA_VERSION}_protected_sets", "created_at_utc": shared.utc_now(), "selection_basis": "EMA-smoothed mean absolute layer output activations on Qwen3-8B AIME-2025 traces 0-11", "alpha": checker.ALPHA, "regimes": {regime: {"layers": layers} for regime, layers in layers_by_regime.items()}}


def score_regime(model_provenance: dict[str, Any], protected_sets: dict[str, Any], regime: str, prompts: list[dict[str, Any]], target_tokens: dict[int, list[int]], args: argparse.Namespace, events: Path) -> tuple[dict[int, dict[str, float]], dict[str, Any] | None]:
    model, tokenizer, device = shared.load_model_and_tokenizer(model_provenance, dtype_name=args.dtype, device_name=args.device)
    excluded = None
    if regime != "bf16":
        excluded = phase4_runner.apply_quantization(model, protected_sets, regime)
    scores = phase4_runner.score_targets(model=model, tokenizer=tokenizer, device=device, prompts=prompts, target_tokens=target_tokens, max_new_tokens=checker.SCORING_POSITION, batch_size=args.batch_size, use_float16_autocast=regime != "bf16", run_events_path=events, regime_name=regime)
    del model, tokenizer, device
    phase4_runner.release_model_memory()
    return scores, excluded


def write_score_cache(run_dir: Path, regime: str, scores: dict[int, dict[str, float]]) -> None:
    shared.write_json(run_dir / "score_cache" / f"{regime}.json", {"schema_version": f"{SCHEMA_VERSION}_score_cache", "created_at_utc": shared.utc_now(), "regime": regime, "scores": {str(index): row for index, row in sorted(scores.items())}})


def write_terminal_packet(run_dir: Path, events: Path, decision: str, reason: str, exc: BaseException | None = None) -> None:
    payload: dict[str, Any] = {"schema_version": f"{SCHEMA_VERSION}_infra_error", "created_at_utc": shared.utc_now(), "decision": decision, "reason": reason}
    if exc is not None:
        payload["exception_type"] = type(exc).__name__
        payload["traceback"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    events.open("a", encoding="utf-8").write(json.dumps({"created_at_utc": shared.utc_now(), "event": "terminal_packet", "decision": decision, "reason": reason}, sort_keys=True) + "\n")
    shared.write_json(run_dir / "infra_error.json", payload)
    shared.write_json(run_dir / "metrics.json", {"schema_version": f"{SCHEMA_VERSION}_metrics", "status": decision, "reason": reason, "model_id": checker.MODEL_ID})
    shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
    checker.evaluate(run_dir)


def summarize_recovery(per_trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    included = [row for row in per_trace_rows if not bool(row["no_recoverable_static_gap"])]
    values = [float(row["recoveries"][checker.RECOVERY_REGIME]) for row in included]
    return {"median_recovery": float(__import__("statistics").median(values)) if values else None, "mean_recovery": float(mean(values)) if values else None, "bootstrap_ci95": checker.bootstrap_median(values), "included_trace_count": len(values), "total_trace_count": len(per_trace_rows), "no_recoverable_static_gap_count": len(per_trace_rows) - len(values)}


def cap_exhausted(start: float, cap_hours: float) -> bool:
    return (time.monotonic() - start) / 3600.0 >= cap_hours


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"om_stage1_e5_qwen3_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--results-dir", type=Path, default=checker.RESULTS_DIR)
    parser.add_argument("--prompt-file", type=Path, default=checker.DEFAULT_PROMPT_FILE)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cap-hours", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=checker.BOOTSTRAP_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.seed != checker.BOOTSTRAP_SEED:
        raise SystemExit(f"E5 preregisters seed {checker.BOOTSTRAP_SEED}")
    manifest = prompt_manifest(args.prompt_file)
    provenance = resolve_snapshot_light(checker.MODEL_ID)
    if args.dry_run:
        print(json.dumps({"schema_version": f"{SCHEMA_VERSION}_dry_run", "model_id": checker.MODEL_ID, "snapshot": provenance, "prompt_manifest": {k: v for k, v in manifest.items() if k != "prompts"}, "capture_positions": list(CAPTURE_POSITIONS), "cap_hours": args.cap_hours}, indent=2, sort_keys=True))
        return 0
    run_dir = args.results_dir / args.run_id
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")
    (run_dir / "logs").mkdir(parents=True)
    sys.stdout = shared.Tee(sys.__stdout__, (run_dir / "logs/stdout.log").open("w", encoding="utf-8", buffering=1))
    sys.stderr = shared.Tee(sys.__stderr__, (run_dir / "logs/stderr.log").open("w", encoding="utf-8", buffering=1))
    events = run_dir / "run_events.jsonl"
    events.write_text(json.dumps({"created_at_utc": shared.utc_now(), "event": "run_started"}, sort_keys=True) + "\n", encoding="utf-8")
    start = time.monotonic()
    random.seed(args.seed)
    try:
        environment = shared.build_environment(schema_version=SCHEMA_VERSION)
        shared.write_json(run_dir / "environment.json", environment)
        phase4_runner.write_environment_text(run_dir / "environment.txt", environment)
        shared.write_json(run_dir / "model_provenance.json", provenance)
        shared.write_json(run_dir / "prompt_manifest.json", manifest)
        shared.write_json(run_dir / "command_metadata.json", {"schema_version": f"{SCHEMA_VERSION}_command", "created_at_utc": shared.utc_now(), "argv": sys.argv if argv is None else argv, "run_dir": str(run_dir), "cap_hours": args.cap_hours})
        shared.write_json(run_dir / "random_seed.json", {"schema_version": f"{SCHEMA_VERSION}_random_seed", "seed": args.seed, "determinism": {"do_sample": False, "num_beams": 1}})
        shared.write_json(run_dir / "decoding_config.json", {"schema_version": f"{SCHEMA_VERSION}_decoding_config", "leaving_positions": list(checker.LEAVING_POSITIONS), "scoring_position": checker.SCORING_POSITION, "scoring_window_tokens": checker.SCORING_WINDOW_TOKENS, "update_positions": list(UPDATE_POSITIONS), "do_sample": False, "num_beams": 1})
        shared.write_json(run_dir / "quantization_config.json", {"schema_version": f"{SCHEMA_VERSION}_quantization_config", "weight_bits": 4, "scheme": "symmetric_per_output_channel_int4", "activation_dtype": "float16", "protected_channel_dtype": "bfloat16"})
        if not provenance.get("snapshot_path"):
            write_terminal_packet(run_dir, events, checker.SKIPPED_INFRA, f"local Qwen3 snapshot unavailable: {provenance.get('error')}")
            return 0
        prompts = manifest["prompts"]
        model, tokenizer, device = shared.load_model_and_tokenizer(provenance, dtype_name=args.dtype, device_name=args.device)
        activation_path = run_dir / "activation_magnitudes.jsonl.gz"
        activation_manifest = shared.capture_activation_magnitudes(model=model, tokenizer=tokenizer, device=device, prompts=prompts, positions=CAPTURE_POSITIONS, max_new_tokens=max(CAPTURE_POSITIONS), batch_size=args.batch_size, output_path=activation_path, run_events_path=events)
        shared.write_json(run_dir / "activation_magnitude_manifest.json", activation_manifest)
        del model, tokenizer, device
        phase4_runner.release_model_memory()
        if cap_exhausted(start, args.cap_hours):
            write_terminal_packet(run_dir, events, checker.SKIPPED_INFRA, "cap exhausted after activation capture")
            return 0
        activation_rows = list(shared.iter_activation_rows(activation_path))
        leaving = checker.compute_leaving_from_rows(activation_rows)
        protected_sets = build_protected_sets(activation_rows)
        shared.write_json(run_dir / "protected_sets.json", protected_sets)
        trace_path = run_dir / "bf16_traces.jsonl.gz"
        model, tokenizer, device = shared.load_model_and_tokenizer(provenance, dtype_name=args.dtype, device_name=args.device)
        trace_manifest = phase4_runner.generate_bf16_traces(model=model, tokenizer=tokenizer, device=device, prompts=prompts, max_new_tokens=checker.SCORING_POSITION, batch_size=args.batch_size, output_path=trace_path, run_events_path=events)
        shared.write_json(run_dir / "bf16_trace_manifest.json", trace_manifest)
        del model, tokenizer, device
        phase4_runner.release_model_memory()
        target_tokens = phase4_runner.load_trace_tokens(trace_path)
        scores: dict[str, dict[int, dict[str, float]]] = {}
        excluded: dict[str, Any] = {}
        for regime in checker.REGIMES:
            if cap_exhausted(start, args.cap_hours):
                write_terminal_packet(run_dir, events, checker.SKIPPED_INFRA, f"cap exhausted before scoring {regime}")
                return 0
            scores[regime], excluded_item = score_regime(provenance, protected_sets, regime, prompts, target_tokens, args, events)
            write_score_cache(run_dir, regime, scores[regime])
            if excluded_item is not None:
                excluded[regime] = excluded_item
        shared.write_json(run_dir / "excluded_tensors.json", {"schema_version": f"{SCHEMA_VERSION}_excluded_tensors", "created_at_utc": shared.utc_now(), "by_regime": excluded})
        traces = []
        for prompt in prompts:
            index = int(prompt["index"])
            perplexities = {regime: float(scores[regime][index]["perplexity"]) for regime in checker.REGIMES}
            static_gap = perplexities["static_1pct"] - perplexities["bf16"]
            no_gap = static_gap <= 0.0
            recoveries = {checker.RECOVERY_REGIME: None if no_gap else 1.0 - (perplexities[checker.RECOVERY_REGIME] - perplexities["bf16"]) / static_gap}
            traces.append({"prompt_index": index, "prompt_id": prompt["prompt_id"], "perplexities": perplexities, "static_gap": static_gap, "no_recoverable_static_gap": no_gap, "recoveries": recoveries, "scored_tokens": int(scores["bf16"][index]["scored_tokens"])})
        shared.write_json(run_dir / "per_trace_metrics.json", {"schema_version": f"{SCHEMA_VERSION}_per_trace_metrics", "created_at_utc": shared.utc_now(), "traces": traces})
        summary = summarize_recovery(traces)
        metrics = {"schema_version": f"{SCHEMA_VERSION}_metrics", "created_at_utc": shared.utc_now(), "model_id": checker.MODEL_ID, "model_snapshot_commit": provenance.get("hf_snapshot_commit"), "trace_count": checker.TRACE_COUNT, "leaving_rate": leaving, "results_by_regime": {checker.RECOVERY_REGIME: summary}, "thresholds": checker.THRESHOLDS}
        shared.write_json(run_dir / "metrics.json", metrics)
        shared.write_json(run_dir / "bootstrap_ci.json", {"schema_version": f"{SCHEMA_VERSION}_bootstrap_ci", "bootstrap_samples": checker.BOOTSTRAP_SAMPLES, "bootstrap_seed": checker.BOOTSTRAP_SEED, "results_by_regime": {checker.RECOVERY_REGIME: summary}})
        events.open("a", encoding="utf-8").write(json.dumps({"created_at_utc": shared.utc_now(), "event": "run_completed"}, sort_keys=True) + "\n")
        shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
        print(json.dumps(checker.evaluate(run_dir), indent=2, sort_keys=True))
        return 0
    except BaseException as exc:
        write_terminal_packet(run_dir, events, checker.FAIL_INFRA, str(exc), exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
