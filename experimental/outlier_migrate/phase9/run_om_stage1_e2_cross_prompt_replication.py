#!/usr/bin/env python3
"""Run Stage 1 E2 cross-prompt set-leaving replication."""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.outlier_migrate.phase9 import check_om_stage1_e2_cross_prompt_replication as checker
from experimental.shared import run_phase0_branch as shared


SCHEMA_VERSION = checker.SCHEMA_VERSION
DEFAULT_RESULTS_DIR = checker.RESULTS_DIR
DEFAULT_SEED = checker.BOOTSTRAP_SEED
MATH_DATASET = "HuggingFaceH4/MATH-500"
GPQA_DATASET = "Idavidrein/gpqa"
GPQA_CONFIG = "gpqa_diamond"


def load_jsonl(path: Path, *, source: str, limit: int, offset: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip() or len(rows) >= limit:
            continue
        item = json.loads(line)
        prompt = item.get("prompt") or item.get("problem") or item.get("question") or item.get("Question")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{path}: row {index} has no prompt/problem/question")
        rows.append(
            {
                "index": offset + len(rows),
                "prompt_id": f"{source}_{len(rows)}",
                "prompt": prompt,
                "answer": item.get("answer") or item.get("Answer") or item.get("Correct Answer"),
                "source_dataset": source,
                "source_file": str(path),
            }
        )
    if len(rows) < limit:
        raise ValueError(f"{path}: loaded {len(rows)} prompts, expected {limit}")
    return rows


def load_hf_rows(dataset: str, split: str, *, limit: int, offset: int, source: str, config: str | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split=split) if config else load_dataset(dataset, split=split)
    rows: list[dict[str, Any]] = []
    for item in ds:
        if len(rows) >= limit:
            break
        prompt = item.get("problem") or item.get("question") or item.get("Question")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        choices = [item.get(f"Incorrect Answer {i}") for i in range(1, 4)] + [item.get("Correct Answer")]
        if source == "gpqa_diamond" and all(isinstance(choice, str) for choice in choices):
            prompt = prompt + "\n\nChoices:\n" + "\n".join(f"- {choice}" for choice in choices)
        rows.append(
            {
                "index": offset + len(rows),
                "prompt_id": f"{source}_{len(rows)}",
                "prompt": prompt,
                "answer": item.get("answer") or item.get("Answer") or item.get("Correct Answer"),
                "source_dataset": source,
                "source_dataset_id": dataset,
                "source_config": config,
                "source_split": split,
            }
        )
    if len(rows) < limit:
        raise ValueError(f"{dataset}: loaded {len(rows)} usable prompts, expected {limit}")
    return rows


def load_prompts(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    math_rows = (
        load_jsonl(args.math_file, source="math_500", limit=args.math_count, offset=0)
        if args.math_file
        else load_hf_rows(args.math_dataset, args.math_split, limit=args.math_count, offset=0, source="math_500")
    )
    gpqa_rows: list[dict[str, Any]] = []
    gpqa_status: dict[str, Any] = {"enabled": not args.math_only}
    if not args.math_only:
        try:
            gpqa_rows = (
                load_jsonl(args.gpqa_file, source="gpqa_diamond", limit=args.gpqa_count, offset=len(math_rows))
                if args.gpqa_file
                else load_hf_rows(
                    args.gpqa_dataset,
                    args.gpqa_split,
                    limit=args.gpqa_count,
                    offset=len(math_rows),
                    source="gpqa_diamond",
                    config=args.gpqa_config,
                )
            )
            gpqa_status["loaded"] = True
        except Exception as exc:
            raise SystemExit(f"GPQA-Diamond loading failed: {exc}. Re-run with --math-only to use the explicit fallback.")
    else:
        gpqa_status["loaded"] = False
        gpqa_status["fallback"] = "explicit --math-only"
    return math_rows + gpqa_rows, {"math_count": len(math_rows), "gpqa_count": len(gpqa_rows), "gpqa_status": gpqa_status}


def prompt_manifest(prompts: list[dict[str, Any]], source: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_VERSION}_prompt_manifest",
        "created_at_utc": shared.utc_now(),
        "source": "MATH-500+GPQA-Diamond" if source["gpqa_count"] else "MATH-500",
        "selection": "first_rows_after_dataset_ordering",
        "prompt_count": len(prompts),
        "prompt_sha256": shared.bytes_sha256("".join(row["prompt"] for row in prompts).encode("utf-8")),
        "prompt_sha256_semantics": "sha256 of concatenated prompt text in manifest order",
        "source_summary": source,
        "prompts": prompts,
    }


def cap_exhausted(start: float, cap_hours: float) -> bool:
    return (time.monotonic() - start) / 3600.0 >= cap_hours


def append_gzip_rows(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(source, "rt", encoding="utf-8") as src, gzip.open(destination, "at", encoding="utf-8") as dst:
        for line in src:
            if line.strip():
                dst.write(line)


def combine_activation_manifests(manifests: list[dict[str, Any]], activation_path: Path) -> dict[str, Any]:
    if not manifests:
        raise RuntimeError("no activation manifests to combine")
    first = dict(manifests[0])
    first["created_at_utc"] = shared.utc_now()
    first["trace_count"] = sum(int(item["trace_count"]) for item in manifests)
    first["row_count"] = sum(int(item["row_count"]) for item in manifests)
    first["prompt_events"] = [
        event
        for item in manifests
        for event in item.get("prompt_events", [])
    ]
    first["artifact_sha256"] = shared.file_sha256(activation_path)
    first["chunking_semantics"] = "one capture call per prompt so the experiment can stop cleanly at cap boundaries"
    return first


def run_one_model(
    args: argparse.Namespace,
    run_dir: Path,
    key: str,
    prompts: list[dict[str, Any]],
    events: Path,
    start_time: float,
) -> dict[str, Any]:
    model_dir = run_dir / "model_runs" / key
    model_dir.mkdir(parents=True, exist_ok=True)
    model_id = checker.MODEL_REFERENCES[key]["model_id"]
    provenance = shared.resolve_model_snapshot(model_id, schema_version=SCHEMA_VERSION)
    shared.write_json(model_dir / "model_provenance.json", provenance)
    if not provenance.get("snapshot_path"):
        raise RuntimeError(f"{key}: no local model snapshot for {model_id}")
    model, tokenizer, device = shared.load_model_and_tokenizer(provenance, dtype_name=args.dtype, device_name=args.device)
    activation_path = model_dir / "activation_magnitudes.jsonl.gz"
    if activation_path.exists():
        activation_path.unlink()
    prompt_dir = model_dir / "prompt_activation_chunks"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    activation_manifests: list[dict[str, Any]] = []
    completed_prompts: list[dict[str, Any]] = []
    for prompt in prompts:
        if cap_exhausted(start_time, args.cap_hours):
            events.open("a", encoding="utf-8").write(
                json.dumps(
                    {
                        "created_at_utc": shared.utc_now(),
                        "event": "cap_exhausted_within_model",
                        "model_key": key,
                        "completed_prompt_count": len(completed_prompts),
                        "expected_prompt_count": len(prompts),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            break
        chunk_path = prompt_dir / f"prompt_{int(prompt['index']):04d}.jsonl.gz"
        manifest = shared.capture_activation_magnitudes(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompts=[prompt],
            positions=checker.POSITIONS,
            max_new_tokens=args.max_new_tokens,
            batch_size=1,
            output_path=chunk_path,
            run_events_path=events,
        )
        append_gzip_rows(chunk_path, activation_path)
        activation_manifests.append(manifest)
        completed_prompts.append(prompt)
    activation_manifest = combine_activation_manifests(activation_manifests, activation_path)
    shared.write_json(model_dir / "activation_magnitude_manifest.json", activation_manifest)
    rows = list(checker.iter_rows(activation_path))
    prompt_by_index = {int(row["index"]): row for row in completed_prompts}
    for row in rows:
        source = prompt_by_index[int(row["prompt_index"])]["source_dataset"]
        row["source_dataset"] = source
    metrics = checker.compute_model_metrics(rows)
    reference = float(checker.MODEL_REFERENCES[key]["aime_strict_set_leaving"])
    observed = float(metrics["aggregate"]["left_set_fraction"])
    result = {
        "model_id": model_id,
        "model_snapshot_commit": provenance.get("hf_snapshot_commit"),
        "aime_reference": reference,
        "delta_vs_aime": observed - reference,
        "completed_prompt_count": len(completed_prompts),
        "expected_prompt_count": len(prompts),
        "incomplete_due_cap": len(completed_prompts) < len(prompts),
        **metrics,
    }
    shared.write_json(model_dir / "metrics.json", result)
    return result


def write_packet(run_dir: Path, payloads: dict[str, Any], completed: list[str]) -> None:
    model_results = {key: payloads[key] for key in completed}
    headline = {
        key: {
            "left_set_fraction": value["aggregate"]["left_set_fraction"],
            "delta_vs_aime": value["delta_vs_aime"],
        }
        for key, value in model_results.items()
    }
    shared.write_json(
        run_dir / "metrics.json",
        {
            "schema_version": f"{SCHEMA_VERSION}_metrics",
            "created_at_utc": shared.utc_now(),
            "metric_name": "strict_top1_set_leaving_fraction",
            "positions": list(checker.POSITIONS),
            "top_channel_fraction": checker.TOP_FRACTION,
            "replication_tolerance": checker.REPLICATION_TOLERANCE,
            "completed_model_keys": completed,
            "model_results": model_results,
            "headline": headline,
        },
    )
    shared.write_json(run_dir / "bootstrap_ci.json", {"schema_version": f"{SCHEMA_VERSION}_bootstrap_ci", "bootstrap_samples": checker.BOOTSTRAP_SAMPLES, "bootstrap_seed": checker.BOOTSTRAP_SEED})
    shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"om_stage1_e2_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--models", default=",".join(checker.EXPECTED_MODEL_KEYS))
    parser.add_argument("--math-count", type=int, default=30)
    parser.add_argument("--gpqa-count", type=int, default=20)
    parser.add_argument("--math-dataset", default=MATH_DATASET)
    parser.add_argument("--math-split", default="test")
    parser.add_argument("--gpqa-dataset", default=GPQA_DATASET)
    parser.add_argument("--gpqa-config", default=GPQA_CONFIG)
    parser.add_argument("--gpqa-split", default="train")
    parser.add_argument("--math-file", type=Path)
    parser.add_argument("--gpqa-file", type=Path)
    parser.add_argument("--math-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=max(checker.POSITIONS))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cap-hours", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    selected = [item.strip() for item in args.models.split(",") if item.strip()]
    if any(key not in checker.EXPECTED_MODEL_KEYS for key in selected):
        raise SystemExit(f"--models must be drawn from {checker.EXPECTED_MODEL_KEYS}")
    prompts, source = load_prompts(args)
    manifest = prompt_manifest(prompts, source)
    snapshots = {key: shared.resolve_model_snapshot(checker.MODEL_REFERENCES[key]["model_id"], schema_version=SCHEMA_VERSION) for key in selected}
    if args.dry_run:
        print(json.dumps({"prompt_manifest": manifest, "selected_models": selected, "model_snapshots": snapshots}, indent=2, sort_keys=True))
        return 0

    run_dir = args.results_dir / args.run_id
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")
    (run_dir / "logs").mkdir(parents=True)
    sys.stdout = shared.Tee(sys.__stdout__, (run_dir / "logs/stdout.log").open("w", encoding="utf-8", buffering=1))
    sys.stderr = shared.Tee(sys.__stderr__, (run_dir / "logs/stderr.log").open("w", encoding="utf-8", buffering=1))
    events = run_dir / "run_events.jsonl"
    events.write_text(json.dumps({"created_at_utc": shared.utc_now(), "event": "run_started"}, sort_keys=True) + "\n", encoding="utf-8")
    random.seed(args.seed)
    shared.write_json(run_dir / "environment.json", shared.build_environment(schema_version=SCHEMA_VERSION))
    shared.write_json(run_dir / "prompt_manifest.json", manifest)
    shared.write_json(run_dir / "random_seed.json", {"schema_version": f"{SCHEMA_VERSION}_random_seed", "seed": args.seed})
    shared.write_json(run_dir / "command_metadata.json", {"schema_version": f"{SCHEMA_VERSION}_command", "argv": sys.argv if argv is None else argv, "cap_hours": args.cap_hours, "math_only": args.math_only, "models": selected})

    start = time.monotonic()
    results: dict[str, Any] = {}
    completed: list[str] = []
    for key in selected:
        if cap_exhausted(start, args.cap_hours):
            events.open("a", encoding="utf-8").write(json.dumps({"created_at_utc": shared.utc_now(), "event": "cap_exhausted_before_model", "model_key": key}, sort_keys=True) + "\n")
            break
        results[key] = run_one_model(args, run_dir, key, prompts, events, start)
        completed.append(key)
        write_packet(run_dir, results, completed)
    write_packet(run_dir, results, completed)
    events.open("a", encoding="utf-8").write(json.dumps({"created_at_utc": shared.utc_now(), "event": "run_completed", "completed_model_keys": completed}, sort_keys=True) + "\n")
    result = checker.evaluate(run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"] != checker.FAIL_INFRA else 1


if __name__ == "__main__":
    raise SystemExit(main())
