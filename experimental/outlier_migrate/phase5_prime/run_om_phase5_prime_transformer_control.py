#!/usr/bin/env python3
"""Run OutlierMigrate Phase 5' pure-Transformer control."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.outlier_migrate.phase5_prime import check_om_phase5_prime_transformer_control as checker
from experimental.shared import run_phase0_branch as shared


DEFAULT_RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase5_prime/results"
DEFAULT_PROMPT_FILE = ROOT / "experimental/shared/prompts/aime_2025_indices_0_23.jsonl"
DEFAULT_MODEL_ID = checker.MODEL_ID
DEFAULT_POSITIONS = checker.POSITIONS
DEFAULT_SEED = checker.THRESHOLDS["bootstrap_seed"]
SCHEMA_VERSION = checker.SCHEMA_VERSION


def parse_positions(text: str) -> tuple[int, ...]:
    positions: list[int] = []
    for part in text.split(","):
        value = int(part.strip())
        if value <= 0:
            raise argparse.ArgumentTypeError("decode positions must be positive integers")
        positions.append(value)
    if tuple(positions) != DEFAULT_POSITIONS:
        raise argparse.ArgumentTypeError(
            f"Phase 5' requires positions {DEFAULT_POSITIONS}; got {tuple(positions)}"
        )
    return tuple(positions)


def parse_prompt_file(prompt_file: Path) -> tuple[list[dict[str, Any]], list[str]]:
    reasons: list[str] = []
    prompts: list[dict[str, Any]] = []
    if not prompt_file.is_file():
        return [], [f"prompt file is missing: {prompt_file}"]
    try:
        lines = [line for line in prompt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row_index, line in enumerate(lines):
            item = json.loads(line)
            if not isinstance(item, dict):
                reasons.append(f"prompt row {row_index} is not a JSON object")
                continue
            prompt = item.get("prompt") or item.get("problem") or item.get("question")
            if not isinstance(prompt, str) or not prompt.strip():
                reasons.append(f"prompt row {row_index} has no prompt/problem/question text")
                continue
            index = int(item.get("index", row_index))
            prompts.append(
                {
                    "index": index,
                    "prompt_id": str(item.get("prompt_id", item.get("id", row_index))),
                    "prompt": prompt,
                    "answer": item.get("answer"),
                    "source_dataset": item.get("source_dataset"),
                    "source_file": item.get("source_file"),
                    "source_commit": item.get("source_commit"),
                }
            )
            if item.get("source_dataset") != checker.phase1.EXPECTED_PROMPT_SOURCE_DATASET:
                reasons.append(
                    f"prompt row {row_index} source_dataset is not "
                    f"{checker.phase1.EXPECTED_PROMPT_SOURCE_DATASET}"
                )
            expected_file = checker.phase1.expected_source_file(index)
            if item.get("source_file") != expected_file:
                reasons.append(f"prompt row {row_index} source_file is not {expected_file}")
            if item.get("source_commit") != checker.phase1.EXPECTED_PROMPT_SOURCE_COMMIT:
                reasons.append(
                    f"prompt row {row_index} source_commit is not "
                    f"{checker.phase1.EXPECTED_PROMPT_SOURCE_COMMIT}"
                )
            if item.get("prompt_id") != checker.phase1.expected_prompt_id(index):
                reasons.append(f"prompt row {row_index} prompt_id mismatch")
    except Exception as exc:
        return [], [f"cannot parse prompt file: {exc!r}"]
    if [row["index"] for row in prompts] != list(range(checker.TRACE_COUNT)):
        reasons.append("prompt indices are not exactly deterministic indices 0-23")
    if len(prompts) != checker.TRACE_COUNT:
        reasons.append(f"prompt count {len(prompts)} is not the preregistered count 24")
    return prompts, reasons


def build_prompt_manifest(prompt_file: Path) -> tuple[dict[str, Any], list[str]]:
    prompts, reasons = parse_prompt_file(prompt_file)
    return (
        {
            "schema_version": f"{SCHEMA_VERSION}_prompt_manifest",
            "created_at_utc": shared.utc_now(),
            "source": "AIME-2025",
            "selection": "deterministic_indices_0_23",
            "source_dataset": checker.phase1.EXPECTED_PROMPT_SOURCE_DATASET,
            "source_dataset_commit": checker.phase1.EXPECTED_PROMPT_SOURCE_COMMIT,
            "source_file_order": ["aime2025-I.jsonl", "aime2025-II.jsonl"],
            "prompt_file": str(prompt_file),
            "prompt_file_sha256": shared.file_sha256(prompt_file) if prompt_file.is_file() else None,
            "prompt_count": len(prompts),
            "prompt_sha256": checker.phase1.prompt_payload_sha256(prompts) if prompts else None,
            "prompt_sha256_semantics": "sha256 of concatenated prompt text in deterministic index order",
            "prompts": prompts,
        },
        reasons,
    )


def main(argv: list[str] | None = None) -> int:
    shared.SCHEMA_VERSION = SCHEMA_VERSION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default=f"om_phase5p_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--positions", type=parse_positions, default=",".join(map(str, DEFAULT_POSITIONS)))
    parser.add_argument("--max-new-tokens", type=int, default=max(DEFAULT_POSITIONS))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=checker.THRESHOLDS["bootstrap_samples"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = parser.parse_args(argv)

    if args.model_id not in {checker.MODEL_ID, checker.FALLBACK_MODEL_ID}:
        raise SystemExit("Phase 5' model must be the preregistered primary or fallback")
    if args.prompt_file.resolve() != DEFAULT_PROMPT_FILE.resolve():
        raise SystemExit(f"Phase 5' requires canonical prompt file {DEFAULT_PROMPT_FILE}")
    if shared.file_sha256(args.prompt_file) != checker.phase1.EXPECTED_PROMPT_FILE_SHA256:
        raise SystemExit("Phase 5' canonical prompt file hash drifted")
    if args.max_new_tokens < max(args.positions):
        raise SystemExit("--max-new-tokens must reach the preregistered 20000-token position")
    if args.bootstrap_samples != checker.THRESHOLDS["bootstrap_samples"]:
        raise SystemExit("Phase 5' preregisters bootstrap n=1000")
    if args.seed != checker.THRESHOLDS["bootstrap_seed"]:
        raise SystemExit("Phase 5' preregisters bootstrap seed 20260511")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")

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

    random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

    prompt_manifest, prompt_reasons = build_prompt_manifest(args.prompt_file)
    environment = shared.build_environment(schema_version=SCHEMA_VERSION)
    model_provenance = shared.resolve_model_snapshot(args.model_id, schema_version=SCHEMA_VERSION)
    command_metadata = {
        "schema_version": f"{SCHEMA_VERSION}_command",
        "created_at_utc": shared.utc_now(),
        "argv": sys.argv if argv is None else ["run_om_phase5_prime_transformer_control.py", *argv],
        "cwd": str(Path.cwd()),
        "branch": "outlier_migrate_phase5_prime_transformer_control",
        "run_dir": str(run_dir),
        "batch_size": args.batch_size,
        "deterministic_decode": {"do_sample": False, "num_beams": 1, "decode_policy": "greedy"},
        "compatibility_precheck": {
            "primary_model": checker.MODEL_ID,
            "fallback_model": checker.FALLBACK_MODEL_ID,
            "selected_model": args.model_id,
            "vllm_supported_architecture": "Qwen2ForCausalLM",
        },
        "assumptions": [
            "manual greedy decoding is used to expose deterministic decode-step hooks",
            "layer forward output absolute values are treated as per-channel activation magnitudes",
            "EOS is recorded but decoding continues to 20000 generated tokens to satisfy the fixed grid",
        ],
    }
    random_seed = {
        "schema_version": f"{SCHEMA_VERSION}_random_seed",
        "seed": args.seed,
        "determinism": {"do_sample": False, "num_beams": 1, "torch_manual_seed": args.seed},
    }
    shared.write_json(run_dir / "prompt_manifest.json", prompt_manifest)
    shared.write_json(run_dir / "environment.json", environment)
    shared.write_json(run_dir / "model_provenance.json", model_provenance)
    shared.write_json(run_dir / "command_metadata.json", command_metadata)
    shared.write_json(run_dir / "random_seed.json", random_seed)
    if prompt_reasons:
        shared.write_json(run_dir / "infra_error.json", {"reasons": prompt_reasons})
        print(f"prompt manifest failed preregistered invariants: {prompt_reasons}", file=sys.stderr)
        shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
        return 1
    if not model_provenance.get("snapshot_path"):
        shared.write_json(run_dir / "infra_error.json", {"reasons": [model_provenance.get("error")]})
        print("model snapshot could not be resolved locally; see infra_error.json", file=sys.stderr)
        shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
        return 1

    prompts = prompt_manifest["prompts"]
    model, tokenizer, device = shared.load_model_and_tokenizer(
        model_provenance, dtype_name=args.dtype, device_name=args.device
    )
    activation_path = run_dir / "activation_magnitudes.jsonl.gz"
    activation_manifest = shared.capture_activation_magnitudes(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompts=prompts,
        positions=args.positions,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        output_path=activation_path,
        run_events_path=run_events_path,
    )
    shared.write_json(run_dir / "activation_magnitude_manifest.json", activation_manifest)
    rows = list(shared.iter_activation_rows(activation_path))
    metrics = checker.compute_metrics(rows, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    decomposition = metrics.pop("migration_decomposition")
    metrics.update(
        {
            "schema_version": f"{SCHEMA_VERSION}_metrics",
            "metric_name": "outlier_rank_migration_fraction",
            "created_at_utc": shared.utc_now(),
            "branch": "outlier_migrate_phase5_prime_transformer_control",
            "model_family": "pure_transformer_rope",
            "model_id": args.model_id,
            "model_snapshot_commit": model_provenance.get("hf_snapshot_commit"),
            "prompt_source": "AIME-2025",
            "prompt_selection": "deterministic_indices_0_23",
            "prompt_sha256": prompt_manifest["prompt_sha256"],
            "positions": list(args.positions),
            "trace_count": len(prompts),
            "layer_count": activation_manifest["layer_count"],
            "hidden_size": rows[0]["channel_count"] if rows else None,
            "activation_artifact": "activation_magnitudes.jsonl.gz",
            "activation_artifact_sha256": shared.file_sha256(activation_path),
            "thresholds": checker.THRESHOLDS,
            "aggregation": {
                "trace_level": "mean over layers per trace",
                "gate_level": "mean over 24 trace-level migration fractions",
                "bootstrap_unit": "trace",
                "layer_weighting": "equal",
            },
        }
    )
    shared.write_json(run_dir / "metrics.json", metrics)
    shared.write_json(run_dir / "migration_decomposition.json", decomposition)
    checker.write_decomposition_report(run_dir / "migration_decomposition.md", decomposition)
    shared.write_json(
        run_dir / "bootstrap_ci.json",
        {
            "schema_version": f"{SCHEMA_VERSION}_bootstrap_ci",
            "metric_name": metrics["metric_name"],
            "bootstrap_samples": metrics["bootstrap_samples"],
            "bootstrap_seed": metrics["bootstrap_seed"],
            "bootstrap_ci95": metrics["bootstrap_ci95"],
            "migration_fraction": metrics["migration_fraction"],
        },
    )
    run_events_path.open("a", encoding="utf-8").write(
        json.dumps({"created_at_utc": shared.utc_now(), "event": "run_completed"}, sort_keys=True) + "\n"
    )
    print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, indent=2, sort_keys=True))
    sys.stdout.flush()
    sys.stderr.flush()
    shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
