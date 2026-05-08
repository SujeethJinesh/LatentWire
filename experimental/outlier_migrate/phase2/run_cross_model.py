#!/usr/bin/env python3
"""Run authorized-window OutlierMigrate Phase 2 partial Nemotron-3 validation."""

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

from experimental.outlier_migrate.phase2 import check_cross_model as checker
from experimental.shared import run_phase0_branch as shared


DEFAULT_RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase2/results"
DEFAULT_PROMPT_FILE = ROOT / "experimental/shared/prompts/aime_2025_indices_0_23.jsonl"
DEFAULT_MODEL_ID = checker.MODEL_ID
DEFAULT_POSITIONS = checker.POSITIONS
DEFAULT_SEED = 20260508
SCHEMA_VERSION = checker.SCHEMA_VERSION


def resolve_model_snapshot_light(model_id: str) -> dict[str, Any]:
    """Resolve local HF snapshot without hashing multi-GB weight shards."""

    safe_id = "models--" + model_id.replace("/", "--")
    cache_roots = [
        Path("/workspace/hf_cache/hub"),
        Path("/workspace/hf_cache"),
        Path.home() / ".cache/huggingface/hub",
    ]
    for cache_root in cache_roots:
        repo_dir = cache_root / safe_id
        refs_main = repo_dir / "refs/main"
        snapshots_dir = repo_dir / "snapshots"
        commits: list[str] = []
        if refs_main.is_file():
            commits.append(refs_main.read_text(encoding="utf-8").strip())
        if snapshots_dir.is_dir():
            commits.extend(path.name for path in snapshots_dir.iterdir() if path.is_dir())
        for commit in dict.fromkeys(commits):
            snapshot = snapshots_dir / commit
            if not (snapshot / "config.json").exists():
                continue
            files = []
            for path in sorted(snapshot.iterdir()):
                if not (path.is_file() or path.is_symlink()):
                    continue
                resolved = path.resolve()
                item: dict[str, Any] = {
                    "path": path.name,
                    "resolved_path": str(resolved),
                    "bytes": resolved.stat().st_size if resolved.exists() else None,
                }
                if path.suffix in {".safetensors", ".bin"}:
                    item["sha256"] = None
                    item["sha256_omitted_reason"] = "large model weight shard; hf_snapshot_commit is exact model revision"
                else:
                    item["sha256"] = shared.file_sha256(resolved) if resolved.is_file() else None
                files.append(item)
            return {
                "schema_version": f"{SCHEMA_VERSION}_model_provenance",
                "created_at_utc": shared.utc_now(),
                "model_id": model_id,
                "local_files_only": True,
                "hf_snapshot_commit": commit,
                "snapshot_path": str(snapshot),
                "cache_repo_path": str(repo_dir),
                "files": files,
                "large_weight_sha_policy": (
                    "Weight shard hashes are omitted to avoid multi-minute network-filesystem hashing; "
                    "the HuggingFace snapshot commit records the exact model revision."
                ),
            }
    return {
        "schema_version": f"{SCHEMA_VERSION}_model_provenance",
        "created_at_utc": shared.utc_now(),
        "model_id": model_id,
        "local_files_only": True,
        "hf_snapshot_commit": None,
        "snapshot_path": None,
        "error": "no local HuggingFace snapshot with config.json found",
    }


def parse_positions(text: str) -> tuple[int, ...]:
    positions: list[int] = []
    for part in text.split(","):
        value = int(part.strip())
        if value <= 0:
            raise argparse.ArgumentTypeError("decode positions must be positive integers")
        positions.append(value)
    if tuple(positions) != DEFAULT_POSITIONS:
        raise argparse.ArgumentTypeError(
            f"OutlierMigrate Phase 2 partial validation requires positions {DEFAULT_POSITIONS}; "
            f"got {tuple(positions)}"
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
        reasons.append(f"prompt count {len(prompts)} is not the Phase 1/2 protocol count 24")
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


def build_validation_scope() -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_VERSION}_validation_scope",
        "created_at_utc": shared.utc_now(),
        "queue_entry": "cross_model_validation_outlier_migrate",
        "scope": "partial_nemotron3_only_authorized_window",
        "authorization_source": "swarm/goal.md#10-hour-authorized-work-window--may-8-2026",
        "model_key": checker.MODEL_KEY,
        "model_id": checker.MODEL_ID,
        "not_full_phase2_gate": True,
        "full_cross_validation_complete": False,
        "deferred_models": checker.DEFERRED_MODELS,
        "forbidden_during_window": [
            "upgrade_vllm",
            "download_qwen36_or_kimi_weights",
            "modify_preregistration_files",
            "claim_full_cross_validation",
        ],
        "interpretation": (
            "This packet can support a partial cross-family check on Nemotron-3 only. "
            "It must not be used as PASS_OM_TRANSFERS_CROSS_MODEL until Qwen3.6 and "
            "Kimi Linear validation are resolved or the human author changes the queue contract."
        ),
    }


def build_phase1_reference() -> dict[str, Any]:
    phase1_run_dir = ROOT / checker.PHASE1_REFERENCE_RUN
    artifact_check = phase1_run_dir / "artifact_check.json"
    metrics = phase1_run_dir / "metrics.json"
    return {
        "schema_version": f"{SCHEMA_VERSION}_phase1_reference",
        "created_at_utc": shared.utc_now(),
        "phase1_run_dir": checker.PHASE1_REFERENCE_RUN,
        "phase1_decision": checker.PHASE1_REFERENCE_DECISION,
        "phase1_migration_fraction": 0.843165650406504,
        "phase1_bootstrap_ci95": {
            "ci95_low": 0.8334349593495936,
            "ci95_high": 0.8511432926829268,
        },
        "phase1_artifact_check_sha256": shared.file_sha256(artifact_check) if artifact_check.is_file() else None,
        "phase1_metrics_sha256": shared.file_sha256(metrics) if metrics.is_file() else None,
    }


def main(argv: list[str] | None = None) -> int:
    shared.SCHEMA_VERSION = SCHEMA_VERSION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default=f"om_phase2_nemotron3_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--partial-nemotron3-only",
        action="store_true",
        help="Required: acknowledges this packet is not full Qwen3.6/Kimi cross-validation.",
    )
    parser.add_argument("--positions", type=parse_positions, default=",".join(map(str, DEFAULT_POSITIONS)))
    parser.add_argument("--max-new-tokens", type=int, default=max(DEFAULT_POSITIONS))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=checker.THRESHOLDS["bootstrap_samples"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = parser.parse_args(argv)

    if not args.partial_nemotron3_only:
        raise SystemExit("--partial-nemotron3-only is required by the authorized-window packet contract")
    if args.model_id != DEFAULT_MODEL_ID:
        raise SystemExit(f"authorized-window Phase 2 validation requires {DEFAULT_MODEL_ID}; got {args.model_id}")
    if args.prompt_file.resolve() != DEFAULT_PROMPT_FILE.resolve():
        raise SystemExit(f"OutlierMigrate Phase 2 partial validation requires canonical prompt file {DEFAULT_PROMPT_FILE}")
    if shared.file_sha256(args.prompt_file) != checker.phase1.EXPECTED_PROMPT_FILE_SHA256:
        raise SystemExit("canonical AIME-2025 indices 0-23 prompt file hash drifted")
    if args.max_new_tokens < max(args.positions):
        raise SystemExit("--max-new-tokens must reach the 20000-token Phase 1 position")
    if args.bootstrap_samples != checker.THRESHOLDS["bootstrap_samples"]:
        raise SystemExit("OutlierMigrate partial Phase 2 uses the Phase 1 bootstrap n=1000")
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
    model_provenance = resolve_model_snapshot_light(args.model_id)
    validation_scope = build_validation_scope()
    phase1_reference = build_phase1_reference()
    command_metadata = {
        "schema_version": f"{SCHEMA_VERSION}_command",
        "created_at_utc": shared.utc_now(),
        "argv": sys.argv if argv is None else ["run_cross_model.py", *argv],
        "cwd": str(Path.cwd()),
        "branch": "outlier_migrate_phase2_partial_nemotron3",
        "run_dir": str(run_dir),
        "batch_size": args.batch_size,
        "assumptions": [
            "Nemotron-3 is the only model executed in this authorized-window packet",
            "Qwen3.6 and Kimi Linear are explicitly deferred; their weights must not be downloaded here",
            "manual greedy decoding is used to expose deterministic decode-step activation hooks",
            "model.prepare_inputs_for_generation is used for the local Transformers cache path",
            "layer forward output absolute values are treated as per-channel activation magnitudes",
            "EOS is recorded but decoding continues to 20000 generated tokens to preserve the Phase 1 grid",
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
    shared.write_json(run_dir / "validation_scope.json", validation_scope)
    shared.write_json(run_dir / "phase1_reference.json", phase1_reference)
    if prompt_reasons:
        shared.write_json(run_dir / "infra_error.json", {"reasons": prompt_reasons})
        print(f"prompt manifest failed protocol invariants: {prompt_reasons}", file=sys.stderr)
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
    metrics.update(
        {
            "schema_version": f"{SCHEMA_VERSION}_metrics",
            "metric_name": "outlier_rank_migration_fraction",
            "created_at_utc": shared.utc_now(),
            "branch": "outlier_migrate_phase2_partial_nemotron3",
            "validation_scope": "partial_nemotron3_only_authorized_window",
            "full_cross_validation_complete": False,
            "deferred_models": checker.DEFERRED_MODELS,
            "phase1_decision": checker.PHASE1_REFERENCE_DECISION,
            "phase1_migration_fraction": 0.843165650406504,
            "model_key": checker.MODEL_KEY,
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
