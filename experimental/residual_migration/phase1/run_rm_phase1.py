#!/usr/bin/env python3
"""Run the preregistered Residual Migration Phase 1 gate."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.residual_migration.phase1 import check_rm_phase1 as checker
from experimental.shared import run_phase0_branch as shared


DEFAULT_RESULTS_DIR = ROOT / "experimental/residual_migration/phase1/results"
DEFAULT_PROMPT_FILE = checker.DEFAULT_PROMPT_FILE
DEFAULT_MODEL_ID = checker.MODEL_ID
DEFAULT_SEED = 20260508
SCHEMA_VERSION = checker.SCHEMA_VERSION
RM_DEFAULT_MAX_NEW_TOKENS = 2048
RM_CLIP_QUANTILE = 0.95


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
            if item.get("source_dataset") != checker.EXPECTED_PROMPT_SOURCE_DATASET:
                reasons.append(
                    f"prompt row {row_index} source_dataset is not {checker.EXPECTED_PROMPT_SOURCE_DATASET}"
                )
            if item.get("source_file") != checker.expected_source_file(index):
                reasons.append(f"prompt row {row_index} source_file mismatch")
            if item.get("source_commit") != checker.EXPECTED_PROMPT_SOURCE_COMMIT:
                reasons.append(
                    f"prompt row {row_index} source_commit is not {checker.EXPECTED_PROMPT_SOURCE_COMMIT}"
                )
            if item.get("prompt_id") != checker.expected_prompt_id(index):
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


def configured_layer_types(model: Any, layer_count: int) -> list[str]:
    config = getattr(model, "config", None)
    for attr in ["layer_types", "layers_block_type"]:
        value = getattr(config, attr, None)
        if isinstance(value, (list, tuple)) and len(value) == layer_count:
            return [str(item).lower() for item in value]
    return ["unknown"] * layer_count


def build_layer_groups(layer_types: list[str]) -> dict[str, list[int]]:
    layer_count = len(layer_types)
    first_half = list(range(layer_count // 2))
    second_half = list(range(layer_count // 2, layer_count))
    attention = [index for index, layer_type in enumerate(layer_types) if "attention" in layer_type]
    mamba = [
        index
        for index, layer_type in enumerate(layer_types)
        if "mamba" in layer_type or "ssm" in layer_type
    ]
    if not attention or not mamba or set(attention) & set(mamba) or set(attention) | set(mamba) != set(range(layer_count)):
        raise RuntimeError(
            "could not derive disjoint attention_only/mamba_only layer groups from model.config.layer_types"
        )
    return {
        "full_ablation": list(range(layer_count)),
        "first_half": first_half,
        "second_half": second_half,
        "attention_only": attention,
        "mamba_only": mamba,
    }


def build_residual_clip_hooks_for_subset(
    layers: list[tuple[str, Any]], target_layer_indices: list[int], *, ablation_set: str
) -> tuple[list[Any], dict[str, Any]]:
    import torch

    target_set = set(target_layer_indices)
    stats: dict[str, Any] = {
        "ablation_set": ablation_set,
        "clip_quantile": RM_CLIP_QUANTILE,
        "threshold_scope": "per layer, per batch element, per forwarded token position, over hidden channels",
        "hook_type": "forward_pre_hook",
        "target_layer_indices": sorted(target_set),
        "layers": {},
    }

    def make_hook(layer_index: int, layer_name: str):
        layer_key = str(layer_index)
        stats["layers"][layer_key] = {
            "layer_index": layer_index,
            "layer_name": layer_name,
            "invocations": 0,
            "total_values": 0,
            "clipped_values": 0,
            "clip_fraction": 0.0,
            "max_abs_observed": 0.0,
            "max_threshold_observed": 0.0,
        }

        def clip_hidden(hidden_states: Any) -> Any:
            if not torch.is_tensor(hidden_states) or hidden_states.ndim < 2:
                return hidden_states
            abs_values = hidden_states.detach().abs().to(torch.float32)
            threshold = torch.quantile(abs_values, RM_CLIP_QUANTILE, dim=-1, keepdim=True)
            mask = abs_values > threshold
            clipped_count = int(mask.sum().item())
            total_count = int(mask.numel())
            clipped = (
                torch.where(
                    mask,
                    torch.sign(hidden_states) * threshold.to(device=hidden_states.device, dtype=hidden_states.dtype),
                    hidden_states,
                )
                if clipped_count
                else hidden_states
            )
            layer_stats = stats["layers"][layer_key]
            layer_stats["invocations"] += 1
            layer_stats["total_values"] += total_count
            layer_stats["clipped_values"] += clipped_count
            layer_stats["clip_fraction"] = layer_stats["clipped_values"] / layer_stats["total_values"]
            layer_stats["max_abs_observed"] = max(
                float(layer_stats["max_abs_observed"]), float(abs_values.max().item())
            )
            layer_stats["max_threshold_observed"] = max(
                float(layer_stats["max_threshold_observed"]), float(threshold.max().item())
            )
            return clipped

        def hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
            if args and torch.is_tensor(args[0]):
                return (clip_hidden(args[0]), *args[1:]), kwargs
            if torch.is_tensor(kwargs.get("hidden_states")):
                updated_kwargs = dict(kwargs)
                updated_kwargs["hidden_states"] = clip_hidden(updated_kwargs["hidden_states"])
                return args, updated_kwargs
            return None

        return hook

    handles = [
        layer.register_forward_pre_hook(make_hook(layer_index, layer_name), with_kwargs=True)
        for layer_index, (layer_name, layer) in enumerate(layers)
        if layer_index in target_set
    ]
    return handles, stats


def generation_rows_with_phase(rows: list[dict[str, Any]], *, phase: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        item = dict(row)
        item["phase"] = phase
        item["ablation_set"] = phase if phase != "baseline" else None
        item["schema_version"] = f"{SCHEMA_VERSION}_generation_row"
        output.append(item)
    return output


def write_generations(path: Path, rows_by_phase: dict[str, list[dict[str, Any]]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for phase in checker.ALL_PHASES:
            for row in rows_by_phase[phase]:
                handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_headroom_diagnostics(full_metrics: dict[str, Any], prompts: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_correct_count = sum(1 for row in full_metrics["per_prompt"] if row["baseline_correct"])
    prompt_count = len(prompts)
    extractor_failure_count = sum(
        1 for row in full_metrics["per_prompt"] if row["baseline_extracted_answer"] is None
    )
    # This is an oracle diagnostic only. It does not affect the gate decision.
    lenient_mention_count = sum(
        1
        for row in full_metrics["per_prompt"]
        if bool(row["baseline_generated_text_contains_canonical"])
    )
    headroom_status = (
        "NO_BASELINE_HEADROOM"
        if baseline_correct_count == 0
        else "LOW_BASELINE_HEADROOM"
        if baseline_correct_count < 6
        else "USABLE_BASELINE_HEADROOM"
    )
    return {
        "schema_version": f"{SCHEMA_VERSION}_headroom_diagnostics",
        "created_at_utc": shared.utc_now(),
        "prompt_count": prompt_count,
        "baseline_correct_count": baseline_correct_count,
        "baseline_accuracy": full_metrics["baseline_accuracy"],
        "full_ablation_accuracy": full_metrics["ablation_accuracy"],
        "extractor_failure_count": extractor_failure_count,
        "lenient_oracle_answer_mention_count": lenient_mention_count,
        "lenient_oracle_accuracy": lenient_mention_count / prompt_count,
        "headroom_status": headroom_status,
        "oracle_answer_key_available": True,
        "oracle_answer_key_correct_count": prompt_count,
        "oracle_answer_key_source": "prompt_manifest canonical AIME-2025 answers",
        "decision_thresholds_unchanged": True,
        "capability_claim_blocked": baseline_correct_count == 0,
        "paper_claim_guard": (
            "A zero-drop gate pass is not evidence of preserved capability when baseline_correct_count is zero; "
            "paper claims must cite this file before interpreting residual clipping."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    shared.RM_SCHEMA_VERSION = SCHEMA_VERSION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-scale", action="store_true", help="Required queue flag; documents Phase 1 scale.")
    parser.add_argument("--run-id", default=f"rm_phase1_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=RM_DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=checker.THRESHOLDS["bootstrap_samples"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = parser.parse_args(argv)

    if not args.full_scale:
        raise SystemExit("Residual Migration Phase 1 requires --full-scale")
    if args.model_id != DEFAULT_MODEL_ID:
        raise SystemExit(f"Residual Migration Phase 1 requires {DEFAULT_MODEL_ID}; got {args.model_id}")
    if args.prompt_file.resolve() != DEFAULT_PROMPT_FILE.resolve():
        raise SystemExit(f"Residual Migration Phase 1 requires canonical prompt file {DEFAULT_PROMPT_FILE}")
    if shared.file_sha256(args.prompt_file) != checker.EXPECTED_PROMPT_FILE_SHA256:
        raise SystemExit("Residual Migration Phase 1 canonical prompt file hash drifted")
    if args.max_new_tokens != RM_DEFAULT_MAX_NEW_TOKENS:
        raise SystemExit(f"Residual Migration Phase 1 freezes --max-new-tokens at {RM_DEFAULT_MAX_NEW_TOKENS}")
    if args.bootstrap_samples != checker.THRESHOLDS["bootstrap_samples"]:
        raise SystemExit("Residual Migration Phase 1 preregisters bootstrap n=1000")
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
        "argv": sys.argv if argv is None else ["run_rm_phase1.py", *argv],
        "cwd": str(Path.cwd()),
        "branch": "residual_migration_phase1",
        "phase0_decision": checker.PHASE0_DECISION,
        "run_dir": str(run_dir),
        "full_scale": args.full_scale,
        "frozen_generation_limit": {
            "max_new_tokens": RM_DEFAULT_MAX_NEW_TOKENS,
            "set_before_analysis": True,
            "reason": "pre-analysis Phase 1 cap for deterministic greedy AIME-2025 scoring",
        },
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "local_files_only": True,
            "prompt_template": "shared AIME solve/final-answer template",
            "batch_size": args.batch_size,
        },
        "headroom_guard": {
            "baseline_accuracy_recorded": True,
            "oracle_answer_key_diagnostic_recorded": True,
            "decision_thresholds_unchanged": True,
        },
    }
    random_seed = {
        "schema_version": f"{SCHEMA_VERSION}_random_seed",
        "seed": args.seed,
        "determinism": {"do_sample": False, "num_beams": 1, "torch_manual_seed": args.seed},
    }
    ablation_config = {
        "schema_version": f"{SCHEMA_VERSION}_ablation_config",
        "created_at_utc": shared.utc_now(),
        "ablation": "residual_stream_hidden_state_95p_clip",
        "clip_quantile": RM_CLIP_QUANTILE,
        "clip_rule": (
            "in targeted transformer-layer forward pre-hooks, values with absolute magnitude above the "
            "per-layer/per-token-position 95th percentile over hidden channels are clipped to that "
            "threshold while preserving sign"
        ),
        "threshold_scope": "per layer, per batch element, per forwarded token position, over hidden channels",
    }
    shared.write_json(run_dir / "prompt_manifest.json", prompt_manifest)
    shared.write_json(run_dir / "environment.json", environment)
    shared.write_json(run_dir / "model_provenance.json", model_provenance)
    shared.write_json(run_dir / "command_metadata.json", command_metadata)
    shared.write_json(run_dir / "random_seed.json", random_seed)
    shared.write_json(run_dir / "ablation_config.json", ablation_config)
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
    layers, layer_origin = shared.discover_transformer_layers(model)
    layer_types = configured_layer_types(model, len(layers))
    layer_groups = build_layer_groups(layer_types)
    command_metadata["layer_origin"] = layer_origin
    command_metadata["layer_count"] = len(layers)
    command_metadata["layer_types"] = layer_types
    shared.write_json(run_dir / "command_metadata.json", command_metadata)

    baseline_raw = shared.run_greedy_generations(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    rows_by_phase: dict[str, list[dict[str, Any]]] = {
        "baseline": generation_rows_with_phase(baseline_raw, phase="baseline")
    }
    run_events_path.open("a", encoding="utf-8").write(
        json.dumps({"created_at_utc": shared.utc_now(), "event": "baseline_completed"}, sort_keys=True) + "\n"
    )

    clip_stats_by_set: dict[str, Any] = {}
    for ablation_set in checker.ABLATION_SETS:
        handles, clip_stats = build_residual_clip_hooks_for_subset(
            layers, layer_groups[ablation_set], ablation_set=ablation_set
        )
        try:
            raw_rows = shared.run_greedy_generations(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
            )
        finally:
            for handle in handles:
                handle.remove()
        rows_by_phase[ablation_set] = generation_rows_with_phase(raw_rows, phase=ablation_set)
        clip_stats_by_set[ablation_set] = clip_stats
        run_events_path.open("a", encoding="utf-8").write(
            json.dumps(
                {"created_at_utc": shared.utc_now(), "event": f"{ablation_set}_completed"},
                sort_keys=True,
            )
            + "\n"
        )

    ablation_config["layer_origin"] = layer_origin
    ablation_config["layer_count"] = len(layers)
    ablation_config["layer_types"] = layer_types
    ablation_config["layer_groups"] = layer_groups
    ablation_config["clip_stats_by_ablation_set"] = clip_stats_by_set
    shared.write_json(run_dir / "ablation_config.json", ablation_config)

    generations_path = run_dir / "generations.jsonl"
    write_generations(generations_path, rows_by_phase)
    stratified_by_set = {
        ablation_set: checker.compute_ablation_metrics(
            rows_by_phase["baseline"],
            rows_by_phase[ablation_set],
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            ablation_set=ablation_set,
        )
        for ablation_set in checker.ABLATION_SETS
    }
    full_metrics = stratified_by_set["full_ablation"]
    metrics = {
        **full_metrics,
        "schema_version": f"{SCHEMA_VERSION}_metrics",
        "metric_name": "aime_accuracy_drop_after_residual_95p_clip",
        "created_at_utc": shared.utc_now(),
        "branch": "residual_migration_phase1",
        "phase0_decision": checker.PHASE0_DECISION,
        "model_id": args.model_id,
        "model_snapshot_commit": model_provenance.get("hf_snapshot_commit"),
        "prompt_source": "AIME-2025",
        "prompt_selection": "deterministic_indices_0_23",
        "prompt_sha256": prompt_manifest["prompt_sha256"],
        "trace_count": len(prompts),
        "generation_artifact": "generations.jsonl",
        "generation_artifact_sha256": shared.file_sha256(generations_path),
        "thresholds": checker.THRESHOLDS,
        "aggregation": {
            "prompt_level": "binary exact-match correctness",
            "gate_level": "mean baseline correctness minus mean full-ablation correctness across 24 prompts",
            "bootstrap_unit": "prompt",
            "stratified_ablation_sets": list(checker.ABLATION_SETS),
        },
    }
    headroom = build_headroom_diagnostics(full_metrics, prompts)
    shared.write_json(run_dir / "metrics.json", metrics)
    shared.write_json(
        run_dir / "bootstrap_ci.json",
        {
            "schema_version": f"{SCHEMA_VERSION}_bootstrap_ci",
            "metric_name": metrics["metric_name"],
            "bootstrap_samples": metrics["bootstrap_samples"],
            "bootstrap_seed": metrics["bootstrap_seed"],
            "bootstrap_ci95": metrics["bootstrap_ci95"],
            "accuracy_drop": metrics["accuracy_drop"],
        },
    )
    shared.write_json(
        run_dir / "stratified_metrics.json",
        {
            "schema_version": f"{SCHEMA_VERSION}_stratified_metrics",
            "metric_name": "aime_accuracy_drop_by_residual_clip_layer_group",
            "ablation_sets": stratified_by_set,
            "baseline_accuracy": full_metrics["baseline_accuracy"],
            "layer_groups": layer_groups,
            "attribution_only": True,
            "decision_ablation_set": "full_ablation",
        },
    )
    shared.write_json(run_dir / "headroom_diagnostics.json", headroom)
    run_events_path.open("a", encoding="utf-8").write(
        json.dumps({"created_at_utc": shared.utc_now(), "event": "run_completed"}, sort_keys=True) + "\n"
    )
    print(json.dumps({"run_dir": str(run_dir), "metrics": metrics, "headroom": headroom}, indent=2, sort_keys=True))
    sys.stdout.flush()
    sys.stderr.flush()
    shared.write_json(run_dir / "artifact_hashes.json", shared.build_artifact_hashes(run_dir, schema_version=SCHEMA_VERSION))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
