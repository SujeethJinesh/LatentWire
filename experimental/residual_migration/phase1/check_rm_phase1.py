#!/usr/bin/env python3
"""Check Residual Migration Phase 1 result packets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "experimental/residual_migration/phase1/results"
DEFAULT_PROMPT_FILE = ROOT / "experimental/shared/prompts/aime_2025_indices_0_23.jsonl"
MODEL_ID = "ibm-granite/granite-4.0-h-small"
TRACE_COUNT = 24
EXPECTED_PROMPT_SOURCE_DATASET = "opencompass/AIME2025"
EXPECTED_PROMPT_SOURCE_COMMIT = "a6ad95f611d72cf628a80b58bd0432ef6638f958"
EXPECTED_PROMPT_FILE_SHA256 = "sha256:ead004dae0848ad43ad102551f48fa22a0b8ed4a57efecdcf9d7ae387bb6d17a"
EXPECTED_PROMPT_PAYLOAD_SHA256 = "sha256:aa038b29332b6d137d558205ee441163e7ea4cb3cc323eb705a2f5928fd2fe4e"
SCHEMA_VERSION = "rm_phase1_v1"
PHASE0_DECISION = "PASS_RM_PHASE0_RETHINKING_REPLICATES"

PASS_REPLICATED = "PASS_RM_PHASE1_REPLICATED_AT_SCALE"
KILL_FAILED_AT_SCALE = "KILL_RM_PHASE1_FAILED_AT_SCALE"
FAIL_INFRA = "FAIL_INFRA_RM_PHASE1"
PREREG_AMBIGUOUS = "PREREG_AMBIGUOUS_RM_PHASE1_CI_OVERLAP"

THRESHOLDS = {
    "replicates_ci_upper_lt": 0.015,
    "bootstrap_samples": 1000,
    "bootstrap_ci": 0.95,
}
ABLATION_SETS = ("full_ablation", "first_half", "second_half", "attention_only", "mamba_only")
ALL_PHASES = ("baseline", *ABLATION_SETS)
REQUIRED_FILES = [
    "environment.json",
    "model_provenance.json",
    "prompt_manifest.json",
    "command_metadata.json",
    "random_seed.json",
    "ablation_config.json",
    "generations.jsonl",
    "metrics.json",
    "bootstrap_ci.json",
    "stratified_metrics.json",
    "headroom_diagnostics.json",
    "artifact_hashes.json",
    "logs/stdout.log",
    "logs/stderr.log",
    "run_events.jsonl",
]
HASHED_FILES = [rel for rel in REQUIRED_FILES if rel != "artifact_hashes.json"]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def is_close(left: float, right: float, *, tol: float = 1e-9) -> bool:
    return abs(left - right) <= tol * max(1.0, abs(left), abs(right))


def latest_run_dir() -> Path:
    candidates = [path for path in RESULTS_DIR.iterdir() if path.is_dir()] if RESULTS_DIR.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"no Residual Migration Phase 1 result dirs found under {RESULTS_DIR}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def expected_source_file(index: int) -> str:
    return "aime2025-I.jsonl" if index < 15 else "aime2025-II.jsonl"


def expected_prompt_id(index: int) -> str:
    if index < 15:
        return f"opencompass_AIME2025_I_{index}"
    return f"opencompass_AIME2025_II_{index - 15}"


def prompt_payload_sha256(prompts: list[dict[str, Any]]) -> str:
    ordered = sorted(prompts, key=lambda row: int(row["index"]))
    payload = "".join(str(row["prompt"]) for row in ordered).encode("utf-8")
    return bytes_sha256(payload)


def normalize_aime_answer(answer: Any) -> str:
    text = str(answer).strip()
    if text.isdigit():
        return str(int(text))
    return text


def extract_aime_answer(text: str) -> str | None:
    import re

    boxed = re.findall(r"\\boxed\{?\s*([0-9]{1,3})\s*\}?", text)
    if boxed:
        return normalize_aime_answer(boxed[-1])
    final_patterns = [
        r"(?:final answer|answer is|answer:)\s*(?:is\s*)?(?:\$?\\?boxed\{?\s*)?([0-9]{1,3})",
        r"(?:therefore|thus|so),?\s*the answer is\s*(?:\$?\\?boxed\{?\s*)?([0-9]{1,3})",
    ]
    for pattern in final_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return normalize_aime_answer(matches[-1])
    return None


def iter_generation_rows(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                row = json.loads(line)
                row["_line_number"] = line_number
                yield row


def compute_ablation_metrics(
    baseline_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
    ablation_set: str,
) -> dict[str, Any]:
    baseline = {int(row["prompt_index"]): row for row in baseline_rows}
    ablation = {int(row["prompt_index"]): row for row in ablation_rows}
    per_prompt: list[dict[str, Any]] = []
    for prompt_index in sorted(baseline):
        base = baseline[prompt_index]
        ablated = ablation[prompt_index]
        base_correct = 1.0 if bool(base["correct"]) else 0.0
        ablation_correct = 1.0 if bool(ablated["correct"]) else 0.0
        per_prompt.append(
            {
                "prompt_index": prompt_index,
                "prompt_id": base["prompt_id"],
                "canonical_answer": base["canonical_answer"],
                "baseline_extracted_answer": base["extracted_answer"],
                "ablation_extracted_answer": ablated["extracted_answer"],
                "baseline_correct": bool(base["correct"]),
                "ablation_correct": bool(ablated["correct"]),
                "baseline_extractor_failed": base["extracted_answer"] is None,
                "baseline_generated_text_contains_canonical": str(base["canonical_answer"])
                in str(base.get("generated_text", "")),
                "drop": base_correct - ablation_correct,
            }
        )
    drops = [float(row["drop"]) for row in per_prompt]
    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(bootstrap_samples):
        sample = [drops[rng.randrange(len(drops))] for _ in drops]
        boot.append(float(mean(sample)))
    boot.sort()
    baseline_accuracy = mean(1.0 if bool(row["correct"]) else 0.0 for row in baseline.values())
    ablation_accuracy = mean(1.0 if bool(row["correct"]) else 0.0 for row in ablation.values())
    return {
        "ablation_set": ablation_set,
        "baseline_accuracy": float(baseline_accuracy),
        "ablation_accuracy": float(ablation_accuracy),
        "accuracy_drop": float(baseline_accuracy - ablation_accuracy),
        "bootstrap_ci95": {
            "ci95_low": boot[int(0.025 * (len(boot) - 1))],
            "ci95_high": boot[int(0.975 * (len(boot) - 1))],
        },
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "per_prompt": per_prompt,
    }


def decision_from_full_metrics(point: float, ci_high: float) -> tuple[str, list[str]]:
    threshold = float(THRESHOLDS["replicates_ci_upper_lt"])
    if ci_high < threshold:
        return PASS_REPLICATED, [
            f"replicated pass: ci_high={ci_high:.8f} < {threshold:.3f}"
        ]
    if point >= threshold:
        return KILL_FAILED_AT_SCALE, [
            f"failed at scale: point={point:.8f} >= {threshold:.3f}"
        ]
    return PREREG_AMBIGUOUS, [
        f"preregistration ambiguous: point={point:.8f} < {threshold:.3f} but ci_high={ci_high:.8f} >= {threshold:.3f}; "
        "Phase 1 preregistration defines PASS for CI upper < threshold and KILL for point drop >= threshold"
    ]


def validate_prompt_manifest(prompt_manifest: dict[str, Any], metrics: dict[str, Any], infra: list[str]) -> set[int]:
    if prompt_manifest.get("source") != "AIME-2025":
        infra.append("prompt_manifest.source must be AIME-2025")
    if prompt_manifest.get("selection") != "deterministic_indices_0_23":
        infra.append("prompt_manifest.selection must be deterministic_indices_0_23")
    if file_sha256(DEFAULT_PROMPT_FILE) != EXPECTED_PROMPT_FILE_SHA256:
        infra.append("local canonical AIME-2025 indices 0-23 prompt file hash drifted")
    if prompt_manifest.get("prompt_file_sha256") != EXPECTED_PROMPT_FILE_SHA256:
        infra.append("prompt_manifest.prompt_file_sha256 must match frozen AIME-2025 indices 0-23 file hash")
    if prompt_manifest.get("prompt_sha256_semantics") != "sha256 of concatenated prompt text in deterministic index order":
        infra.append("prompt_manifest.prompt_sha256_semantics mismatch")
    prompts = prompt_manifest.get("prompts", [])
    if not isinstance(prompts, list):
        infra.append("prompt_manifest.prompts must be a list")
        prompts = []
    indices: set[int] = set()
    for row in prompts:
        if not isinstance(row, dict):
            infra.append("prompt_manifest contains a non-object row")
            continue
        try:
            index = int(row["index"])
        except Exception:
            infra.append("prompt_manifest row missing integer index")
            continue
        indices.add(index)
        if row.get("prompt_id") != expected_prompt_id(index):
            infra.append(f"prompt {index}: prompt_id mismatch")
        if row.get("source_dataset") != EXPECTED_PROMPT_SOURCE_DATASET:
            infra.append(f"prompt {index}: source_dataset must be {EXPECTED_PROMPT_SOURCE_DATASET}")
        if row.get("source_file") != expected_source_file(index):
            infra.append(f"prompt {index}: source_file mismatch")
        if row.get("source_commit") != EXPECTED_PROMPT_SOURCE_COMMIT:
            infra.append(f"prompt {index}: source_commit must be {EXPECTED_PROMPT_SOURCE_COMMIT}")
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            infra.append(f"prompt {index}: missing prompt text")
    if [int(row.get("index", -1)) for row in prompts if isinstance(row, dict)] != list(range(TRACE_COUNT)):
        infra.append("prompt manifest row order must be deterministic indices 0-23")
    if sorted(indices) != list(range(TRACE_COUNT)):
        infra.append("prompt indices must be exactly 0-23")
    if len(prompts) != TRACE_COUNT or int(prompt_manifest.get("prompt_count", -1)) != TRACE_COUNT:
        infra.append("prompt_manifest must contain exactly 24 prompts")
    if prompts:
        expected_sha = prompt_payload_sha256(prompts)
        if prompt_manifest.get("prompt_sha256") != expected_sha:
            infra.append("prompt_manifest.prompt_sha256 mismatch")
        if prompt_manifest.get("prompt_sha256") != EXPECTED_PROMPT_PAYLOAD_SHA256:
            infra.append("prompt_manifest.prompt_sha256 must match frozen AIME-2025 indices 0-23 payload hash")
    if metrics.get("prompt_source") != "AIME-2025":
        infra.append("metrics.prompt_source must be AIME-2025")
    if metrics.get("prompt_selection") != "deterministic_indices_0_23":
        infra.append("metrics.prompt_selection mismatch")
    if metrics.get("prompt_sha256") != prompt_manifest.get("prompt_sha256"):
        infra.append("metrics.prompt_sha256 must match prompt_manifest")
    return indices


def validate_artifact_hashes(run_dir: Path, artifact_hashes: dict[str, Any], infra: list[str]) -> None:
    entries = artifact_hashes.get("artifacts", [])
    if not isinstance(entries, list):
        infra.append("artifact_hashes.artifacts must be a list")
        return
    by_path = {str(row.get("path")): row for row in entries if isinstance(row, dict)}
    for rel in HASHED_FILES:
        item = by_path.get(rel)
        path = run_dir / rel
        if item is None:
            infra.append(f"artifact_hashes missing {rel}")
            continue
        if not path.is_file():
            infra.append(f"hashed artifact missing on disk: {rel}")
            continue
        if item.get("bytes") != path.stat().st_size:
            infra.append(f"artifact_hashes byte mismatch for {rel}")
        if item.get("sha256") != file_sha256(path):
            infra.append(f"artifact_hashes sha256 mismatch for {rel}")


def validate_generation_rows(
    rows: list[dict[str, Any]],
    *,
    prompt_manifest: dict[str, Any],
    command: dict[str, Any],
    infra: list[str],
) -> dict[str, list[dict[str, Any]]]:
    prompts = {
        int(row["index"]): row
        for row in prompt_manifest.get("prompts", [])
        if isinstance(row, dict) and "index" in row
    }
    by_phase: dict[str, list[dict[str, Any]]] = {phase: [] for phase in ALL_PHASES}
    if len(rows) != TRACE_COUNT * len(ALL_PHASES):
        infra.append(f"generations.jsonl must contain exactly {TRACE_COUNT * len(ALL_PHASES)} rows")
    seen: set[tuple[str, int]] = set()
    for row in rows:
        phase = str(row.get("phase", ""))
        if phase not in by_phase:
            infra.append(f"generation row {row.get('_line_number')}: invalid phase {phase!r}")
            continue
        by_phase[phase].append(row)
        try:
            prompt_index = int(row["prompt_index"])
        except Exception:
            infra.append(f"generation row {row.get('_line_number')}: missing integer prompt_index")
            continue
        key = (phase, prompt_index)
        if key in seen:
            infra.append(f"duplicate {phase} generation row for prompt {prompt_index}")
        seen.add(key)
        prompt = prompts.get(prompt_index)
        if prompt is None:
            infra.append(f"generation row prompt_index {prompt_index} not in prompt manifest")
            continue
        if row.get("prompt_id") != prompt.get("prompt_id"):
            infra.append(f"generation row prompt_id mismatch for prompt {prompt_index}")
        canonical = normalize_aime_answer(prompt.get("answer"))
        if row.get("canonical_answer") != canonical:
            infra.append(f"generation row canonical_answer mismatch for prompt {prompt_index}")
        if not isinstance(row.get("generated_text"), str):
            infra.append(f"generation row {phase}/{prompt_index} missing generated_text")
            continue
        extracted = extract_aime_answer(row["generated_text"])
        correct = extracted == canonical
        if row.get("extracted_answer") != extracted:
            infra.append(f"generation row extracted_answer mismatch for {phase}/{prompt_index}")
        if bool(row.get("correct")) != correct:
            infra.append(f"generation row correct flag mismatch for {phase}/{prompt_index}")
        if int(row.get("max_new_tokens", -1)) != int(command.get("frozen_generation_limit", {}).get("max_new_tokens", -2)):
            infra.append(f"generation row max_new_tokens mismatch for {phase}/{prompt_index}")
    expected_indices = set(range(TRACE_COUNT))
    for phase in ALL_PHASES:
        phase_indices = {int(row.get("prompt_index", -1)) for row in by_phase[phase]}
        if phase_indices != expected_indices:
            infra.append(f"{phase} generation prompt indices must be exactly 0-23")
        by_phase[phase].sort(key=lambda row: int(row["prompt_index"]))
    return by_phase


def validate_ablation_config(ablation_config: dict[str, Any], command: dict[str, Any], infra: list[str]) -> None:
    if float(ablation_config.get("clip_quantile", -1.0)) != 0.95:
        infra.append("ablation_config.clip_quantile must be 0.95")
    if "forward pre-hook" not in str(ablation_config.get("clip_rule", "")):
        infra.append("ablation_config.clip_rule must document the forward pre-hook")
    try:
        layer_count = int(command["layer_count"])
    except Exception:
        layer_count = 0
        infra.append("command_metadata.layer_count must record discovered transformer layer count")
    if int(ablation_config.get("layer_count", -1)) != layer_count:
        infra.append("ablation_config.layer_count must match command_metadata.layer_count")
    groups = ablation_config.get("layer_groups")
    if not isinstance(groups, dict):
        infra.append("ablation_config.layer_groups must be present")
        groups = {}
    expected_all = set(range(layer_count))
    expected_first = set(range(layer_count // 2))
    expected_second = set(range(layer_count // 2, layer_count))
    for name, expected in {
        "full_ablation": expected_all,
        "first_half": expected_first,
        "second_half": expected_second,
    }.items():
        observed = {int(value) for value in groups.get(name, [])} if isinstance(groups.get(name), list) else set()
        if observed != expected:
            infra.append(f"ablation_config.layer_groups.{name} mismatch")
    attention = {int(value) for value in groups.get("attention_only", [])} if isinstance(groups.get("attention_only"), list) else set()
    mamba = {int(value) for value in groups.get("mamba_only", [])} if isinstance(groups.get("mamba_only"), list) else set()
    if not attention:
        infra.append("ablation_config.layer_groups.attention_only must be nonempty")
    if not mamba:
        infra.append("ablation_config.layer_groups.mamba_only must be nonempty")
    if attention & mamba:
        infra.append("attention_only and mamba_only layer groups must be disjoint")
    if attention | mamba != expected_all:
        infra.append("attention_only and mamba_only layer groups must partition all layers")

    stats_by_set = ablation_config.get("clip_stats_by_ablation_set")
    if not isinstance(stats_by_set, dict):
        infra.append("ablation_config.clip_stats_by_ablation_set must be present")
        return
    for ablation_set in ABLATION_SETS:
        stats = stats_by_set.get(ablation_set)
        if not isinstance(stats, dict):
            infra.append(f"missing clip stats for {ablation_set}")
            continue
        if stats.get("hook_type") != "forward_pre_hook":
            infra.append(f"{ablation_set} clip_stats.hook_type must be forward_pre_hook")
        target = {int(value) for value in stats.get("target_layer_indices", [])}
        expected_target = {int(value) for value in groups.get(ablation_set, [])}
        if target != expected_target:
            infra.append(f"{ablation_set} target_layer_indices must match layer_groups")
        layers = stats.get("layers")
        if not isinstance(layers, dict) or set(layers) != {str(index) for index in expected_target}:
            infra.append(f"{ablation_set} clip_stats.layers must exactly match targeted layers")
            continue
        for layer_key, layer_stats in layers.items():
            if not isinstance(layer_stats, dict):
                infra.append(f"{ablation_set} layer stats {layer_key} must be an object")
                continue
            if int(layer_stats.get("invocations", 0) or 0) <= 0:
                infra.append(f"{ablation_set} layer {layer_key} must have at least one hook invocation")
            total = int(layer_stats.get("total_values", -1))
            clipped = int(layer_stats.get("clipped_values", -1))
            fraction = float(layer_stats.get("clip_fraction", -1.0))
            if total <= 0:
                infra.append(f"{ablation_set} layer {layer_key} total_values must be positive")
            if clipped < 0 or clipped > total:
                infra.append(f"{ablation_set} layer {layer_key} clipped_values out of range")
            if total > 0 and not is_close(fraction, clipped / total):
                infra.append(f"{ablation_set} layer {layer_key} clip_fraction mismatch")


def validate_headroom(headroom: dict[str, Any], computed_full: dict[str, Any], infra: list[str]) -> None:
    if headroom.get("schema_version") != f"{SCHEMA_VERSION}_headroom_diagnostics":
        infra.append("headroom_diagnostics schema_version mismatch")
    baseline_correct_count = sum(1 for row in computed_full["per_prompt"] if row["baseline_correct"])
    extractor_failure_count = sum(
        1 for row in computed_full["per_prompt"] if row["baseline_extracted_answer"] is None
    )
    lenient_mention_count = sum(
        1
        for row in computed_full["per_prompt"]
        if bool(row["baseline_generated_text_contains_canonical"])
    )
    if int(headroom.get("baseline_correct_count", -1)) != baseline_correct_count:
        infra.append("headroom baseline_correct_count mismatch")
    if int(headroom.get("prompt_count", -1)) != TRACE_COUNT:
        infra.append("headroom prompt_count must be 24")
    if not is_close(float(headroom.get("baseline_accuracy", -1.0)), float(computed_full["baseline_accuracy"])):
        infra.append("headroom baseline_accuracy mismatch")
    if int(headroom.get("extractor_failure_count", -1)) != extractor_failure_count:
        infra.append("headroom extractor_failure_count mismatch")
    if int(headroom.get("lenient_oracle_answer_mention_count", -1)) != lenient_mention_count:
        infra.append("headroom lenient_oracle_answer_mention_count mismatch")
    expected_lenient_accuracy = lenient_mention_count / TRACE_COUNT
    if not is_close(float(headroom.get("lenient_oracle_accuracy", -1.0)), expected_lenient_accuracy):
        infra.append("headroom lenient_oracle_accuracy mismatch")
    if bool(headroom.get("oracle_answer_key_available")) is not True:
        infra.append("headroom must document oracle_answer_key_available=true")
    if int(headroom.get("oracle_answer_key_correct_count", -1)) != TRACE_COUNT:
        infra.append("headroom oracle_answer_key_correct_count must equal 24")
    if headroom.get("oracle_answer_key_source") != "prompt_manifest canonical AIME-2025 answers":
        infra.append("headroom oracle_answer_key_source mismatch")
    if bool(headroom.get("decision_thresholds_unchanged")) is not True:
        infra.append("headroom must document decision_thresholds_unchanged=true")
    expected_blocked = baseline_correct_count == 0
    if bool(headroom.get("capability_claim_blocked")) != expected_blocked:
        infra.append("headroom capability_claim_blocked must reflect zero baseline correctness")
    expected_status = (
        "NO_BASELINE_HEADROOM"
        if baseline_correct_count == 0
        else "LOW_BASELINE_HEADROOM"
        if baseline_correct_count < 6
        else "USABLE_BASELINE_HEADROOM"
    )
    if headroom.get("headroom_status") != expected_status:
        infra.append("headroom headroom_status mismatch")


def infra_result(run_dir: Path, reasons: list[str]) -> dict[str, Any]:
    result = {
        "decision": FAIL_INFRA,
        "run_dir": str(run_dir),
        "reasons": reasons,
        "artifact_complete": False,
    }
    if run_dir.is_dir():
        write_json(run_dir / "checker_result.json", result)
        write_json(
            run_dir / "artifact_check.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_artifact_check",
                "decision": FAIL_INFRA,
                "run_dir": str(run_dir),
                "required_files": REQUIRED_FILES,
                "artifact_complete": False,
                "reasons": reasons,
            },
        )
    return result


def evaluate(run_dir: Path) -> dict[str, Any]:
    infra: list[str] = []
    for rel in REQUIRED_FILES:
        if not (run_dir / rel).is_file():
            infra.append(f"missing required artifact: {rel}")
    try:
        environment = load_json(run_dir / "environment.json")
        model = load_json(run_dir / "model_provenance.json")
        prompt_manifest = load_json(run_dir / "prompt_manifest.json")
        command = load_json(run_dir / "command_metadata.json")
        random_seed = load_json(run_dir / "random_seed.json")
        ablation_config = load_json(run_dir / "ablation_config.json")
        metrics = load_json(run_dir / "metrics.json")
        bootstrap_ci = load_json(run_dir / "bootstrap_ci.json")
        stratified = load_json(run_dir / "stratified_metrics.json")
        headroom = load_json(run_dir / "headroom_diagnostics.json")
        artifact_hashes = load_json(run_dir / "artifact_hashes.json")
    except Exception as exc:
        return infra_result(run_dir, [*infra, f"bad JSON artifacts: {exc!r}"])

    if environment.get("schema_version") != f"{SCHEMA_VERSION}_environment":
        infra.append("environment schema_version mismatch")
    if command.get("branch") != "residual_migration_phase1":
        infra.append("command_metadata.branch must be residual_migration_phase1")
    if command.get("phase0_decision") != PHASE0_DECISION:
        infra.append(f"command_metadata.phase0_decision must be {PHASE0_DECISION}")
    frozen_limit = command.get("frozen_generation_limit", {})
    if int(frozen_limit.get("max_new_tokens", 0) or 0) != 2048:
        infra.append("command_metadata.frozen_generation_limit.max_new_tokens must be 2048")
    if frozen_limit.get("set_before_analysis") is not True:
        infra.append("command_metadata must document a pre-analysis frozen generation limit")
    generation = command.get("generation", {})
    if generation.get("do_sample") is not False or int(generation.get("num_beams", 0)) != 1:
        infra.append("command_metadata.generation must document deterministic greedy decoding")
    if generation.get("local_files_only") is not True:
        infra.append("command_metadata.generation.local_files_only must be true")
    headroom_guard = command.get("headroom_guard", {})
    if headroom_guard.get("baseline_accuracy_recorded") is not True:
        infra.append("command_metadata.headroom_guard.baseline_accuracy_recorded must be true")
    if headroom_guard.get("oracle_answer_key_diagnostic_recorded") is not True:
        infra.append("command_metadata.headroom_guard.oracle_answer_key_diagnostic_recorded must be true")
    if headroom_guard.get("decision_thresholds_unchanged") is not True:
        infra.append("command_metadata.headroom_guard.decision_thresholds_unchanged must be true")
    if model.get("model_id") != MODEL_ID:
        infra.append("model_provenance.model_id mismatch")
    if model.get("local_files_only") is not True:
        infra.append("model provenance must record local_files_only true")
    if not model.get("hf_snapshot_commit") or not model.get("snapshot_path"):
        infra.append("model provenance must record local HF snapshot commit and path")
    if metrics.get("branch") != "residual_migration_phase1":
        infra.append("metrics.branch must be residual_migration_phase1")
    if metrics.get("phase0_decision") != PHASE0_DECISION:
        infra.append(f"metrics.phase0_decision must be {PHASE0_DECISION}")
    if metrics.get("model_id") != MODEL_ID:
        infra.append("metrics.model_id mismatch")
    if metrics.get("thresholds") != THRESHOLDS:
        infra.append("metrics.thresholds mismatch preregistered residual Phase 1 thresholds")
    if int(metrics.get("bootstrap_samples", 0) or 0) != THRESHOLDS["bootstrap_samples"]:
        infra.append("metrics.bootstrap_samples must be 1000")
    if int(bootstrap_ci.get("bootstrap_samples", 0) or 0) != THRESHOLDS["bootstrap_samples"]:
        infra.append("bootstrap_ci.bootstrap_samples must be 1000")
    if int(random_seed.get("seed", -1)) != 20260508:
        infra.append("random_seed.seed must be 20260508")
    if int(random_seed.get("seed", -1)) != int(metrics.get("bootstrap_seed", -2)):
        infra.append("random_seed.seed must match metrics.bootstrap_seed")

    validate_prompt_manifest(prompt_manifest, metrics, infra)
    validate_ablation_config(ablation_config, command, infra)
    validate_artifact_hashes(run_dir, artifact_hashes, infra)
    rows: list[dict[str, Any]] = []
    try:
        rows = list(iter_generation_rows(run_dir / "generations.jsonl"))
    except Exception as exc:
        infra.append(f"cannot read generations.jsonl: {exc!r}")
    if not rows:
        infra.append("generations.jsonl must contain generation rows")
    by_phase: dict[str, list[dict[str, Any]]] = {phase: [] for phase in ALL_PHASES}
    if rows:
        by_phase = validate_generation_rows(
            rows,
            prompt_manifest=prompt_manifest,
            command=command,
            infra=infra,
        )
        if metrics.get("generation_artifact") != "generations.jsonl":
            infra.append("metrics.generation_artifact must be generations.jsonl")
        if metrics.get("generation_artifact_sha256") != file_sha256(run_dir / "generations.jsonl"):
            infra.append("metrics.generation_artifact_sha256 mismatch")

    computed_by_set: dict[str, dict[str, Any]] = {}
    if rows and not infra:
        seed = int(metrics["bootstrap_seed"])
        for ablation_set in ABLATION_SETS:
            computed_by_set[ablation_set] = compute_ablation_metrics(
                by_phase["baseline"],
                by_phase[ablation_set],
                bootstrap_samples=THRESHOLDS["bootstrap_samples"],
                seed=seed,
                ablation_set=ablation_set,
            )
        computed_full = computed_by_set["full_ablation"]
        for key in ["baseline_accuracy", "ablation_accuracy", "accuracy_drop"]:
            if not is_close(float(metrics[key]), float(computed_full[key])):
                infra.append(f"metrics.{key} does not match recomputation")
        for key in ["ci95_low", "ci95_high"]:
            if not is_close(float(metrics["bootstrap_ci95"][key]), float(computed_full["bootstrap_ci95"][key])):
                infra.append(f"metrics.bootstrap_ci95.{key} does not match recomputation")
            if not is_close(float(bootstrap_ci["bootstrap_ci95"][key]), float(computed_full["bootstrap_ci95"][key])):
                infra.append(f"bootstrap_ci.bootstrap_ci95.{key} does not match recomputation")
        if not is_close(float(bootstrap_ci["accuracy_drop"]), float(computed_full["accuracy_drop"])):
            infra.append("bootstrap_ci.accuracy_drop does not match recomputation")
        if metrics.get("per_prompt") != computed_full.get("per_prompt"):
            infra.append("metrics.per_prompt does not match recomputation")
        if stratified.get("ablation_sets") != computed_by_set:
            infra.append("stratified_metrics.ablation_sets does not match recomputation")
        validate_headroom(headroom, computed_full, infra)

    if infra:
        result = {
            "decision": FAIL_INFRA,
            "run_dir": str(run_dir),
            "reasons": infra,
            "artifact_complete": False,
        }
    else:
        computed_full = computed_by_set["full_ablation"]
        point = float(computed_full["accuracy_drop"])
        ci_high = float(computed_full["bootstrap_ci95"]["ci95_high"])
        decision, reasons = decision_from_full_metrics(point, ci_high)
        headroom_warning = bool(headroom.get("capability_claim_blocked"))
        result = {
            "decision": decision,
            "run_dir": str(run_dir),
            "reasons": reasons,
            "artifact_complete": True,
            "baseline_accuracy": computed_full["baseline_accuracy"],
            "ablation_accuracy": computed_full["ablation_accuracy"],
            "accuracy_drop": computed_full["accuracy_drop"],
            "bootstrap_ci95": computed_full["bootstrap_ci95"],
            "per_prompt": computed_full["per_prompt"],
            "stratified_accuracy_drops": {
                name: computed_by_set[name]["accuracy_drop"] for name in ABLATION_SETS
            },
            "headroom_warning": headroom_warning,
            "headroom_diagnostics": headroom,
            "thresholds": THRESHOLDS,
        }
    artifact_check = {
        "schema_version": f"{SCHEMA_VERSION}_artifact_check",
        "decision": result["decision"],
        "run_dir": str(run_dir),
        "required_files": REQUIRED_FILES,
        "artifact_complete": result.get("artifact_complete", False),
        "reasons": result["reasons"],
    }
    write_json(run_dir / "checker_result.json", result)
    write_json(run_dir / "artifact_check.json", artifact_check)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve() if args.run_dir else latest_run_dir().resolve()
    result = evaluate(run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"] == PASS_REPLICATED else 1


if __name__ == "__main__":
    raise SystemExit(main())
