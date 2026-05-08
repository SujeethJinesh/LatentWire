#!/usr/bin/env python3
"""Check shared Phase 0 gate packets.

Only the OutlierMigrate Phase 0 decision rule is implemented.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase0/results"
DEFAULT_PROMPT_FILE = ROOT / "experimental/shared/prompts/aime_2025_indices_0_11.jsonl"
MODEL_ID = "ibm-granite/granite-4.0-h-tiny"
POSITIONS = (100, 500, 1000, 5000, 10000)
TRACE_COUNT = 12
EXPECTED_PROMPT_SOURCE_DATASET = "opencompass/AIME2025"
EXPECTED_PROMPT_SOURCE_FILE = "aime2025-I.jsonl"
EXPECTED_PROMPT_SOURCE_COMMIT = "a6ad95f611d72cf628a80b58bd0432ef6638f958"
SCHEMA_VERSION = "om_phase0_v1"

PASS_DYNAMIC = "PASS_OM_PHASE0_DECODE_TIME_MIGRATION"
PASS_STATIC = "PASS_OM_PHASE0_STATIC_OUTLIERS"
KILL_AMBIGUOUS = "KILL_OM_PHASE0_AMBIGUOUS_EFFECT_SIZE"
FAIL_INFRA = "FAIL_INFRA_OM_PHASE0"

THRESHOLDS = {
    "migration_fraction_threshold": 0.05,
    "rank_delta_strictly_greater_than": 2,
    "top_channel_fraction": 0.01,
    "bootstrap_samples": 1000,
    "bootstrap_ci": 0.95,
}

REQUIRED_FILES = [
    "environment.json",
    "model_provenance.json",
    "prompt_manifest.json",
    "command_metadata.json",
    "random_seed.json",
    "activation_magnitude_manifest.json",
    "activation_magnitudes.jsonl.gz",
    "metrics.json",
    "bootstrap_ci.json",
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
        raise FileNotFoundError(f"no OutlierMigrate Phase 0 result dirs found under {RESULTS_DIR}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def prompt_payload_sha256(prompts: list[dict[str, Any]]) -> str:
    ordered = sorted(prompts, key=lambda row: int(row["index"]))
    payload = "".join(str(row["prompt"]) for row in ordered).encode("utf-8")
    return bytes_sha256(payload)


def iter_activation_rows(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                row = json.loads(line)
                row["_line_number"] = line_number
                yield row


def ranks_desc(values: list[float]) -> list[int]:
    ordered = sorted(range(len(values)), key=lambda index: (-float(values[index]), index))
    ranks = [0] * len(values)
    for rank, channel in enumerate(ordered):
        ranks[channel] = rank
    return ranks


def select_top_channels_by_layer(
    by_trace_layer: dict[int, dict[int, dict[int, list[float]]]]
) -> dict[int, list[int]]:
    top_fraction = float(THRESHOLDS["top_channel_fraction"])
    layer_indices = sorted({layer for trace in by_trace_layer.values() for layer in trace})
    selected: dict[int, list[int]] = {}
    for layer_index in layer_indices:
        base_vectors = [
            by_trace_layer[prompt_index][layer_index][POSITIONS[0]]
            for prompt_index in sorted(by_trace_layer)
            if layer_index in by_trace_layer[prompt_index]
        ]
        if not base_vectors:
            continue
        channel_count = len(base_vectors[0])
        top_k = max(1, math.ceil(channel_count * top_fraction))
        mean_magnitudes = [
            mean(float(vector[channel]) for vector in base_vectors) for channel in range(channel_count)
        ]
        selected[layer_index] = sorted(
            range(channel_count), key=lambda channel: (-mean_magnitudes[channel], channel)
        )[:top_k]
    return selected


def compute_metrics(rows: list[dict[str, Any]], *, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    by_trace_layer: dict[int, dict[int, dict[int, list[float]]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        by_trace_layer[int(row["prompt_index"])][int(row["layer_index"])][int(row["decode_position"])] = [
            float(value) for value in row["channel_magnitudes"]
        ]
    trace_metrics: list[dict[str, Any]] = []
    layer_metrics: dict[int, list[float]] = defaultdict(list)
    rank_delta = int(THRESHOLDS["rank_delta_strictly_greater_than"])
    top_channels_by_layer = select_top_channels_by_layer(by_trace_layer)
    for prompt_index in sorted(by_trace_layer):
        layer_fractions: list[float] = []
        for layer_index in sorted(by_trace_layer[prompt_index]):
            base = by_trace_layer[prompt_index][layer_index][POSITIONS[0]]
            final = by_trace_layer[prompt_index][layer_index][POSITIONS[-1]]
            base_ranks = ranks_desc(base)
            final_ranks = ranks_desc(final)
            base_top_channels = top_channels_by_layer[layer_index]
            migrated = sum(
                1 for channel in base_top_channels if abs(final_ranks[channel] - base_ranks[channel]) > rank_delta
            )
            fraction = migrated / len(base_top_channels)
            layer_fractions.append(fraction)
            layer_metrics[layer_index].append(fraction)
        trace_metrics.append(
            {
                "prompt_index": prompt_index,
                "migration_fraction": float(mean(layer_fractions)),
                "layer_count": len(layer_fractions),
            }
        )
    trace_values = [float(row["migration_fraction"]) for row in trace_metrics]
    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(bootstrap_samples):
        sample = [trace_values[rng.randrange(len(trace_values))] for _ in trace_values]
        boot.append(float(mean(sample)))
    boot.sort()
    ci_low = boot[int(0.025 * (len(boot) - 1))]
    ci_high = boot[int(0.975 * (len(boot) - 1))]
    return {
        "migration_fraction": float(mean(trace_values)),
        "bootstrap_ci95": {"ci95_low": ci_low, "ci95_high": ci_high},
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "trace_metrics": trace_metrics,
        "layer_metrics": [
            {
                "layer_index": layer_index,
                "migration_fraction_mean": float(mean(values)),
                "trace_count": len(values),
            }
            for layer_index, values in sorted(layer_metrics.items())
        ],
        "top_channels_by_layer": {
            str(layer_index): channels for layer_index, channels in sorted(top_channels_by_layer.items())
        },
        "top_channel_selection": "top 1% channels per layer by mean magnitude at decode position 100 across all 12 traces",
    }


def validate_prompt_manifest(prompt_manifest: dict[str, Any], metrics: dict[str, Any], infra: list[str]) -> set[int]:
    if prompt_manifest.get("source") != "AIME-2025":
        infra.append("prompt_manifest.source must be AIME-2025")
    if prompt_manifest.get("selection") != "deterministic_indices_0_11":
        infra.append("prompt_manifest.selection must be deterministic_indices_0_11")
    if prompt_manifest.get("prompt_file_sha256") != file_sha256(DEFAULT_PROMPT_FILE):
        infra.append("prompt_manifest.prompt_file_sha256 must match the canonical AIME-2025 indices 0-11 file")
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
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            infra.append(f"prompt {index}: missing prompt text")
        if row.get("source_dataset") != EXPECTED_PROMPT_SOURCE_DATASET:
            infra.append(f"prompt {index}: source_dataset must be {EXPECTED_PROMPT_SOURCE_DATASET}")
        if row.get("source_file") != EXPECTED_PROMPT_SOURCE_FILE:
            infra.append(f"prompt {index}: source_file must be {EXPECTED_PROMPT_SOURCE_FILE}")
        if row.get("source_commit") != EXPECTED_PROMPT_SOURCE_COMMIT:
            infra.append(f"prompt {index}: source_commit must be {EXPECTED_PROMPT_SOURCE_COMMIT}")
    if sorted(indices) != list(range(TRACE_COUNT)):
        infra.append("prompt indices must be exactly 0-11")
    if len(prompts) != TRACE_COUNT or int(prompt_manifest.get("prompt_count", -1)) != TRACE_COUNT:
        infra.append("prompt_manifest must contain exactly 12 prompts")
    if prompts:
        expected_sha = prompt_payload_sha256(prompts)
        if prompt_manifest.get("prompt_sha256") != expected_sha:
            infra.append("prompt_manifest.prompt_sha256 mismatch")
    if metrics.get("prompt_source") != "AIME-2025":
        infra.append("metrics.prompt_source must be AIME-2025")
    if metrics.get("prompt_selection") != "deterministic_indices_0_11":
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


def validate_activation_rows(
    rows: list[dict[str, Any]],
    *,
    prompt_indices: set[int],
    activation_manifest: dict[str, Any],
    metrics: dict[str, Any],
    infra: list[str],
) -> None:
    if list(metrics.get("positions", [])) != list(POSITIONS):
        infra.append("metrics.positions must equal the preregistered position grid")
    if list(activation_manifest.get("positions", [])) != list(POSITIONS):
        infra.append("activation manifest positions must equal the preregistered position grid")
    layer_count = int(activation_manifest.get("layer_count", 0) or 0)
    if layer_count <= 0:
        infra.append("activation manifest layer_count must be positive")
    if int(metrics.get("layer_count", 0) or 0) != layer_count:
        infra.append("metrics.layer_count must match activation manifest")
    expected_count = TRACE_COUNT * layer_count * len(POSITIONS)
    if int(activation_manifest.get("trace_count", 0) or 0) != TRACE_COUNT:
        infra.append("activation manifest trace_count must be 12")
    if int(activation_manifest.get("row_count", -1)) != expected_count:
        infra.append("activation manifest row_count does not match 12 * layers * positions")
    if len(rows) != expected_count:
        infra.append(f"activation row count {len(rows)} does not match expected {expected_count}")
    seen: set[tuple[int, int, int]] = set()
    hidden_size: int | None = None
    layer_names = activation_manifest.get("layer_names", [])
    for row in rows:
        try:
            prompt_index = int(row["prompt_index"])
            layer_index = int(row["layer_index"])
            position = int(row["decode_position"])
            channel_count = int(row["channel_count"])
            magnitudes = row["channel_magnitudes"]
        except Exception as exc:
            infra.append(f"activation row {row.get('_line_number')}: missing typed fields: {exc!r}")
            continue
        key = (prompt_index, layer_index, position)
        if key in seen:
            infra.append(f"duplicate activation row for prompt/layer/position {key}")
        seen.add(key)
        if prompt_index not in prompt_indices:
            infra.append(f"activation row prompt_index {prompt_index} not in prompt manifest")
        if layer_index < 0 or layer_index >= layer_count:
            infra.append(f"activation row layer_index {layer_index} out of range")
        if position not in POSITIONS:
            infra.append(f"activation row decode_position {position} outside preregistered grid")
        if isinstance(layer_names, list) and layer_index < len(layer_names):
            if row.get("layer_name") != layer_names[layer_index]:
                infra.append(f"activation row layer_name mismatch for layer {layer_index}")
        if not isinstance(magnitudes, list) or len(magnitudes) != channel_count:
            infra.append(f"activation row {key} channel_magnitudes length mismatch")
            continue
        if hidden_size is None:
            hidden_size = channel_count
        elif hidden_size != channel_count:
            infra.append("activation rows do not have a consistent hidden size")
        if channel_count <= 0:
            infra.append(f"activation row {key} channel_count must be positive")
        for value in magnitudes:
            try:
                numeric = float(value)
            except Exception:
                infra.append(f"activation row {key} contains nonnumeric magnitude")
                break
            if not math.isfinite(numeric) or numeric < 0.0:
                infra.append(f"activation row {key} contains invalid magnitude")
                break
    if hidden_size is not None and metrics.get("hidden_size") != hidden_size:
        infra.append("metrics.hidden_size must match activation rows")


def validate_activation_artifact_reference(run_dir: Path, metrics: dict[str, Any], infra: list[str]) -> None:
    artifact = str(metrics.get("activation_artifact", ""))
    if artifact != "activation_magnitudes.jsonl.gz":
        infra.append("metrics.activation_artifact must be activation_magnitudes.jsonl.gz")
        return
    path = run_dir / artifact
    if metrics.get("activation_artifact_sha256") != file_sha256(path):
        infra.append("metrics.activation_artifact_sha256 mismatch")


def decision_from_metrics(point: float, ci_low: float, ci_high: float) -> tuple[str, list[str]]:
    threshold = float(THRESHOLDS["migration_fraction_threshold"])
    if point >= threshold and ci_low > threshold:
        return PASS_DYNAMIC, [
            f"dynamic pass: point={point:.8f} >= {threshold:.2f} and ci_low={ci_low:.8f} > {threshold:.2f}"
        ]
    if point < threshold and ci_high < threshold:
        return PASS_STATIC, [
            f"static pass: point={point:.8f} < {threshold:.2f} and ci_high={ci_high:.8f} < {threshold:.2f}"
        ]
    return KILL_AMBIGUOUS, [
        f"ambiguous kill: point={point:.8f}, ci95=[{ci_low:.8f}, {ci_high:.8f}], threshold={threshold:.2f}"
    ]


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
        activation_manifest = load_json(run_dir / "activation_magnitude_manifest.json")
        metrics = load_json(run_dir / "metrics.json")
        bootstrap_ci = load_json(run_dir / "bootstrap_ci.json")
        artifact_hashes = load_json(run_dir / "artifact_hashes.json")
    except Exception as exc:
        result = {
            "decision": FAIL_INFRA,
            "run_dir": str(run_dir),
            "reasons": [*infra, f"bad JSON artifacts: {exc!r}"],
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
                    "reasons": result["reasons"],
                },
            )
        return result

    if environment.get("schema_version") != f"{SCHEMA_VERSION}_environment":
        infra.append("environment schema_version mismatch")
    if command.get("branch") != "outlier_migrate":
        infra.append("command_metadata.branch must be outlier_migrate")
    if model.get("model_id") != MODEL_ID:
        infra.append("model_provenance.model_id mismatch")
    if not model.get("hf_snapshot_commit") or not model.get("snapshot_path"):
        infra.append("model provenance must record local HF snapshot commit and path")
    if metrics.get("model_id") != MODEL_ID:
        infra.append("metrics.model_id mismatch")
    if metrics.get("thresholds") != THRESHOLDS:
        infra.append("metrics.thresholds mismatch preregistered checker thresholds")
    if int(metrics.get("bootstrap_samples", 0) or 0) != THRESHOLDS["bootstrap_samples"]:
        infra.append("metrics.bootstrap_samples must be 1000")
    if int(bootstrap_ci.get("bootstrap_samples", 0) or 0) != THRESHOLDS["bootstrap_samples"]:
        infra.append("bootstrap_ci.bootstrap_samples must be 1000")
    if int(random_seed.get("seed", -1)) != int(metrics.get("bootstrap_seed", -2)):
        infra.append("random_seed.seed must match metrics.bootstrap_seed")

    prompt_indices = validate_prompt_manifest(prompt_manifest, metrics, infra)
    validate_artifact_hashes(run_dir, artifact_hashes, infra)
    rows: list[dict[str, Any]] = []
    try:
        rows = list(iter_activation_rows(run_dir / "activation_magnitudes.jsonl.gz"))
    except Exception as exc:
        infra.append(f"cannot read activation_magnitudes.jsonl.gz: {exc!r}")
    if rows:
        validate_activation_rows(
            rows,
            prompt_indices=prompt_indices,
            activation_manifest=activation_manifest,
            metrics=metrics,
            infra=infra,
        )
        validate_activation_artifact_reference(run_dir, metrics, infra)

    computed: dict[str, Any] | None = None
    if rows and not infra:
        computed = compute_metrics(
            rows,
            bootstrap_samples=THRESHOLDS["bootstrap_samples"],
            seed=int(metrics["bootstrap_seed"]),
        )
        if not is_close(float(metrics["migration_fraction"]), computed["migration_fraction"]):
            infra.append("metrics.migration_fraction does not match recomputation")
        for key in ["ci95_low", "ci95_high"]:
            if not is_close(float(metrics["bootstrap_ci95"][key]), float(computed["bootstrap_ci95"][key])):
                infra.append(f"metrics.bootstrap_ci95.{key} does not match recomputation")
            if not is_close(float(bootstrap_ci["bootstrap_ci95"][key]), float(computed["bootstrap_ci95"][key])):
                infra.append(f"bootstrap_ci.bootstrap_ci95.{key} does not match recomputation")
        if not is_close(float(bootstrap_ci["migration_fraction"]), computed["migration_fraction"]):
            infra.append("bootstrap_ci.migration_fraction does not match recomputation")
        if metrics.get("top_channels_by_layer") != computed.get("top_channels_by_layer"):
            infra.append("metrics.top_channels_by_layer does not match recomputation")
        if metrics.get("top_channel_selection") != computed.get("top_channel_selection"):
            infra.append("metrics.top_channel_selection mismatch")

    if infra:
        result = {
            "decision": FAIL_INFRA,
            "run_dir": str(run_dir),
            "reasons": infra,
            "artifact_complete": False,
        }
    else:
        assert computed is not None
        point = float(computed["migration_fraction"])
        ci_low = float(computed["bootstrap_ci95"]["ci95_low"])
        ci_high = float(computed["bootstrap_ci95"]["ci95_high"])
        decision, reasons = decision_from_metrics(point, ci_low, ci_high)
        result = {
            "decision": decision,
            "run_dir": str(run_dir),
            "reasons": reasons,
            "artifact_complete": True,
            "migration_fraction": point,
            "bootstrap_ci95": computed["bootstrap_ci95"],
            "trace_metrics": computed["trace_metrics"],
            "layer_metrics": computed["layer_metrics"],
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
    parser.add_argument("--branch", required=True, choices=["outlier_migrate"])
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve() if args.run_dir else latest_run_dir().resolve()
    result = evaluate(run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"] in {PASS_DYNAMIC, PASS_STATIC} else 1


if __name__ == "__main__":
    raise SystemExit(main())
