#!/usr/bin/env python3
"""Check OutlierMigrate Phase 5' pure-Transformer control packets."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase5_prime/results"

from experimental.outlier_migrate.phase1 import check_om_phase1 as phase1


SCHEMA_VERSION = "om_phase5_prime_transformer_v1"
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_SNAPSHOT_COMMIT = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
FALLBACK_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
FALLBACK_MODEL_SNAPSHOT_COMMIT = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
POSITIONS = phase1.POSITIONS
TRACE_COUNT = phase1.TRACE_COUNT
THRESHOLDS = {
    **phase1.THRESHOLDS,
    "bootstrap_seed": 20260511,
}

DYNAMIC = "DYNAMIC_REGIME_TRANSFORMER"
STATIC = "STATIC_REGIME_TRANSFORMER"
AMBIGUOUS = "AMBIGUOUS_TRANSFORMER"
FAIL_INFRA = "FAIL_INFRA_TRANSFORMER"

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
    "migration_decomposition.json",
    "migration_decomposition.md",
    "artifact_hashes.json",
    "logs/stdout.log",
    "logs/stderr.log",
    "run_events.jsonl",
]
HASHED_FILES = [rel for rel in REQUIRED_FILES if rel != "artifact_hashes.json"]

load_json = phase1.load_json
write_json = phase1.write_json
file_sha256 = phase1.file_sha256
iter_activation_rows = phase1.iter_activation_rows
is_close = phase1.is_close


def latest_run_dir() -> Path:
    candidates = [path for path in RESULTS_DIR.iterdir() if path.is_dir()] if RESULTS_DIR.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"no OutlierMigrate Phase 5' result dirs found under {RESULTS_DIR}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def ranks_desc(values: list[float]) -> list[int]:
    return phase1.ranks_desc(values)


def bootstrap_ci(values: list[float], *, samples: int, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(samples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        boot.append(float(mean(sample)))
    boot.sort()
    return {
        "ci95_low": boot[int(0.025 * (len(boot) - 1))],
        "ci95_high": boot[int(0.975 * (len(boot) - 1))],
    }


def compute_decomposition(rows: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    by_trace_layer: dict[int, dict[int, dict[int, list[float]]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        by_trace_layer[int(row["prompt_index"])][int(row["layer_index"])][int(row["decode_position"])] = [
            float(value) for value in row["channel_magnitudes"]
        ]
    top_fraction = float(THRESHOLDS["top_channel_fraction"])
    rank_delta = int(THRESHOLDS["rank_delta_strictly_greater_than"])
    base_position = POSITIONS[0]
    final_position = POSITIONS[-1]
    trace_metrics: list[dict[str, Any]] = []
    layer_metrics: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for prompt_index in sorted(by_trace_layer):
        trace_left: list[float] = []
        trace_within: list[float] = []
        trace_original: list[float] = []
        for layer_index in sorted(by_trace_layer[prompt_index]):
            base = by_trace_layer[prompt_index][layer_index][base_position]
            final = by_trace_layer[prompt_index][layer_index][final_position]
            top_k = max(1, math.ceil(len(base) * top_fraction))
            top_boundary = top_k - 1
            base_ranks = ranks_desc(base)
            final_ranks = ranks_desc(final)
            selected = [channel for channel, rank in enumerate(base_ranks) if rank <= top_boundary]
            left = 0
            within = 0
            original = 0
            for channel in selected:
                delta = abs(final_ranks[channel] - base_ranks[channel])
                if delta > rank_delta:
                    original += 1
                if final_ranks[channel] > top_boundary:
                    left += 1
                elif delta > rank_delta:
                    within += 1
            denom = len(selected)
            left_fraction = left / denom
            within_fraction = within / denom
            original_fraction = original / denom
            trace_left.append(left_fraction)
            trace_within.append(within_fraction)
            trace_original.append(original_fraction)
            layer_metrics[layer_index]["left_set_fraction"].append(left_fraction)
            layer_metrics[layer_index]["within_set_rank_shuffle_fraction"].append(within_fraction)
            layer_metrics[layer_index]["strict_original_fraction"].append(original_fraction)
        trace_metrics.append(
            {
                "prompt_index": prompt_index,
                "left_set_fraction": float(mean(trace_left)),
                "within_set_rank_shuffle_fraction": float(mean(trace_within)),
                "strict_original_fraction": float(mean(trace_original)),
                "layer_count": len(trace_left),
            }
        )
    left_values = [float(row["left_set_fraction"]) for row in trace_metrics]
    within_values = [float(row["within_set_rank_shuffle_fraction"]) for row in trace_metrics]
    strict_original_values = [float(row["strict_original_fraction"]) for row in trace_metrics]
    samples = int(THRESHOLDS["bootstrap_samples"])
    return {
        "schema_version": f"{SCHEMA_VERSION}_decomposition",
        "base_position": base_position,
        "final_position": final_position,
        "top_channel_fraction": top_fraction,
        "rank_delta_strictly_greater_than": rank_delta,
        "strict_set_leaving_definition": (
            "channel rank <= top_1_percent_boundary at position 100 and "
            "rank > top_1_percent_boundary at final position"
        ),
        "within_set_rank_shuffling_definition": (
            "channel remains inside final top-1% set but moves by more than 2 rank positions"
        ),
        "aggregate": {
            "left_set_fraction": float(mean(left_values)),
            "left_set_ci95": bootstrap_ci(left_values, samples=samples, seed=seed),
            "within_set_rank_shuffle_fraction": float(mean(within_values)),
            "within_set_rank_shuffle_ci95": bootstrap_ci(within_values, samples=samples, seed=seed),
            "strict_original_fraction": float(mean(strict_original_values)),
            "strict_original_ci95": bootstrap_ci(strict_original_values, samples=samples, seed=seed),
        },
        "trace_metrics": trace_metrics,
        "layer_metrics": [
            {
                "layer_index": layer_index,
                "left_set_fraction": float(mean(values["left_set_fraction"])),
                "within_set_rank_shuffle_fraction": float(mean(values["within_set_rank_shuffle_fraction"])),
                "strict_original_fraction": float(mean(values["strict_original_fraction"])),
                "trace_count": len(values["left_set_fraction"]),
            }
            for layer_index, values in sorted(layer_metrics.items())
        ],
    }


def compute_metrics(rows: list[dict[str, Any]], *, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    metrics = phase1.compute_metrics(rows, bootstrap_samples=bootstrap_samples, seed=seed)
    metrics["migration_decomposition"] = compute_decomposition(rows, seed=seed)
    return metrics


def write_decomposition_report(path: Path, payload: dict[str, Any]) -> None:
    agg = payload["aggregate"]
    lines = [
        "# OutlierMigrate Phase 5' Migration Decomposition",
        "",
        "| Component | Fraction | 95% bootstrap CI |",
        "| --- | ---: | ---: |",
        (
            f"| Strict set-leaving | {agg['left_set_fraction']:.12f} | "
            f"[{agg['left_set_ci95']['ci95_low']:.12f}, {agg['left_set_ci95']['ci95_high']:.12f}] |"
        ),
        (
            f"| Within-set rank shuffling | {agg['within_set_rank_shuffle_fraction']:.12f} | "
            f"[{agg['within_set_rank_shuffle_ci95']['ci95_low']:.12f}, "
            f"{agg['within_set_rank_shuffle_ci95']['ci95_high']:.12f}] |"
        ),
        (
            f"| Strict original migration | {agg['strict_original_fraction']:.12f} | "
            f"[{agg['strict_original_ci95']['ci95_low']:.12f}, "
            f"{agg['strict_original_ci95']['ci95_high']:.12f}] |"
        ),
        "",
        "The gate decision still uses the preregistered Phase 0/1/2 mean-top-channel migration metric in `metrics.json`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


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


def decision_from_metrics(point: float, ci_low: float, ci_high: float) -> tuple[str, list[str]]:
    threshold = float(THRESHOLDS["migration_fraction_threshold"])
    if point > threshold and ci_low > threshold:
        return DYNAMIC, [
            f"dynamic transformer regime: point={point:.8f} > {threshold:.2f} "
            f"and ci_low={ci_low:.8f} > {threshold:.2f}"
        ]
    if point < threshold and ci_high < threshold:
        return STATIC, [
            f"static transformer regime: point={point:.8f} < {threshold:.2f} "
            f"and ci_high={ci_high:.8f} < {threshold:.2f}"
        ]
    return AMBIGUOUS, [
        f"ambiguous transformer regime: point={point:.8f}, ci=[{ci_low:.8f}, {ci_high:.8f}] "
        f"overlaps the {threshold:.2f} threshold"
    ]


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
        activation_manifest = load_json(run_dir / "activation_magnitude_manifest.json")
        metrics = load_json(run_dir / "metrics.json")
        bootstrap_ci = load_json(run_dir / "bootstrap_ci.json")
        decomposition = load_json(run_dir / "migration_decomposition.json")
        artifact_hashes = load_json(run_dir / "artifact_hashes.json")
    except Exception as exc:
        return infra_result(run_dir, [*infra, f"bad JSON artifacts: {exc!r}"])

    if environment.get("schema_version") != f"{SCHEMA_VERSION}_environment":
        infra.append("environment schema_version mismatch")
    if command.get("branch") != "outlier_migrate_phase5_prime_transformer_control":
        infra.append("command_metadata.branch mismatch")
    if model.get("model_id") not in {MODEL_ID, FALLBACK_MODEL_ID}:
        infra.append("model_provenance.model_id must be primary or preregistered fallback")
    if model.get("model_id") == MODEL_ID and model.get("hf_snapshot_commit") != MODEL_SNAPSHOT_COMMIT:
        infra.append("primary model snapshot commit mismatch")
    if model.get("model_id") == FALLBACK_MODEL_ID and model.get("hf_snapshot_commit") != FALLBACK_MODEL_SNAPSHOT_COMMIT:
        infra.append("fallback model snapshot commit mismatch")
    if not model.get("snapshot_path"):
        infra.append("model provenance must record a local snapshot path")
    if metrics.get("model_id") != model.get("model_id"):
        infra.append("metrics.model_id must match model provenance")
    if metrics.get("model_family") != "pure_transformer_rope":
        infra.append("metrics.model_family must be pure_transformer_rope")
    if metrics.get("thresholds") != THRESHOLDS:
        infra.append("metrics.thresholds mismatch")
    if int(metrics.get("bootstrap_samples", 0) or 0) != THRESHOLDS["bootstrap_samples"]:
        infra.append("metrics.bootstrap_samples must be 1000")
    if int(metrics.get("bootstrap_seed", -1)) != THRESHOLDS["bootstrap_seed"]:
        infra.append("metrics.bootstrap_seed must be 20260511")
    if int(bootstrap_ci.get("bootstrap_samples", 0) or 0) != THRESHOLDS["bootstrap_samples"]:
        infra.append("bootstrap_ci.bootstrap_samples must be 1000")
    if int(random_seed.get("seed", -1)) != int(metrics.get("bootstrap_seed", -2)):
        infra.append("random_seed.seed must match metrics.bootstrap_seed")
    if decomposition.get("schema_version") != f"{SCHEMA_VERSION}_decomposition":
        infra.append("migration_decomposition schema_version mismatch")

    prompt_indices = phase1.validate_prompt_manifest(prompt_manifest, metrics, infra)
    validate_artifact_hashes(run_dir, artifact_hashes, infra)
    rows: list[dict[str, Any]] = []
    try:
        rows = list(iter_activation_rows(run_dir / "activation_magnitudes.jsonl.gz"))
    except Exception as exc:
        infra.append(f"cannot read activation_magnitudes.jsonl.gz: {exc!r}")
    if rows:
        prompt_ids_by_index = {
            int(row["index"]): str(row["prompt_id"])
            for row in prompt_manifest.get("prompts", [])
            if isinstance(row, dict) and "index" in row and "prompt_id" in row
        }
        phase1.validate_activation_rows(
            rows,
            prompt_indices=prompt_indices,
            prompt_ids_by_index=prompt_ids_by_index,
            activation_manifest=activation_manifest,
            metrics=metrics,
            infra=infra,
        )
        phase1.validate_activation_artifact_reference(run_dir, metrics, infra)

    computed: dict[str, Any] | None = None
    if rows and not infra:
        computed = compute_metrics(
            rows,
            bootstrap_samples=THRESHOLDS["bootstrap_samples"],
            seed=THRESHOLDS["bootstrap_seed"],
        )
        if not is_close(float(metrics["migration_fraction"]), computed["migration_fraction"]):
            infra.append("metrics.migration_fraction does not match recomputation")
        for key in ["ci95_low", "ci95_high"]:
            if not is_close(float(metrics["bootstrap_ci95"][key]), float(computed["bootstrap_ci95"][key])):
                infra.append(f"metrics.bootstrap_ci95.{key} does not match recomputation")
            if not is_close(float(bootstrap_ci["bootstrap_ci95"][key]), float(computed["bootstrap_ci95"][key])):
                infra.append(f"bootstrap_ci.bootstrap_ci95.{key} does not match recomputation")
        if metrics.get("top_channels_by_layer") != computed.get("top_channels_by_layer"):
            infra.append("metrics.top_channels_by_layer does not match recomputation")
        if decomposition != computed["migration_decomposition"]:
            infra.append("migration_decomposition.json does not match recomputation")

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
            "migration_decomposition": computed["migration_decomposition"]["aggregate"],
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
    return 1 if result["decision"] == FAIL_INFRA else 0


if __name__ == "__main__":
    raise SystemExit(main())
