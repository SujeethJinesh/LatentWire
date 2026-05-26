#!/usr/bin/env python3
"""Check Stage 1 E2 cross-prompt set-leaving packets."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase9/results"

SCHEMA_VERSION = "om_stage1_e2_cross_prompt_replication_v1"
PASS = "PASS_E2"
AMBIGUOUS = "AMBIGUOUS_E2"
KILL = "KILL_E2"
INCOMPLETE = "INCOMPLETE_E2_CAP_EXHAUSTED"
FAIL_INFRA = "FAIL_INFRA_E2"

POSITIONS = (100, 20000)
TOP_FRACTION = 0.01
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20260526
REPLICATION_TOLERANCE = 0.15
MODEL_REFERENCES = {
    "granite_small": {
        "model_id": "ibm-granite/granite-4.0-h-small",
        "aime_strict_set_leaving": 0.566234756098,
    },
    "deepseek_r1_distill": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "aime_strict_set_leaving": 0.670572916667,
    },
    "falcon_h1": {
        "model_id": "tiiuae/Falcon-H1-0.5B-Instruct",
        "aime_strict_set_leaving": 0.673611111111,
    },
}
EXPECTED_MODEL_KEYS = tuple(MODEL_REFERENCES)
REQUIRED_FILES = [
    "environment.json",
    "prompt_manifest.json",
    "command_metadata.json",
    "random_seed.json",
    "metrics.json",
    "bootstrap_ci.json",
    "artifact_hashes.json",
    "logs/stdout.log",
    "logs/stderr.log",
    "run_events.jsonl",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def latest_run_dir() -> Path:
    candidates = [path for path in RESULTS_DIR.glob("om_stage1_e2_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no E2 result dirs found under {RESULTS_DIR}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def iter_rows(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def ranks_desc(values: list[float]) -> list[int]:
    ordered = sorted(range(len(values)), key=lambda index: (-float(values[index]), index))
    ranks = [0] * len(values)
    for rank, channel in enumerate(ordered):
        ranks[channel] = rank
    return ranks


def bootstrap_ci(values: list[float], *, seed: int = BOOTSTRAP_SEED) -> dict[str, float]:
    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [values[rng.randrange(len(values))] for _ in values]
        boot.append(float(mean(sample)))
    boot.sort()
    return {
        "ci95_low": boot[int(0.025 * (len(boot) - 1))],
        "ci95_high": boot[int(0.975 * (len(boot) - 1))],
    }


def compute_model_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_trace_layer: dict[int, dict[int, dict[int, list[float]]]] = defaultdict(lambda: defaultdict(dict))
    prompt_dataset: dict[int, str] = {}
    prompt_id: dict[int, str] = {}
    for row in rows:
        prompt_index = int(row["prompt_index"])
        by_trace_layer[prompt_index][int(row["layer_index"])][int(row["decode_position"])] = [
            float(value) for value in row["channel_magnitudes"]
        ]
        prompt_dataset[prompt_index] = str(row.get("source_dataset", "unknown"))
        prompt_id[prompt_index] = str(row.get("prompt_id", prompt_index))

    trace_metrics: list[dict[str, Any]] = []
    for prompt_index in sorted(by_trace_layer):
        layer_values: list[float] = []
        for layer_index in sorted(by_trace_layer[prompt_index]):
            base = by_trace_layer[prompt_index][layer_index][POSITIONS[0]]
            final = by_trace_layer[prompt_index][layer_index][POSITIONS[-1]]
            top_k = max(1, math.ceil(len(base) * TOP_FRACTION))
            base_ranks = ranks_desc(base)
            final_ranks = ranks_desc(final)
            selected = [channel for channel, rank in enumerate(base_ranks) if rank < top_k]
            left = sum(1 for channel in selected if final_ranks[channel] >= top_k)
            layer_values.append(left / len(selected))
        trace_metrics.append(
            {
                "prompt_index": prompt_index,
                "prompt_id": prompt_id[prompt_index],
                "source_dataset": prompt_dataset[prompt_index],
                "left_set_fraction": float(mean(layer_values)),
                "layer_count": len(layer_values),
            }
        )

    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        values = [float(row["left_set_fraction"]) for row in selected]
        return {"left_set_fraction": float(mean(values)), "ci95": bootstrap_ci(values), "trace_count": len(values)}

    by_dataset = {
        dataset: summarize([row for row in trace_metrics if row["source_dataset"] == dataset])
        for dataset in sorted({row["source_dataset"] for row in trace_metrics})
    }
    return {
        "trace_metrics": trace_metrics,
        "aggregate": summarize(trace_metrics),
        "by_dataset": by_dataset,
    }


def classify(metrics: dict[str, Any]) -> tuple[str, list[str]]:
    completed = metrics.get("completed_model_keys", [])
    if set(completed) != set(EXPECTED_MODEL_KEYS):
        return INCOMPLETE, [f"completed models {completed}; expected {list(EXPECTED_MODEL_KEYS)}"]
    partial = [
        key
        for key in EXPECTED_MODEL_KEYS
        if metrics["model_results"][key].get("incomplete_due_cap")
    ]
    if partial:
        return INCOMPLETE, [f"cap exhausted before all prompts for models: {partial}"]
    replicated: list[str] = []
    failed: list[str] = []
    for key in EXPECTED_MODEL_KEYS:
        row = metrics["model_results"][key]
        if abs(float(row["delta_vs_aime"])) <= REPLICATION_TOLERANCE:
            replicated.append(key)
        else:
            failed.append(key)
    if len(replicated) == len(EXPECTED_MODEL_KEYS):
        return PASS, [f"all models within +/-{REPLICATION_TOLERANCE} of AIME references"]
    if replicated:
        return AMBIGUOUS, [f"replicated={replicated}; outside tolerance={failed}"]
    return KILL, [f"no measured model stayed within +/-{REPLICATION_TOLERANCE} of AIME references"]


def evaluate(run_dir: Path) -> dict[str, Any]:
    infra = [f"missing required artifact: {rel}" for rel in REQUIRED_FILES if not (run_dir / rel).is_file()]
    metrics: dict[str, Any] = {}
    if not infra:
        metrics = load_json(run_dir / "metrics.json")
        if metrics.get("schema_version") != f"{SCHEMA_VERSION}_metrics":
            infra.append("metrics schema_version mismatch")
        if tuple(metrics.get("positions", [])) != POSITIONS:
            infra.append("metrics positions mismatch")
        for key in metrics.get("completed_model_keys", []):
            path = run_dir / "model_runs" / key / "activation_magnitudes.jsonl.gz"
            if not path.is_file():
                infra.append(f"missing activation rows for {key}")
                continue
            computed = compute_model_metrics(list(iter_rows(path)))
            observed = metrics["model_results"][key]["aggregate"]["left_set_fraction"]
            if abs(float(observed) - float(computed["aggregate"]["left_set_fraction"])) > 1e-9:
                infra.append(f"{key}: aggregate strict set-leaving does not recompute")
    if infra:
        decision, reasons = FAIL_INFRA, infra
    else:
        decision, reasons = classify(metrics)
    result = {
        "schema_version": f"{SCHEMA_VERSION}_checker_result",
        "decision": decision,
        "artifact_complete": decision not in {FAIL_INFRA, INCOMPLETE},
        "reasons": reasons,
        "run_dir": str(run_dir),
        "headline": metrics.get("headline") if metrics else None,
    }
    write_json(run_dir / "checker_result.json", result)
    write_json(run_dir / "artifact_check.json", {k: result[k] for k in ["schema_version", "decision", "artifact_complete", "reasons", "run_dir"]})
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    result = evaluate(args.run_dir or latest_run_dir())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["decision"] == FAIL_INFRA else 0


if __name__ == "__main__":
    raise SystemExit(main())
