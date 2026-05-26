#!/usr/bin/env python3
"""Check Stage 1 E3 ParoQuant plus M11b composition packets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "experimental/outlier_migrate/phase9/results"
PREREG_PATH = ROOT / "experimental/outlier_migrate/phase9/preregister_om_stage1_e3_paroquant_m11b_composition.md"

SCHEMA_VERSION = "om_stage1_e3_paroquant_m11b_composition_v1"
PASS_SUPER = "PASS_E3_SUPER_ADDITIVE"
PASS_COMPLEMENTARY = "PASS_E3_COMPLEMENTARY"
KILL_SUB_ADDITIVE = "KILL_E3_SUB_ADDITIVE"
AMBIGUOUS = "AMBIGUOUS_E3"
FAIL_INFRA = "FAIL_INFRA_E3"

REQUIRED_FILES = [
    "environment.json",
    "environment.txt",
    "model_provenance.json",
    "prompt_manifest.json",
    "command_metadata.json",
    "random_seed.json",
    "decoding_config.json",
    "quantization_config.json",
    "source_artifacts.json",
    "protected_sets.json",
    "excluded_tensors.json",
    "per_trace_metrics.json",
    "metrics.json",
    "bootstrap_ci.json",
    "control_metrics.json",
    "artifact_hashes.json",
    "logs/stdout.log",
    "logs/stderr.log",
    "run_events.jsonl",
]

REGIMES = [
    "bf16",
    "static_1pct",
    "paroquant_w4a16",
    "m11b_top10",
    "paroquant_m11b_top10",
    "paroquant_random_top10",
]
RECOVERY_REGIMES = ["paroquant_w4a16", "m11b_top10", "paroquant_m11b_top10", "paroquant_random_top10"]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def latest_run_dir() -> Path:
    candidates = [path for path in RESULTS_DIR.iterdir() if path.is_dir()] if RESULTS_DIR.is_dir() else []
    candidates = [path for path in candidates if path.name.startswith("om_stage1_e3_")]
    if not candidates:
        raise FileNotFoundError(f"no E3 result dirs found under {RESULTS_DIR}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def classify(metrics: dict[str, Any]) -> tuple[str, list[str]]:
    results = metrics["results_by_regime"]
    comp = results["paroquant_m11b_top10"]["median_recovery"]
    comp_low = results["paroquant_m11b_top10"]["bootstrap_ci95"]["ci95_low"]
    paro = results["paroquant_w4a16"]["median_recovery"]
    m11b = results["m11b_top10"]["median_recovery"]
    best = max(paro, m11b)
    weaker = min(paro, m11b)
    if comp - best >= 0.10 and comp_low > 0.10:
        return PASS_SUPER, [f"composition beats best individual by {comp - best:.6f} with CI low {comp_low:.6f}"]
    if comp >= best - 0.05 and comp - weaker >= 0.15:
        return PASS_COMPLEMENTARY, [f"composition is within 0.05 of best and beats weaker individual by {comp - weaker:.6f}"]
    if comp <= best - 0.05:
        return KILL_SUB_ADDITIVE, [f"composition trails best individual by {best - comp:.6f}"]
    return AMBIGUOUS, ["composition CI/effect overlaps individual methods"]


def evaluate(run_dir: Path) -> dict[str, Any]:
    infra: list[str] = []
    missing = [rel for rel in REQUIRED_FILES if not (run_dir / rel).is_file()]
    if missing:
        infra.append(f"missing required files: {missing}")
    metrics: dict[str, Any] = {}
    if not infra:
        metrics = load_json(run_dir / "metrics.json")
        if set(metrics.get("results_by_regime", {})) != set(RECOVERY_REGIMES):
            infra.append("metrics.results_by_regime mismatch")
        rows = load_json(run_dir / "per_trace_metrics.json").get("traces", [])
        if len(rows) != int(metrics.get("trace_count", -1)):
            infra.append("per_trace_metrics trace count mismatch")
        for row in rows:
            if set(row.get("perplexities", {})) != set(REGIMES):
                infra.append(f"trace {row.get('prompt_index')}: perplexity regimes mismatch")
                break
            if set(row.get("recoveries", {})) != set(RECOVERY_REGIMES):
                infra.append(f"trace {row.get('prompt_index')}: recovery regimes mismatch")
                break

    if infra:
        decision = FAIL_INFRA
        reasons = infra
    else:
        decision, reasons = classify(metrics)
    result = {
        "schema_version": f"{SCHEMA_VERSION}_checker_result",
        "decision": decision,
        "artifact_complete": not infra,
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
