"""Analyze native HybridKernel profiler summaries against the promote/kill gate.

This script is intentionally small and JSON-based so a future NVIDIA run can
export a compact summary from Nsight/vLLM without ad hoc spreadsheet decisions.
It does not parse raw Nsight traces; the runbook defines how to reduce those
traces into these fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "experimental/hybridkernel/phase2"
DEFAULT_INPUT = OUT_DIR / "profiler_metrics_template.json"
DEFAULT_OUTPUT = OUT_DIR / "profiler_analysis_gate.json"


TEMPLATE = {
    "description": "Fill with repeated native NVIDIA/vLLM measurements reduced from Nsight traces.",
    "rows": [
        {
            "model": "ibm-granite-4.0-h-tiny",
            "run_id": 0,
            "total_step_ms": None,
            "attention_ssm_boundary_ms": None,
            "matched_non_boundary_ms": None,
            "recoverable_fraction": 0.60,
            "notes": "Replace nulls after native profiling.",
        }
    ],
}


def _valid_rows(payload: dict[str, object]) -> list[dict[str, float | str]]:
    rows = []
    for raw in payload.get("rows", []):
        if raw.get("total_step_ms") is None or raw.get("attention_ssm_boundary_ms") is None:
            continue
        total = float(raw["total_step_ms"])
        boundary = float(raw["attention_ssm_boundary_ms"])
        matched = float(raw.get("matched_non_boundary_ms") or 0.0)
        recoverable = float(raw.get("recoverable_fraction", 0.60))
        if total <= 0:
            raise ValueError("total_step_ms must be positive")
        avoidable = max(0.0, boundary - matched)
        rows.append(
            {
                "model": str(raw.get("model", "unknown")),
                "run_id": str(raw.get("run_id", len(rows))),
                "total_step_ms": total,
                "boundary_ms": boundary,
                "matched_non_boundary_ms": matched,
                "avoidable_boundary_ms": avoidable,
                "recoverable_fraction": recoverable,
                "boundary_share": boundary / total,
                "avoidable_share": avoidable / total,
                "recoverable_gain_upper_bound": avoidable * recoverable / total,
            }
        )
    return rows


def analyze(payload: dict[str, object]) -> dict[str, object]:
    rows = _valid_rows(payload)
    if not rows:
        return {
            "status": "PENDING native profiler data.",
            "rows": [],
            "summary": {},
            "decision": "No native speed or overhead claim is allowed.",
        }

    by_model: dict[str, list[dict[str, float | str]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)
    summary = {}
    for model, model_rows in by_model.items():
        gains = [float(row["recoverable_gain_upper_bound"]) for row in model_rows]
        avoidable = [float(row["avoidable_share"]) for row in model_rows]
        summary[model] = {
            "runs": len(model_rows),
            "mean_avoidable_share": mean(avoidable),
            "mean_recoverable_gain_upper_bound": mean(gains),
            "min_recoverable_gain_upper_bound": min(gains),
            "clears_3pct_gate_all_runs": len(model_rows) >= 3 and min(gains) >= 0.03,
        }
    if any(row["clears_3pct_gate_all_runs"] for row in summary.values()):
        status = "PROMOTE to prototype: repeated profiler summaries clear the 3% upper-bound gate."
        decision = "Build the smallest boundary-fusion prototype for the clearing model only."
    elif max(row["mean_recoverable_gain_upper_bound"] for row in summary.values()) < 0.01:
        status = "KILL or shelve: native profiler summaries show less than 1% recoverable gain."
        decision = "Do not spend kernel implementation time without a new profiler anomaly."
    else:
        status = "WEAKLY ALIVE: profiler evidence is nonzero but below the prototype gate."
        decision = "Collect more repeated traces or narrow to the largest boundary anomaly."
    return {"status": status, "rows": rows, "summary": summary, "decision": decision}


def _write_markdown(result: dict[str, object], path: Path) -> None:
    lines = [
        "# HybridKernel Profiler Analysis Gate",
        "",
        f"Status: **{result['status']}**",
        "",
        result["decision"],
        "",
        "## Model Summary",
        "",
        "| Model | Runs | Mean avoidable share | Mean recoverable gain UB | Min gain UB | Clears all-run 3% gate? |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for model, row in result.get("summary", {}).items():
        lines.append(
            "| {model} | {runs} | {mean_avoidable_share:.2%} | {mean_recoverable_gain_upper_bound:.2%} | {min_recoverable_gain_upper_bound:.2%} | {clears} |".format(
                model=model,
                clears="yes" if row["clears_3pct_gate_all_runs"] else "no",
                **row,
            )
        )
    if not result.get("summary"):
        lines.append("| pending | 0 | -- | -- | -- | no |")
    lines.extend(
        [
            "",
            "Definitions: `attention_ssm_boundary_ms` is the measured boundary-local cost from the native profiler pass. `matched_non_boundary_ms` is the matched local control cost. The avoidable share is `max(boundary - control, 0) / total_step_ms`; the recoverable-gain upper bound additionally multiplies by the assumed recoverable fraction.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write-template", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.write_template or not args.input.exists():
        args.input.write_text(json.dumps(TEMPLATE, indent=2) + "\n", encoding="utf-8")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = analyze(payload)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_markdown(result, args.output.with_suffix(".md"))
    print(json.dumps({"status": result["status"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
