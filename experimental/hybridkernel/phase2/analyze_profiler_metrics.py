"""Analyze native HybridKernel profiler summaries against the promote/kill gate.

This script is intentionally small and JSON-based so a future NVIDIA run can
export a compact summary from Nsight/vLLM without ad hoc spreadsheet decisions.
It does not parse raw Nsight traces; the runbook defines how to reduce those
traces into these fields.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean, median


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
            "recoverable_fraction": None,
            "dtype": "bfloat16",
            "cuda_graph_enabled": None,
            "batch_shape": {
                "batch_size": None,
                "prefill_tokens": None,
                "decode_tokens": None,
                "requests": None,
            },
            "control_model_or_segment": None,
            "notes": "Replace nulls after native profiling.",
        }
    ],
}


def _is_pending_row(raw: dict[str, object]) -> bool:
    measured_fields = [
        "total_step_ms",
        "attention_ssm_boundary_ms",
        "matched_non_boundary_ms",
        "recoverable_fraction",
    ]
    return all(raw.get(field) is None for field in measured_fields)


def _require_present(raw: dict[str, object], field: str) -> object:
    value = raw.get(field)
    if value is None:
        raise ValueError(f"{field} must be explicitly recorded for every native row")
    return value


def _config_key(raw: dict[str, object], model: str) -> tuple[str, dict[str, object]]:
    dtype = str(_require_present(raw, "dtype"))
    cuda_graph_enabled = _require_present(raw, "cuda_graph_enabled")
    batch_shape = _require_present(raw, "batch_shape")
    control = str(_require_present(raw, "control_model_or_segment")).strip()
    if not dtype.strip():
        raise ValueError("dtype must be a non-empty string")
    if not isinstance(cuda_graph_enabled, bool):
        raise ValueError("cuda_graph_enabled must be a boolean, not a string or numeric placeholder")
    if not control:
        raise ValueError("control_model_or_segment must be a non-empty string")
    if not isinstance(batch_shape, dict):
        raise ValueError("batch_shape must be an object with batch/request settings")
    for field in ["batch_size", "prefill_tokens", "decode_tokens", "requests"]:
        value = batch_shape.get(field)
        if value is None:
            raise ValueError(f"batch_shape.{field} must be explicitly recorded")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"batch_shape.{field} must be a positive integer")
        if value <= 0:
            raise ValueError(f"batch_shape.{field} must be positive")
    normalized = {
        "model": model,
        "dtype": dtype,
        "cuda_graph_enabled": cuda_graph_enabled,
        "batch_shape": {
            "batch_size": int(batch_shape["batch_size"]),
            "prefill_tokens": int(batch_shape["prefill_tokens"]),
            "decode_tokens": int(batch_shape["decode_tokens"]),
            "requests": int(batch_shape["requests"]),
        },
        "control_model_or_segment": control,
    }
    return json.dumps(normalized, sort_keys=True), normalized


def _iqr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    ordered = sorted(values)
    q1 = ordered[len(ordered) // 4]
    q3 = ordered[(3 * len(ordered)) // 4]
    return q3 - q1


def _bootstrap_ci(values: list[float], *, draws: int = 2000) -> dict[str, float]:
    if not values:
        return {"low": 0.0, "high": 0.0}
    rng = random.Random(1729)
    means = []
    for _ in range(draws):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(mean(sample))
    means.sort()
    return {
        "low": means[int(0.025 * (draws - 1))],
        "high": means[int(0.975 * (draws - 1))],
    }


def _valid_rows(payload: dict[str, object]) -> list[dict[str, float | str]]:
    rows = []
    for raw in payload.get("rows", []):
        if not isinstance(raw, dict):
            raise ValueError("every profiler metric row must be a JSON object")
        if _is_pending_row(raw):
            continue
        total = float(_require_present(raw, "total_step_ms"))
        boundary = float(_require_present(raw, "attention_ssm_boundary_ms"))
        matched = float(_require_present(raw, "matched_non_boundary_ms"))
        recoverable = float(_require_present(raw, "recoverable_fraction"))
        model = str(_require_present(raw, "model")).strip()
        if not model:
            raise ValueError("model must be explicitly recorded and non-empty")
        run_id = str(_require_present(raw, "run_id")).strip()
        if not run_id:
            raise ValueError("run_id must be explicitly recorded and non-empty")
        config_key, config = _config_key(raw, model)
        if total <= 0:
            raise ValueError("total_step_ms must be positive")
        if boundary < 0:
            raise ValueError("attention_ssm_boundary_ms must be non-negative")
        if matched < 0:
            raise ValueError("matched_non_boundary_ms must be non-negative")
        if not 0.0 <= recoverable <= 1.0:
            raise ValueError("recoverable_fraction must be between 0 and 1")
        avoidable = max(0.0, boundary - matched)
        rows.append(
            {
                "model": model,
                "run_id": run_id,
                "config_key": config_key,
                "dtype": str(config["dtype"]),
                "cuda_graph_enabled": str(config["cuda_graph_enabled"]),
                "batch_size": float(config["batch_shape"]["batch_size"]),
                "prefill_tokens": float(config["batch_shape"]["prefill_tokens"]),
                "decode_tokens": float(config["batch_shape"]["decode_tokens"]),
                "requests": float(config["batch_shape"]["requests"]),
                "control_model_or_segment": str(config["control_model_or_segment"]),
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

    by_config: dict[str, list[dict[str, float | str]]] = {}
    for row in rows:
        by_config.setdefault(str(row["config_key"]), []).append(row)
    summary = {}
    for config_key, config_rows in by_config.items():
        gains = [float(row["recoverable_gain_upper_bound"]) for row in config_rows]
        avoidable = [float(row["avoidable_share"]) for row in config_rows]
        ci = _bootstrap_ci(gains)
        first = config_rows[0]
        summary[config_key] = {
            "model": str(first["model"]),
            "config_key": config_key,
            "runs": len(config_rows),
            "distinct_run_ids": len({str(row["run_id"]) for row in config_rows}),
            "dtype": str(first["dtype"]),
            "cuda_graph_enabled": str(first["cuda_graph_enabled"]),
            "batch_size": int(float(first["batch_size"])),
            "prefill_tokens": int(float(first["prefill_tokens"])),
            "decode_tokens": int(float(first["decode_tokens"])),
            "requests": int(float(first["requests"])),
            "control_model_or_segment": str(first["control_model_or_segment"]),
            "mean_avoidable_share": mean(avoidable),
            "mean_recoverable_gain_upper_bound": mean(gains),
            "median_recoverable_gain_upper_bound": median(gains),
            "iqr_recoverable_gain_upper_bound": _iqr(gains),
            "bootstrap_ci95_recoverable_gain_upper_bound": ci,
            "min_recoverable_gain_upper_bound": min(gains),
            "clears_3pct_gate_all_runs": len(config_rows) >= 3 and min(gains) >= 0.03,
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
        "| Model/config | Runs | Batch | Mean avoidable share | Mean gain UB | Median gain UB | IQR | Bootstrap 95% CI | Min gain UB | Clears? |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for config_key, row in result.get("summary", {}).items():
        lines.append(
            "| {model} / {dtype} / graph={graph} / control={control} | {runs} | b{batch} p{prefill} d{decode} r{requests} | {mean_avoidable_share:.2%} | {mean_recoverable_gain_upper_bound:.2%} | {median_recoverable_gain_upper_bound:.2%} | {iqr_recoverable_gain_upper_bound:.2%} | [{ci_low:.2%}, {ci_high:.2%}] | {min_recoverable_gain_upper_bound:.2%} | {clears} |".format(
                model=row["model"],
                dtype=row["dtype"],
                graph=row["cuda_graph_enabled"],
                control=row["control_model_or_segment"],
                batch=row["batch_size"],
                prefill=row["prefill_tokens"],
                decode=row["decode_tokens"],
                requests=row["requests"],
                runs=row["runs"],
                mean_avoidable_share=row["mean_avoidable_share"],
                mean_recoverable_gain_upper_bound=row["mean_recoverable_gain_upper_bound"],
                median_recoverable_gain_upper_bound=row["median_recoverable_gain_upper_bound"],
                iqr_recoverable_gain_upper_bound=row["iqr_recoverable_gain_upper_bound"],
                ci_low=row["bootstrap_ci95_recoverable_gain_upper_bound"]["low"],
                ci_high=row["bootstrap_ci95_recoverable_gain_upper_bound"]["high"],
                min_recoverable_gain_upper_bound=row["min_recoverable_gain_upper_bound"],
                clears="yes" if row["clears_3pct_gate_all_runs"] else "no",
            )
        )
    if not result.get("summary"):
        lines.append("| pending | 0 | -- | -- | -- | -- | -- | -- | -- | no |")
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
