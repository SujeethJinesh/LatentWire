"""Check native HybridKernel profiler run artifacts for review completeness.

This verifier is intentionally about admissibility, not success. A passing run
directory has enough metadata, native profiler artifacts, and repeated reduced
metrics for a reviewer to inspect the promote/kill decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experimental.hybridkernel.phase2.analyze_profiler_metrics import analyze


REQUIRED_FILES = [
    "metadata/environment.txt",
    "metadata/architecture_map.json",
    "metadata/profile_scope.json",
    "profiler_metrics.json",
    "readout.md",
]

ENVIRONMENT_MARKERS = ["nvidia-smi", "nsys", "ncu", "python"]

READOUT_MARKERS = [
    "Distinct boundary conversion/materialization kernel?",
    "Boundary idle or launch gap?",
    "Extra DRAM/L2 traffic near boundary?",
    "End-to-end impact estimate clears 3%?",
    "Same-family controls available?",
    "Cross-family falsification attempted?",
]

NSYS_PATTERNS = ["*.nsys-rep", "*.sqlite", "*.qdrep"]
NCU_PATTERNS = ["*.ncu-rep"]

ALLOWED_PROFILED_PROCESSES = {"vllm_server", "single_process_vllm_benchmark"}


def _has_any(root: Path, patterns: list[str]) -> bool:
    return any(any(root.glob(pattern)) for pattern in patterns)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def check_run_artifacts(
    run_dir: Path,
    min_repeated_runs: int = 3,
    require_native_artifacts: bool = True,
) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    run_dir = run_dir.resolve()

    if not run_dir.exists():
        errors.append(f"run directory does not exist: {run_dir}")
        return {"status": "FAIL", "run_dir": str(run_dir), "errors": errors, "warnings": warnings}
    if not run_dir.is_dir():
        errors.append(f"run path is not a directory: {run_dir}")

    for relative in REQUIRED_FILES:
        if not (run_dir / relative).is_file():
            errors.append(f"missing required artifact: {relative}")

    logs_dir = run_dir / "logs"
    if not logs_dir.is_dir() or not _has_any(logs_dir, ["*.log", "*.txt"]):
        errors.append("missing profiling logs under logs/*.log or logs/*.txt")

    if require_native_artifacts:
        if not _has_any(run_dir / "nsys", NSYS_PATTERNS):
            errors.append("missing Nsight Systems artifact under nsys/")
        if not _has_any(run_dir / "ncu", NCU_PATTERNS):
            errors.append("missing Nsight Compute artifact under ncu/")

    environment_path = run_dir / "metadata/environment.txt"
    if environment_path.is_file():
        environment = _read_text(environment_path).lower()
        for marker in ENVIRONMENT_MARKERS:
            if marker not in environment:
                warnings.append(f"environment metadata does not mention {marker}")

    readout_path = run_dir / "readout.md"
    if readout_path.is_file():
        readout = _read_text(readout_path)
        for marker in READOUT_MARKERS:
            if marker not in readout:
                errors.append(f"readout.md missing decision row: {marker}")

    profile_scope_path = run_dir / "metadata/profile_scope.json"
    if profile_scope_path.is_file():
        try:
            profile_scope = json.loads(profile_scope_path.read_text(encoding="utf-8"))
            profiled_process = str(profile_scope.get("profiled_process", ""))
            trace_scope = str(profile_scope.get("trace_scope", "")).lower()
            vllm_command = str(profile_scope.get("vllm_command", ""))
            if profiled_process not in ALLOWED_PROFILED_PROCESSES:
                errors.append(
                    "profile_scope.json must identify the profiled CUDA-serving process, "
                    "not just an HTTP client"
                )
            if "server" not in trace_scope and "single_process" not in trace_scope:
                errors.append("profile_scope.json trace_scope must cover server-side CUDA work")
            if "vllm" not in vllm_command.lower():
                warnings.append("profile_scope.json vllm_command does not mention vLLM")
        except (json.JSONDecodeError, TypeError) as exc:
            errors.append(f"profile_scope.json is invalid: {exc}")

    metrics_status = "unavailable"
    metrics_rows = 0
    model_run_counts: dict[str, int] = {}
    metrics_path = run_dir / "profiler_metrics.json"
    if metrics_path.is_file():
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            result = analyze(payload)
            metrics_status = str(result["status"])
            rows = result["rows"]
            metrics_rows = len(rows)
            counts = Counter(str(row["model"]) for row in rows)
            model_run_counts = dict(counts)
            if metrics_rows == 0:
                errors.append("profiler_metrics.json has no valid native rows")
            if counts and max(counts.values()) < min_repeated_runs:
                errors.append(
                    f"no model has at least {min_repeated_runs} repeated native rows"
                )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"profiler_metrics.json is invalid: {exc}")

    status = "FAIL" if errors else "PASS"
    return {
        "status": status,
        "run_dir": str(run_dir),
        "errors": errors,
        "warnings": warnings,
        "metrics_status": metrics_status,
        "metrics_rows": metrics_rows,
        "model_run_counts": model_run_counts,
        "min_repeated_runs": min_repeated_runs,
        "native_artifacts_required": require_native_artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--min-repeated-runs", type=int, default=3)
    parser.add_argument("--allow-missing-native-artifacts", action="store_true")
    args = parser.parse_args()

    result = check_run_artifacts(
        args.run_dir,
        min_repeated_runs=args.min_repeated_runs,
        require_native_artifacts=not args.allow_missing_native_artifacts,
    )
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
