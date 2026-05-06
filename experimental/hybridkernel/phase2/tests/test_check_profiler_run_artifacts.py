from __future__ import annotations

import json
from pathlib import Path

from experimental.hybridkernel.phase2.check_profiler_run_artifacts import (
    READOUT_MARKERS,
    check_run_artifacts,
)


def _write_complete_run(run_dir: Path, runs: int = 3) -> None:
    (run_dir / "metadata").mkdir(parents=True)
    (run_dir / "logs").mkdir()
    (run_dir / "nsys").mkdir()
    (run_dir / "ncu").mkdir()
    (run_dir / "metadata/environment.txt").write_text(
        "nvidia-smi\nnsys version\nncu version\npython -VV\n",
        encoding="utf-8",
    )
    (run_dir / "metadata/architecture_map.json").write_text("[]\n", encoding="utf-8")
    (run_dir / "metadata/profile_scope.json").write_text(
        json.dumps(
            {
                "profiled_process": "vllm_server",
                "nsys_profiled_process": "vllm_server",
                "ncu_profiled_process": "vllm_server",
                "trace_scope": "server-side CUDA kernels under fixed request replay",
                "nsys_trace_scope": "server-side CUDA kernels under fixed request replay",
                "ncu_trace_scope": "server-side CUDA kernels under suspicious-kernel replay",
                "vllm_command": "python -m vllm.entrypoints.openai.api_server --model granite",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "logs/nsys_b1.log").write_text("profile log\n", encoding="utf-8")
    (run_dir / "nsys/granite_tiny_b1_decode64.nsys-rep").write_text(
        "placeholder\n", encoding="utf-8"
    )
    (run_dir / "ncu/suspicious_boundary_kernel.ncu-rep").write_text(
        "placeholder\n", encoding="utf-8"
    )
    readout_rows = "\n".join(f"| {marker} | evidence | no |" for marker in READOUT_MARKERS)
    (run_dir / "readout.md").write_text(
        "| Question | Evidence | Decision |\n|---|---|---|\n" + readout_rows + "\n",
        encoding="utf-8",
    )
    (run_dir / "profiler_metrics.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "model": "granite",
                        "run_id": idx,
                        "total_step_ms": 100.0,
                        "attention_ssm_boundary_ms": 4.0,
                        "matched_non_boundary_ms": 2.0,
                        "recoverable_fraction": 0.60,
                    }
                    for idx in range(runs)
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_complete_native_run_artifacts_pass(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "PASS"
    assert result["metrics_rows"] == 3
    assert result["model_run_counts"] == {"granite": 3}
    assert result["model_distinct_run_counts"] == {"granite": 3}


def test_requires_native_profiler_artifacts_by_default(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    (tmp_path / "nsys/granite_tiny_b1_decode64.nsys-rep").unlink()

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("Nsight Systems" in error for error in result["errors"])


def test_requires_three_repeated_rows_for_review_gate(tmp_path: Path) -> None:
    _write_complete_run(tmp_path, runs=2)

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("at least 3 repeated" in error for error in result["errors"])


def test_rejects_client_only_profile_scope(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    (tmp_path / "metadata/profile_scope.json").write_text(
        json.dumps(
            {
                "profiled_process": "http_client",
                "nsys_profiled_process": "http_client",
                "ncu_profiled_process": "http_client",
                "trace_scope": "client request replay only",
                "vllm_command": "python phase2/profiler_driver.py",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("HTTP client" in error for error in result["errors"])


def test_rejects_client_only_ncu_profile_scope(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    profile_scope_path = tmp_path / "metadata/profile_scope.json"
    profile_scope = json.loads(profile_scope_path.read_text(encoding="utf-8"))
    profile_scope["ncu_profiled_process"] = "http_client"
    profile_scope["ncu_trace_scope"] = "client request replay only"
    profile_scope_path.write_text(json.dumps(profile_scope) + "\n", encoding="utf-8")

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("ncu_profiled_process" in error for error in result["errors"])
    assert any("ncu_trace_scope" in error for error in result["errors"])


def test_requires_distinct_repeated_run_ids(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    metrics_path = tmp_path / "profiler_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    for row in payload["rows"]:
        row["run_id"] = "same_trace"
    metrics_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("distinct repeated run_id" in error for error in result["errors"])
