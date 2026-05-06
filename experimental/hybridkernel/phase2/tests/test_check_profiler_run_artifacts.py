from __future__ import annotations

import json
import shutil
from pathlib import Path

from experimental.hybridkernel.phase2.analyze_profiler_metrics import analyze
from experimental.hybridkernel.phase2.check_profiler_run_artifacts import (
    READOUT_MARKERS,
    check_run_artifacts,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures/synthetic_profiler_run_packet"


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
    (run_dir / "logs/nsys_server_b1.log").write_text(
        "server profiler log\n", encoding="utf-8"
    )
    (run_dir / "logs/client_replay_b1.log").write_text(
        "client replay log\n", encoding="utf-8"
    )
    (run_dir / "nsys/granite_tiny_b1_decode64.nsys-rep").write_text(
        "native profiler export bytes\n" + ("x" * 2048), encoding="utf-8"
    )
    (run_dir / "ncu/suspicious_boundary_kernel.ncu-rep").write_text(
        "native profiler export bytes\n" + ("x" * 2048), encoding="utf-8"
    )
    readout_rows = "\n".join(f"| {marker} | evidence | no |" for marker in READOUT_MARKERS)
    (run_dir / "readout.md").write_text(
        "| Question | Evidence | Decision |\n|---|---|---|\n" + readout_rows + "\n",
        encoding="utf-8",
    )
    metrics_payload = {
        "rows": [
            {
                "model": "granite",
                "run_id": idx,
                "total_step_ms": 100.0,
                "attention_ssm_boundary_ms": 4.0,
                "matched_non_boundary_ms": 2.0,
                "recoverable_fraction": 0.60,
                "dtype": "bfloat16",
                "cuda_graph_enabled": True,
                "batch_shape": {
                    "batch_size": 1,
                    "prefill_tokens": 128,
                    "decode_tokens": 64,
                    "requests": 16,
                },
                "control_model_or_segment": "matched_transformer_block",
            }
            for idx in range(runs)
        ]
    }
    analysis = analyze(metrics_payload)
    (run_dir / "profiler_metrics.json").write_text(
        json.dumps(metrics_payload) + "\n",
        encoding="utf-8",
    )
    (run_dir / "profiler_analysis_gate.json").write_text(
        json.dumps(analysis, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "profiler_analysis_gate.md").write_text(
        "# HybridKernel Profiler Analysis Gate\n\n"
        f"Status: **{analysis['status']}**\n",
        encoding="utf-8",
    )


def test_complete_native_run_artifacts_pass(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "PASS"
    assert result["metrics_rows"] == 3
    assert result["model_run_counts"] == {"granite": 3}
    assert result["model_distinct_run_counts"] == {"granite": 3}
    assert max(result["model_config_run_counts"].values()) == 3


def test_requires_native_profiler_artifacts_by_default(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    (tmp_path / "nsys/granite_tiny_b1_decode64.nsys-rep").unlink()

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("Nsight Systems" in error for error in result["errors"])


def test_requires_server_and_client_logs(tmp_path: Path) -> None:
    server_missing = tmp_path / "server_missing"
    _write_complete_run(server_missing)
    (server_missing / "logs/nsys_server_b1.log").unlink()

    server_result = check_run_artifacts(server_missing)

    assert server_result["status"] == "FAIL"
    assert any("server profiler log" in error for error in server_result["errors"])

    client_missing = tmp_path / "client_missing"
    _write_complete_run(client_missing)
    (client_missing / "logs/client_replay_b1.log").unlink()

    client_result = check_run_artifacts(client_missing)

    assert client_result["status"] == "FAIL"
    assert any("client replay log" in error for error in client_result["errors"])


def test_rejects_server_warmup_log_without_profiler_log(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    (tmp_path / "logs/nsys_server_b1.log").unlink()
    (tmp_path / "logs/server_warmup.log").write_text(
        "server warmup log, not a profiler log\n", encoding="utf-8"
    )

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("server profiler log" in error for error in result["errors"])


def test_requires_complete_environment_capture(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    (tmp_path / "metadata/environment.txt").write_text(
        "nvidia-smi\nnsys version\npython -VV\n",
        encoding="utf-8",
    )

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any(
        "environment metadata does not mention ncu" in error for error in result["errors"]
    )


def test_rejects_tiny_or_placeholder_native_profiler_artifacts(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    (tmp_path / "nsys/granite_tiny_b1_decode64.nsys-rep").write_text(
        "placeholder\n" + ("x" * 2048), encoding="utf-8"
    )
    (tmp_path / "ncu/suspicious_boundary_kernel.ncu-rep").write_text(
        "tiny\n", encoding="utf-8"
    )

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("placeholder" in error for error in result["errors"])
    assert any("too small" in error for error in result["errors"])


def test_rejects_uppercase_todo_marker_inside_native_profiler_artifacts(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    (tmp_path / "nsys/granite_tiny_b1_decode64.nsys-rep").write_text(
        "TODO_NATIVE_PROFILE_FILL\n" + ("x" * 2048), encoding="utf-8"
    )

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("placeholder" in error for error in result["errors"])


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


def test_requires_vllm_serving_command(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    profile_scope_path = tmp_path / "metadata/profile_scope.json"
    profile_scope = json.loads(profile_scope_path.read_text(encoding="utf-8"))
    profile_scope["vllm_command"] = "python -m other_server --model granite"
    profile_scope_path.write_text(json.dumps(profile_scope) + "\n", encoding="utf-8")

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("vllm_command must mention vLLM" in error for error in result["errors"])


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


def test_requires_repeated_rows_for_same_run_config(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    metrics_path = tmp_path / "profiler_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    for idx, row in enumerate(payload["rows"]):
        row["batch_shape"]["decode_tokens"] = 64 + idx
    metrics_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("same model/config" in error for error in result["errors"])


def test_requires_profiler_analysis_gate_outputs(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    (tmp_path / "profiler_analysis_gate.json").unlink()

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("profiler_analysis_gate.json" in error for error in result["errors"])


def test_rejects_stale_profiler_analysis_gate_json(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    analysis_path = tmp_path / "profiler_analysis_gate.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis["status"] = "PROMOTE to prototype: stale copied result"
    analysis_path.write_text(json.dumps(analysis) + "\n", encoding="utf-8")

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("status does not match" in error for error in result["errors"])


def test_rejects_stale_profiler_analysis_rows(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    analysis_path = tmp_path / "profiler_analysis_gate.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis["rows"][0]["recoverable_gain_upper_bound"] = 0.999
    analysis_path.write_text(json.dumps(analysis) + "\n", encoding="utf-8")

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("rows do not match" in error for error in result["errors"])


def test_synthetic_fixture_documents_complete_packet_shape(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    shutil.copytree(FIXTURE_DIR, packet)

    result = check_run_artifacts(packet, require_native_artifacts=False)

    assert result["status"] == "PASS"
    assert result["metrics_rows"] == 3
    assert result["model_distinct_run_counts"] == {"synthetic-granite-fixture": 3}


def test_synthetic_fixture_is_not_native_evidence_by_default(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    shutil.copytree(FIXTURE_DIR, packet)

    result = check_run_artifacts(packet)

    assert result["status"] == "FAIL"
    assert any("placeholder" in error for error in result["errors"])
    assert any("too small" in error for error in result["errors"])
