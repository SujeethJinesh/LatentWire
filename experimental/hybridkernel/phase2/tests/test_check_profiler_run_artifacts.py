from __future__ import annotations

import json
import shutil
import hashlib
from pathlib import Path

from experimental.hybridkernel.phase2.analyze_profiler_metrics import analyze
from experimental.hybridkernel.phase2.check_profiler_run_artifacts import (
    DEFAULT_CROSS_FAMILY_MODEL,
    READOUT_MARKERS,
    check_run_artifacts,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures/synthetic_profiler_run_packet"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_profiler_artifact(path: Path) -> None:
    path.write_bytes((b"\x93NSIGHT\x00\xffnative-binary-export\x00" * 128)[:4096])


def _write_client_replay_log(
    run_dir: Path,
    filename: str,
    *,
    model: str,
    run_id: str,
    batch_size: int = 1,
    prefill_tokens: int = 128,
    decode_tokens: int = 64,
    requests: int = 16,
) -> None:
    (run_dir / "logs" / filename).write_text(
        json.dumps(
            {
                "model": model,
                "run_id": str(run_id),
                "dry_run": False,
                "token_counts_required": True,
                "token_count_source": "test_tokenizer",
                "requests": [
                    {
                        "status": "ok",
                        "batch_size": batch_size,
                        "prompt_sha256": [
                            _sha256_text(f"{run_id}:{request_id}:{sample_id}:prompt")
                            for sample_id in range(batch_size)
                        ],
                        "payload_sha256": _sha256_text(f"{run_id}:{request_id}:payload"),
                        "prompt_token_counts": [prefill_tokens for _ in range(batch_size)],
                        "prompt_token_count_total": batch_size * prefill_tokens,
                        "requested_decode_tokens": decode_tokens,
                        "expected_completion_tokens_total": batch_size * decode_tokens,
                        "response_usage": {"completion_tokens": batch_size * decode_tokens},
                    }
                    for request_id in range(requests)
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_snapshot_manifest(run_dir: Path, name: str) -> tuple[str, str]:
    path = run_dir / f"metadata/{name}_snapshot_manifest.json"
    path.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "config.json",
                        "sha256": "sha256:" + ("1" * 64),
                    },
                    {
                        "path": "tokenizer.json",
                        "sha256": "sha256:" + ("2" * 64),
                    },
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return f"metadata/{name}_snapshot_manifest.json", _sha256(path)


def _write_reduction_manifest(run_dir: Path, payload: dict[str, object] | None = None) -> None:
    if payload is None:
        payload = json.loads((run_dir / "profiler_metrics.json").read_text(encoding="utf-8"))
    rows = []
    for row in payload["rows"]:
        rows.append(
            {
                "run_id": row["run_id"],
                "row_role": row["row_role"],
                "model": row["model"],
                "reduction_source_path": "metadata/reduction_worksheet.tsv",
                "source_nsys_artifact": row["nsys_artifact"],
                "source_nsys_artifact_sha256": row["nsys_artifact_sha256"],
                "source_time_window_ms": row["time_window_ms"],
                "source_ncu_artifact": row["ncu_artifact"],
                "source_ncu_artifact_sha256": row["ncu_artifact_sha256"],
                "reduction_command": row["reduction_command"],
                "reduction_script_sha256": _sha256(run_dir / "metadata/reduction_worksheet.tsv"),
                "reduction_notes": row["reduction_notes"],
            }
        )
    (run_dir / "metadata/reduction_input_manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": "hybridkernel_reduction_inputs_v1",
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
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
    (run_dir / "metadata/environment.json").write_text(
        json.dumps(
            {
                "environment_version": "hybridkernel_environment_v1",
                "timestamp_utc": "2026-05-07T00:00:00Z",
                "hostname": "synthetic-gpu-node",
                "nvidia_smi": "NVIDIA-SMI synthetic driver",
                "nsys_version": "Nsight Systems synthetic",
                "ncu_version": "Nsight Compute synthetic",
                "python_version": "Python 3.11.0",
                "packages": {
                    "vllm": "0.8.0",
                    "torch": "2.7.0",
                    "triton": "3.3.0",
                    "transformers": "4.51.0",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    granite_manifest_path, granite_manifest_sha = _write_snapshot_manifest(run_dir, "granite")
    (run_dir / "metadata/model_provenance.json").write_text(
        json.dumps(
            {
                "provenance_version": "hybridkernel_model_provenance_v1",
                "models": [
                    {
                        "model_id": "granite",
                        "served_model_id": "granite",
                        "model_revision": "test-granite-model-revision",
                        "tokenizer_revision": "test-granite-tokenizer-revision",
                        "model_revision_is_immutable": True,
                        "tokenizer_revision_is_immutable": True,
                        "cache_source": "synthetic fixture",
                        "snapshot_manifest_path": granite_manifest_path,
                        "snapshot_manifest_sha256": granite_manifest_sha,
                        "local_files_only": True,
                        "trust_remote_code": True,
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "metadata/reduction_worksheet.tsv").write_text(
        "run_id\trow_role\tmodel\tnotes\nfixture\tprimary_hybrid\tgranite\tfilled\n",
        encoding="utf-8",
    )
    (run_dir / "metadata/architecture_map.json").write_text(
        '[{"model":"granite","boundary_count":1}]\n', encoding="utf-8"
    )
    (run_dir / "metadata/native_control_matrix.json").write_text(
        json.dumps(
            {
                "decision": "CONTROL_MATRIX_READY_NOT_NATIVE_EVIDENCE",
                "request_shape": {
                    "batch_size": 1,
                    "prefill_tokens": 128,
                    "decode_tokens": 64,
                    "requests": 16,
                    "dtype": "bfloat16",
                    "cuda_graph_enabled": True,
                },
                "rows": [
                    {
                        "row_role": "primary_hybrid",
                        "model": "granite",
                        "control_family": "same_family_matched_segment",
                        "control_model_or_segment": "matched_transformer_block",
                        "boundary_direction": "mixed_attention_ssm",
                    },
                    {
                        "row_role": "same_family_control",
                        "model": "granite",
                        "control_family": "same_model_non_boundary_segment_control",
                        "control_model_or_segment": "same_model_non_boundary_segment_control",
                        "boundary_direction": "non_boundary_same_family",
                    },
                    {
                        "row_role": "cross_family_falsification",
                        "model": DEFAULT_CROSS_FAMILY_MODEL,
                        "control_family": "cross_family_hybrid_control",
                        "control_model_or_segment": "qwen_hybrid_control",
                        "boundary_direction": "linear_attention_gated_delta_boundary",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
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
                "model": "granite",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "logs/nsys_server_b1.log").write_text(
        "nsys vllm server cuda profiler log\n", encoding="utf-8"
    )
    for idx in range(runs):
        _write_client_replay_log(
            run_dir,
            f"client_replay_granite_run{idx}.log",
            model="granite",
            run_id=str(idx),
        )
        _write_profiler_artifact(run_dir / f"nsys/granite_tiny_b1_decode64_run{idx}.nsys-rep")
        _write_profiler_artifact(run_dir / f"ncu/suspicious_boundary_kernel_run{idx}.ncu-rep")
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
                "row_role": "primary_hybrid",
                "control_family": "same_family_matched_segment",
                "boundary_direction": "mixed_attention_ssm",
                "nsys_artifact": f"nsys/granite_tiny_b1_decode64_run{idx}.nsys-rep",
                "nsys_artifact_sha256": _sha256(
                    run_dir / f"nsys/granite_tiny_b1_decode64_run{idx}.nsys-rep"
                ),
                "ncu_artifact": f"ncu/suspicious_boundary_kernel_run{idx}.ncu-rep",
                "ncu_artifact_sha256": _sha256(
                    run_dir / f"ncu/suspicious_boundary_kernel_run{idx}.ncu-rep"
                ),
                "kernel_names": ["synthetic_boundary_kernel"],
                "boundary_indices": [0],
                "control_window_ids": [],
                "time_window_ms": {"start": float(idx), "end": float(idx) + 1.0},
                "ncu_launch_selection": {
                    "kernel_regex": "synthetic_boundary_kernel",
                    "launch_skip": idx,
                    "launch_count": 1,
                    "source_nsys_artifact": f"nsys/granite_tiny_b1_decode64_run{idx}.nsys-rep",
                    "source_time_window_ms": {"start": float(idx), "end": float(idx) + 1.0},
                    "derivation_notes": "Selected from the matching synthetic Nsight Systems window.",
                },
                "recoverable_fraction_basis": "Synthetic fixture assumes 60% recovery from a fused boundary operator.",
                "reduction_command": (
                    "python experimental/hybridkernel/phase2/tests/"
                    "test_check_profiler_run_artifacts.py::_write_complete_run"
                ),
                "reduction_notes": "Reduced from synthetic fixture row.",
            }
            for idx in range(runs)
        ]
    }
    analysis = analyze(metrics_payload)
    (run_dir / "profiler_metrics.json").write_text(
        json.dumps(metrics_payload) + "\n",
        encoding="utf-8",
    )
    _write_reduction_manifest(run_dir, metrics_payload)
    (run_dir / "profiler_analysis_gate.json").write_text(
        json.dumps(analysis, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "profiler_analysis_gate.md").write_text(
        "# HybridKernel Profiler Analysis Gate\n\n"
        f"Status: **{analysis['status']}**\n",
        encoding="utf-8",
    )


def _add_qwen_control_rows(run_dir: Path) -> None:
    (run_dir / "metadata/architecture_map.json").write_text(
        json.dumps(
            [
                {"model": "granite", "boundary_count": 1},
                {"model": DEFAULT_CROSS_FAMILY_MODEL, "boundary_count": 1},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    provenance_path = run_dir / "metadata/model_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    qwen_manifest_path, qwen_manifest_sha = _write_snapshot_manifest(run_dir, "qwen")
    provenance["models"].append(
        {
            "model_id": DEFAULT_CROSS_FAMILY_MODEL,
            "served_model_id": DEFAULT_CROSS_FAMILY_MODEL,
            "model_revision": "test-qwen-model-revision",
            "tokenizer_revision": "test-qwen-tokenizer-revision",
            "model_revision_is_immutable": True,
            "tokenizer_revision_is_immutable": True,
            "cache_source": "synthetic fixture",
            "snapshot_manifest_path": qwen_manifest_path,
            "snapshot_manifest_sha256": qwen_manifest_sha,
            "local_files_only": True,
            "trust_remote_code": True,
        }
    )
    provenance_path.write_text(json.dumps(provenance) + "\n", encoding="utf-8")
    metrics_path = run_dir / "profiler_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    qwen_rows = []
    for idx, base in enumerate(payload["rows"]):
        _write_client_replay_log(
            run_dir,
            f"client_replay_qwen_run{idx}.log",
            model=DEFAULT_CROSS_FAMILY_MODEL,
            run_id=f"qwen-{idx}",
        )
        row = dict(base)
        row.update(
            {
                "model": DEFAULT_CROSS_FAMILY_MODEL,
                "run_id": f"qwen-{idx}",
                "row_role": "cross_family_falsification",
                "control_family": "cross_family_hybrid_control",
                "control_model_or_segment": "qwen_hybrid_control",
                "boundary_direction": "linear_attention_gated_delta_boundary",
                "attention_ssm_boundary_ms": 2.0,
                "matched_non_boundary_ms": 1.9,
                "boundary_indices": [0],
            }
        )
        qwen_rows.append(row)
    payload["rows"].extend(qwen_rows)
    metrics_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    _write_reduction_manifest(run_dir, payload)
    analysis = analyze(payload)
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


def test_rejects_missing_run_specific_client_replay_log(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    (tmp_path / "logs/client_replay_granite_run1.log").unlink()

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any(
        "client replay log does not match profiler_metrics.json run_id and batch_shape"
        in error
        for error in result["errors"]
    )


def test_rejects_missing_ncu_launch_selection(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    metrics_path = tmp_path / "profiler_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["rows"][0].pop("ncu_launch_selection")
    metrics_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    analysis = analyze(payload)
    (tmp_path / "profiler_analysis_gate.json").write_text(
        json.dumps(analysis, indent=2) + "\n",
        encoding="utf-8",
    )

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("ncu_launch_selection must be an object" in error for error in result["errors"])


def test_rejects_multi_model_metrics_without_profile_model_scopes(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    _add_qwen_control_rows(tmp_path)

    result = check_run_artifacts(tmp_path, require_native_artifacts=False)

    assert result["status"] == "FAIL"
    assert any("profile_scope.json models do not cover" in error for error in result["errors"])
    assert any("multi-model profiler_metrics.json requires" in error for error in result["errors"])


def test_accepts_multi_model_metrics_with_profile_model_scopes(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    _add_qwen_control_rows(tmp_path)
    profile_scope_path = tmp_path / "metadata/profile_scope.json"
    profile_scope = json.loads(profile_scope_path.read_text(encoding="utf-8"))
    profile_scope["model_scopes"] = [
        {
            "model": "granite",
            "row_roles": ["primary_hybrid", "same_family_control"],
            "vllm_command": "python -m vllm.entrypoints.openai.api_server --model granite",
        },
        {
            "model": DEFAULT_CROSS_FAMILY_MODEL,
            "row_roles": ["cross_family_falsification"],
            "vllm_command": f"python -m vllm.entrypoints.openai.api_server --model {DEFAULT_CROSS_FAMILY_MODEL}",
        },
    ]
    profile_scope_path.write_text(json.dumps(profile_scope) + "\n", encoding="utf-8")

    result = check_run_artifacts(tmp_path, require_native_artifacts=False)

    assert result["status"] == "PASS"
    assert set(result["model_run_counts"]) == {"granite", DEFAULT_CROSS_FAMILY_MODEL}


def test_primary_gate_clear_without_controls_stays_audit_only(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    metrics_path = tmp_path / "profiler_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    for row in payload["rows"]:
        row["attention_ssm_boundary_ms"] = 8.0
    analysis = analyze(payload)
    metrics_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    _write_reduction_manifest(tmp_path, payload)
    (tmp_path / "profiler_analysis_gate.json").write_text(
        json.dumps(analysis, indent=2) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "profiler_analysis_gate.md").write_text(
        "# HybridKernel Profiler Analysis Gate\n\n"
        f"Status: **{analysis['status']}**\n",
        encoding="utf-8",
    )

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "PASS"
    assert result["metrics_status"].startswith("WEAKLY ALIVE")
    assert any("control/falsification rows" in warning for warning in result["warnings"])


def test_require_full_matrix_rejects_audit_only_primary_rows(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)

    result = check_run_artifacts(tmp_path, require_full_matrix=True)

    assert result["status"] == "FAIL"
    assert result["require_full_matrix"] is True
    assert any("control/falsification rows" in error for error in result["errors"])


def test_require_full_matrix_rejects_single_control_rows(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    (tmp_path / "metadata/architecture_map.json").write_text(
        json.dumps(
            [
                {"model": "granite", "boundary_count": 1},
                {"model": DEFAULT_CROSS_FAMILY_MODEL, "boundary_count": 1},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    profile_scope_path = tmp_path / "metadata/profile_scope.json"
    profile_scope = json.loads(profile_scope_path.read_text(encoding="utf-8"))
    profile_scope["model_scopes"] = [
        {
            "model": "granite",
            "row_roles": ["primary_hybrid", "same_family_control"],
            "vllm_command": "python -m vllm.entrypoints.openai.api_server --model granite",
        },
        {
            "model": DEFAULT_CROSS_FAMILY_MODEL,
            "row_roles": ["cross_family_falsification"],
            "vllm_command": f"python -m vllm.entrypoints.openai.api_server --model {DEFAULT_CROSS_FAMILY_MODEL}",
        },
    ]
    profile_scope_path.write_text(json.dumps(profile_scope) + "\n", encoding="utf-8")
    provenance_path = tmp_path / "metadata/model_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    qwen_manifest_path, qwen_manifest_sha = _write_snapshot_manifest(tmp_path, "qwen")
    provenance["models"].append(
        {
            "model_id": DEFAULT_CROSS_FAMILY_MODEL,
            "served_model_id": DEFAULT_CROSS_FAMILY_MODEL,
            "model_revision": "test-qwen-model-revision",
            "tokenizer_revision": "test-qwen-tokenizer-revision",
            "model_revision_is_immutable": True,
            "tokenizer_revision_is_immutable": True,
            "cache_source": "synthetic fixture",
            "snapshot_manifest_path": qwen_manifest_path,
            "snapshot_manifest_sha256": qwen_manifest_sha,
            "local_files_only": True,
            "trust_remote_code": True,
        }
    )
    provenance_path.write_text(json.dumps(provenance) + "\n", encoding="utf-8")

    metrics_path = tmp_path / "profiler_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    base = payload["rows"][0]
    control_rows = []
    for role, model, run_id, nsys_name, ncu_name, boundary_direction, control_family, segment in [
        (
            "same_family_control",
            "granite",
            "same-family-single",
            "nsys/same_family_single.nsys-rep",
            "ncu/same_family_single.ncu-rep",
            "non_boundary_same_family",
            "same_model_non_boundary_segment_control",
            "same_model_non_boundary_segment_control",
        ),
        (
            "cross_family_falsification",
            DEFAULT_CROSS_FAMILY_MODEL,
            "cross-family-single",
            "nsys/cross_family_single.nsys-rep",
            "ncu/cross_family_single.ncu-rep",
            "linear_attention_gated_delta_boundary",
            "cross_family_hybrid_control",
            "qwen_hybrid_control",
        ),
    ]:
        _write_client_replay_log(
            tmp_path,
            f"client_replay_{run_id}.log",
            model=model,
            run_id=run_id,
        )
        _write_profiler_artifact(tmp_path / nsys_name)
        _write_profiler_artifact(tmp_path / ncu_name)
        row = dict(base)
        row.update(
            {
                "model": model,
                "run_id": run_id,
                "row_role": role,
                "control_family": control_family,
                "control_model_or_segment": segment,
                "boundary_direction": boundary_direction,
                "attention_ssm_boundary_ms": 2.0,
                "matched_non_boundary_ms": 1.9,
                "nsys_artifact": nsys_name,
                "nsys_artifact_sha256": _sha256(tmp_path / nsys_name),
                "ncu_artifact": ncu_name,
                "ncu_artifact_sha256": _sha256(tmp_path / ncu_name),
                "boundary_indices": [0] if role == "cross_family_falsification" else [],
                "control_window_ids": (
                    ["granite-non-boundary-window-0"]
                    if role == "same_family_control"
                    else []
                ),
                "time_window_ms": {"start": 10.0 + len(control_rows), "end": 11.0 + len(control_rows)},
                "ncu_launch_selection": {
                    "kernel_regex": "synthetic_boundary_kernel",
                    "launch_skip": 10 + len(control_rows),
                    "launch_count": 1,
                    "source_nsys_artifact": nsys_name,
                    "source_time_window_ms": {"start": 10.0 + len(control_rows), "end": 11.0 + len(control_rows)},
                    "derivation_notes": "Selected from the matching synthetic control window.",
                },
            }
        )
        control_rows.append(row)
    payload["rows"].extend(control_rows)
    metrics_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    analysis = analyze(payload)
    (tmp_path / "profiler_analysis_gate.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")

    result = check_run_artifacts(tmp_path, require_full_matrix=True)

    assert result["status"] == "FAIL"
    assert any("at least 3 same_family_control rows" in error for error in result["errors"])
    assert any("at least 3 cross_family_falsification rows" in error for error in result["errors"])


def test_requires_native_profiler_artifacts_by_default(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    (tmp_path / "nsys/granite_tiny_b1_decode64_run0.nsys-rep").unlink()

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("nsys_artifact does not exist" in error for error in result["errors"])


def test_no_boundary_signal_kill_packet_allows_missing_ncu(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    for artifact in (tmp_path / "ncu").glob("*.ncu-rep"):
        artifact.unlink()
    metrics_path = tmp_path / "profiler_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    for row in payload["rows"]:
        row["attention_ssm_boundary_ms"] = 2.0
        row["matched_non_boundary_ms"] = 2.0
        row["ncu_artifact"] = "not_run_no_boundary_signal"
        row["ncu_artifact_sha256"] = "not_run_no_boundary_signal"
        row["kernel_names"] = ["no_suspicious_boundary_kernel_in_nsys"]
        row["reduction_notes"] = (
            "No suspicious boundary kernel in Nsight Systems; no boundary signal "
            "available for an Nsight Compute target."
        )
    analysis = analyze(payload)
    metrics_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    _write_reduction_manifest(tmp_path, payload)
    (tmp_path / "profiler_analysis_gate.json").write_text(
        json.dumps(analysis, indent=2) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "profiler_analysis_gate.md").write_text(
        "# HybridKernel Profiler Analysis Gate\n\n"
        f"Status: **{analysis['status']}**\n",
        encoding="utf-8",
    )

    result = check_run_artifacts(tmp_path, packet_mode="no_boundary_signal_kill")

    assert result["status"] == "PASS"
    assert any("Nsight Compute artifact is optional" in warning for warning in result["warnings"])


def test_no_boundary_signal_kill_requires_clean_negative_nsys_evidence(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    for artifact in (tmp_path / "ncu").glob("*.ncu-rep"):
        artifact.unlink()
    metrics_path = tmp_path / "profiler_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    for row in payload["rows"]:
        row["ncu_artifact"] = "not_run_no_boundary_signal"
        row["ncu_artifact_sha256"] = "not_run_no_boundary_signal"
    metrics_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = check_run_artifacts(tmp_path, packet_mode="no_boundary_signal_kill")

    assert result["status"] == "FAIL"
    assert any("clean kill" in error for error in result["errors"])
    assert any("explicit no-boundary-signal" in error for error in result["errors"])


def test_requires_server_and_client_logs(tmp_path: Path) -> None:
    server_missing = tmp_path / "server_missing"
    _write_complete_run(server_missing)
    (server_missing / "logs/nsys_server_b1.log").unlink()

    server_result = check_run_artifacts(server_missing)

    assert server_result["status"] == "FAIL"
    assert any("server profiler log" in error for error in server_result["errors"])

    client_missing = tmp_path / "client_missing"
    _write_complete_run(client_missing)
    (client_missing / "logs/client_replay_granite_run0.log").unlink()

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


def test_rejects_empty_or_placeholder_native_logs(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    (tmp_path / "logs/nsys_server_b1.log").write_text("", encoding="utf-8")
    (tmp_path / "logs/client_replay_granite_run0.log").write_text(
        "placeholder client replay\n", encoding="utf-8"
    )

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("profiling log is too small" in error for error in result["errors"])
    assert any("placeholder evidence" in error for error in result["errors"])


def test_rejects_arbitrary_native_log_payloads(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    (tmp_path / "logs/nsys_server_b1.log").write_text("abcdefgh\n", encoding="utf-8")
    (tmp_path / "logs/client_replay_granite_run0.log").write_text("abcdefgh\n", encoding="utf-8")

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("server profiler log lacks" in error for error in result["errors"])
    assert any("client replay log lacks" in error for error in result["errors"])


def test_rejects_marker_only_client_log_without_profiler_driver_json(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    (tmp_path / "logs/client_replay_granite_run0.log").write_text(
        '"model" "requests" "status"\n', encoding="utf-8"
    )

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("not valid profiler_driver JSON" in error for error in result["errors"])


def test_rejects_unregistered_high_recoverable_fraction(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    metrics_path = tmp_path / "profiler_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["rows"][0]["recoverable_fraction"] = 0.95
    metrics_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("recoverable_fraction must be in [0, 0.60]" in error for error in result["errors"])


def test_rejects_client_replay_without_top_level_model(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    (tmp_path / "logs/client_replay_granite_run0.log").write_text(
        '{"dry_run":false,"requests":[{"model":"nested-only","status":"ok"}]}\n',
        encoding="utf-8",
    )

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("top-level model" in error for error in result["errors"])


def test_rejects_client_replay_without_explicit_non_dry_run(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    (tmp_path / "logs/client_replay_granite_run0.log").write_text(
        '{"model":"granite","requests":[{"status":"ok"}]}\n',
        encoding="utf-8",
    )

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("dry_run=false" in error for error in result["errors"])


def test_rejects_client_replay_shape_mismatch(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    (tmp_path / "logs/client_replay_granite_run0.log").write_text(
        json.dumps(
            {
                "model": "granite",
                "run_id": "0",
                "dry_run": False,
                "token_counts_required": True,
                "token_count_source": "test_tokenizer",
                "requests": [
                    {
                        "status": "ok",
                        "batch_size": 1,
                        "prompt_token_counts": [127],
                        "prompt_token_count_total": 127,
                        "requested_decode_tokens": 64,
                        "expected_completion_tokens_total": 64,
                        "response_usage": {"completion_tokens": 64},
                    }
                    for _ in range(16)
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("run_id and batch_shape" in error for error in result["errors"])


def test_rejects_client_replay_completion_length_mismatch(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    log_path = tmp_path / "logs/client_replay_granite_run0.log"
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    payload["requests"][0]["response_usage"] = {"completion_tokens": 63}
    log_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("completion_tokens must equal batch_size * requested_decode_tokens" in error for error in result["errors"])


def test_rejects_client_replay_expected_completion_total_mismatch(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    log_path = tmp_path / "logs/client_replay_granite_run0.log"
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    payload["requests"][0]["expected_completion_tokens_total"] = 65
    log_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any(
        "expected_completion_tokens_total must equal batch_size * requested_decode_tokens" in error
        for error in result["errors"]
    )


def test_rejects_client_replay_without_expected_completion_total(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    log_path = tmp_path / "logs/client_replay_granite_run0.log"
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    del payload["requests"][0]["expected_completion_tokens_total"]
    log_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("positive expected_completion_tokens_total" in error for error in result["errors"])


def test_rejects_client_replay_boolean_integer_fields(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    log_path = tmp_path / "logs/client_replay_granite_run0.log"
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    payload["requests"][0]["batch_size"] = True
    payload["requests"][1]["requested_decode_tokens"] = True
    payload["requests"][2]["response_usage"] = {"completion_tokens": True}
    log_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("positive batch_size" in error for error in result["errors"])
    assert any("positive requested_decode_tokens" in error for error in result["errors"])
    assert any("response_usage.completion_tokens must be positive" in error for error in result["errors"])


def test_rejects_client_replay_mixed_request_shapes(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    log_path = tmp_path / "logs/client_replay_granite_run0.log"
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    payload["requests"][1]["batch_size"] = 8
    payload["requests"][1]["prompt_token_counts"] = [128] * 8
    payload["requests"][1]["prompt_token_count_total"] = 1024
    payload["requests"][1]["expected_completion_tokens_total"] = 512
    payload["requests"][1]["response_usage"] = {"completion_tokens": 512}
    log_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("one fixed request shape" in error for error in result["errors"])


def test_rejects_client_replay_without_prompt_payload_hashes(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    log_path = tmp_path / "logs/client_replay_granite_run0.log"
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    payload["requests"][0].pop("prompt_sha256")
    payload["requests"][1]["payload_sha256"] = ""
    log_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("prompt_sha256" in error for error in result["errors"])
    assert any("payload_sha256" in error for error in result["errors"])


def test_rejects_client_replay_batch_size_mismatch(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    metrics_path = tmp_path / "profiler_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    for row in payload["rows"]:
        row["batch_shape"]["batch_size"] = 8
    analysis = analyze(payload)
    metrics_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    (tmp_path / "profiler_analysis_gate.json").write_text(
        json.dumps(analysis, indent=2) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "profiler_analysis_gate.md").write_text(
        "# HybridKernel Profiler Analysis Gate\n\n"
        f"Status: **{analysis['status']}**\n",
        encoding="utf-8",
    )

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("client replay shape does not match" in error for error in result["errors"])


def test_accepts_batch_replay_with_per_sample_prefill_tokens(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    for log_path in sorted((tmp_path / "logs").glob("client_replay_granite_run*.log")):
        log_payload = json.loads(log_path.read_text(encoding="utf-8"))
        for request in log_payload["requests"]:
            request["batch_size"] = 8
            request["prompt_sha256"] = [_sha256_text(f"{log_path.name}:{idx}:prompt") for idx in range(8)]
            request["prompt_token_counts"] = [128] * 8
            request["prompt_token_count_total"] = 1024
            request["expected_completion_tokens_total"] = 512
            request["response_usage"] = {"completion_tokens": 512}
        log_path.write_text(json.dumps(log_payload) + "\n", encoding="utf-8")

    metrics_path = tmp_path / "profiler_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    for row in payload["rows"]:
        row["batch_shape"]["batch_size"] = 8
        row["batch_shape"]["prefill_tokens"] = 128
    control_matrix_path = tmp_path / "metadata/native_control_matrix.json"
    control_matrix = json.loads(control_matrix_path.read_text(encoding="utf-8"))
    control_matrix["request_shape"]["batch_size"] = 8
    control_matrix_path.write_text(json.dumps(control_matrix) + "\n", encoding="utf-8")
    analysis = analyze(payload)
    metrics_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    (tmp_path / "profiler_analysis_gate.json").write_text(
        json.dumps(analysis, indent=2) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "profiler_analysis_gate.md").write_text(
        "# HybridKernel Profiler Analysis Gate\n\n"
        f"Status: **{analysis['status']}**\n",
        encoding="utf-8",
    )

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "PASS"


def test_rejects_dry_run_or_failed_client_replay_logs(tmp_path: Path) -> None:
    dry_run = tmp_path / "dry_run"
    _write_complete_run(dry_run)
    (dry_run / "logs/client_replay_granite_run0.log").write_text(
        '{"model":"granite","dry_run":true,"requests":[{"status":"dry_run"}]}\n',
        encoding="utf-8",
    )

    dry_result = check_run_artifacts(dry_run)

    assert dry_result["status"] == "FAIL"
    assert any("dry_run=false" in error for error in dry_result["errors"])
    assert any("status=ok" in error for error in dry_result["errors"])

    failed = tmp_path / "failed"
    _write_complete_run(failed)
    (failed / "logs/client_replay_granite_run0.log").write_text(
        '{"model":"granite","dry_run":false,"requests":[{"status":"error:timeout"}]}\n',
        encoding="utf-8",
    )

    failed_result = check_run_artifacts(failed)

    assert failed_result["status"] == "FAIL"
    assert any("status=ok" in error for error in failed_result["errors"])


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


def test_requires_structured_environment_versions(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    environment_path = tmp_path / "metadata/environment.json"
    payload = json.loads(environment_path.read_text(encoding="utf-8"))
    payload["packages"]["vllm"] = "unavailable"
    environment_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("environment.json packages.vllm" in error for error in result["errors"])


def test_requires_model_snapshot_manifest_hashes(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    provenance_path = tmp_path / "metadata/model_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["models"][0]["snapshot_manifest_sha256"] = "sha256:" + ("0" * 64)
    provenance_path.write_text(json.dumps(provenance) + "\n", encoding="utf-8")

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("snapshot_manifest_sha256 must match" in error for error in result["errors"])


def test_requires_immutable_model_revision_attestation(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    provenance_path = tmp_path / "metadata/model_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["models"][0]["model_revision_is_immutable"] = False
    provenance["models"][0].pop("tokenizer_revision_is_immutable")
    provenance_path.write_text(json.dumps(provenance) + "\n", encoding="utf-8")

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("model_revision_is_immutable must be true" in error for error in result["errors"])
    assert any("tokenizer_revision_is_immutable must be true" in error for error in result["errors"])


def test_rejects_mutable_model_revision_aliases(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    provenance_path = tmp_path / "metadata/model_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["models"][0]["model_revision"] = "main"
    provenance["models"][0]["tokenizer_revision"] = "refs/heads/master"
    provenance_path.write_text(json.dumps(provenance) + "\n", encoding="utf-8")

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("model_revision must not be a mutable branch alias" in error for error in result["errors"])
    assert any("tokenizer_revision must not be a mutable branch alias" in error for error in result["errors"])


def test_requires_same_family_control_window_ids(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    metrics_path = tmp_path / "profiler_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    control_row = dict(payload["rows"][0])
    control_row.update(
        {
            "run_id": "same-family-control-0",
            "row_role": "same_family_control",
            "control_family": "same_model_non_boundary_segment_control",
            "control_model_or_segment": "same_model_non_boundary_segment_control",
            "boundary_direction": "non_boundary_same_family",
            "attention_ssm_boundary_ms": 2.0,
            "matched_non_boundary_ms": 1.9,
            "boundary_indices": [],
            "control_window_ids": [],
            "time_window_ms": {"start": 10.0, "end": 11.0},
        }
    )
    payload["rows"].append(control_row)
    metrics_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    analysis = analyze(payload)
    (tmp_path / "profiler_analysis_gate.json").write_text(
        json.dumps(analysis, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_reduction_manifest(tmp_path, payload)

    result = check_run_artifacts(tmp_path, require_native_artifacts=False)

    assert result["status"] == "FAIL"
    assert any("same_family_control rows must name control_window_ids" in error for error in result["errors"])


def test_rejects_unfilled_readout_template_cells(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    (tmp_path / "readout.md").write_text(
        "| Question | Evidence | Decision |\n"
        "|---|---|---|\n"
        "| Distinct boundary conversion/materialization kernel? | kernel names and timestamps | yes/no |\n"
        "| Boundary idle or launch gap? | median and paired deltas | yes/no |\n"
        "| Extra DRAM/L2 traffic near boundary? | NCU bytes vs matched controls | yes/no |\n"
        "| End-to-end impact estimate clears 3%? | formula and confidence interval | yes/no |\n"
        "| Same-family controls available? | model/control rows | yes/no |\n"
        "| Cross-family falsification attempted? | model/control rows | yes/no |\n",
        encoding="utf-8",
    )

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("unfilled template placeholder" in error for error in result["errors"])


def test_rejects_tiny_or_placeholder_native_profiler_artifacts(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    (tmp_path / "nsys/granite_tiny_b1_decode64_run0.nsys-rep").write_text(
        "placeholder\n" + ("x" * 2048), encoding="utf-8"
    )
    (tmp_path / "ncu/suspicious_boundary_kernel_run0.ncu-rep").write_text(
        "tiny\n", encoding="utf-8"
    )

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("placeholder" in error for error in result["errors"])
    assert any("too small" in error for error in result["errors"])


def test_rejects_plain_text_native_profiler_artifacts(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    (tmp_path / "nsys/granite_tiny_b1_decode64_run0.nsys-rep").write_text(
        "nsight exported timeline report\n" + ("x" * 2048), encoding="utf-8"
    )

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("plain text" in error for error in result["errors"])


def test_rejects_uppercase_todo_marker_inside_native_profiler_artifacts(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    (tmp_path / "nsys/granite_tiny_b1_decode64_run0.nsys-rep").write_text(
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


def test_requires_metric_row_provenance(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    metrics_path = tmp_path / "profiler_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    del payload["rows"][0]["kernel_names"]
    metrics_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("missing provenance fields" in error for error in result["errors"])


def test_requires_reduction_input_manifest(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    (tmp_path / "metadata/reduction_input_manifest.json").unlink()

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("reduction_input_manifest.json" in error for error in result["errors"])


def test_rejects_reduction_manifest_that_does_not_match_metrics(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    manifest_path = tmp_path / "metadata/reduction_input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manifest_version"] = 1
    manifest["rows"][0]["source_time_window_ms"] = {"start": 99.0, "end": 100.0}
    manifest["rows"][1]["reduction_script_sha256"] = "sha256:not-a-real-hash"
    manifest["rows"][2]["reduction_notes"] = "Different manual reduction notes."
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("manifest_version" in error for error in result["errors"])
    assert any("source_time_window_ms must match" in error for error in result["errors"])
    assert any("reduction_script_sha256" in error for error in result["errors"])
    assert any("reduction_notes must match" in error for error in result["errors"])


def test_rejects_metric_row_missing_artifact_path(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    metrics_path = tmp_path / "profiler_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["rows"][0]["nsys_artifact"] = "nsys/missing_trace.nsys-rep"
    metrics_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("nsys_artifact does not exist" in error for error in result["errors"])


def test_rejects_metric_row_artifact_hash_mismatch(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    metrics_path = tmp_path / "profiler_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["rows"][0]["nsys_artifact_sha256"] = "sha256:" + ("0" * 64)
    metrics_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("nsys_artifact_sha256 mismatch" in error for error in result["errors"])


def test_rejects_metric_row_artifact_path_outside_run_dir(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    metrics_path = tmp_path / "profiler_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["rows"][0]["nsys_artifact"] = "../outside.nsys-rep"
    metrics_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("must stay inside the run directory" in error for error in result["errors"])


def test_rejects_metric_row_artifact_wrong_extension(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    wrong_path = tmp_path / "ncu/suspicious_boundary_kernel.txt"
    wrong_path.write_text("native profiler export bytes\n" + ("x" * 2048), encoding="utf-8")
    metrics_path = tmp_path / "profiler_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["rows"][0]["ncu_artifact"] = "ncu/suspicious_boundary_kernel.txt"
    metrics_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("ncu_artifact must point to one of" in error for error in result["errors"])


def test_rejects_packet_model_mismatches(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    profile_scope_path = tmp_path / "metadata/profile_scope.json"
    profile_scope = json.loads(profile_scope_path.read_text(encoding="utf-8"))
    profile_scope["model"] = "other-model"
    profile_scope_path.write_text(json.dumps(profile_scope) + "\n", encoding="utf-8")

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("model does not match" in error for error in result["errors"])


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


def test_rejects_reused_artifacts_for_repeated_rows(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    metrics_path = tmp_path / "profiler_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    for row in payload["rows"]:
        row["nsys_artifact"] = "nsys/granite_tiny_b1_decode64_run0.nsys-rep"
        row["ncu_artifact"] = "ncu/suspicious_boundary_kernel_run0.ncu-rep"
    metrics_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("distinct nsys_artifact" in error for error in result["errors"])
    assert any("distinct ncu_artifact" in error for error in result["errors"])


def test_rejects_reused_artifacts_across_profiler_roles(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    architecture_map_path = tmp_path / "metadata/architecture_map.json"
    architecture_map_path.write_text(
        json.dumps(
            [
                {"model": "granite", "boundary_count": 1},
                {"model": DEFAULT_CROSS_FAMILY_MODEL, "boundary_count": 1},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    provenance_path = tmp_path / "metadata/model_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    qwen_manifest_path, qwen_manifest_sha = _write_snapshot_manifest(tmp_path, "qwen")
    provenance["models"].append(
        {
            "model_id": DEFAULT_CROSS_FAMILY_MODEL,
            "served_model_id": DEFAULT_CROSS_FAMILY_MODEL,
            "model_revision": "test-qwen-model-revision",
            "tokenizer_revision": "test-qwen-tokenizer-revision",
            "model_revision_is_immutable": True,
            "tokenizer_revision_is_immutable": True,
            "cache_source": "synthetic fixture",
            "snapshot_manifest_path": qwen_manifest_path,
            "snapshot_manifest_sha256": qwen_manifest_sha,
            "local_files_only": True,
            "trust_remote_code": True,
        }
    )
    provenance_path.write_text(json.dumps(provenance) + "\n", encoding="utf-8")
    metrics_path = tmp_path / "profiler_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = list(payload["rows"])
    for idx, base in enumerate(payload["rows"]):
        _write_client_replay_log(
            tmp_path,
            f"client_replay_qwen_reuse_run{idx}.log",
            model=DEFAULT_CROSS_FAMILY_MODEL,
            run_id=str(base["run_id"]),
        )
        same_family = dict(base)
        same_family.update(
            {
                "row_role": "same_family_control",
                "control_family": "same_family_non_boundary_segment_control",
                "control_model_or_segment": "same_model_non_boundary_segment_control",
                "attention_ssm_boundary_ms": 2.0,
                "matched_non_boundary_ms": 1.9,
                "boundary_indices": [],
                "control_window_ids": ["granite-reused-non-boundary-window"],
            }
        )
        cross_family = dict(base)
        cross_family.update(
            {
                "model": DEFAULT_CROSS_FAMILY_MODEL,
                "row_role": "cross_family_falsification",
                "control_family": "cross_family_hybrid_control",
                "control_model_or_segment": "qwen_hybrid_control",
                "attention_ssm_boundary_ms": 2.0,
                "matched_non_boundary_ms": 1.9,
                "boundary_indices": [0],
            }
        )
        rows.extend([same_family, cross_family])
    payload["rows"] = rows
    metrics_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    analysis = analyze(payload)
    (tmp_path / "profiler_analysis_gate.json").write_text(
        json.dumps(analysis, indent=2) + "\n",
        encoding="utf-8",
    )

    result = check_run_artifacts(tmp_path)

    assert result["status"] == "FAIL"
    assert any("reuse the same Nsight Systems artifact" in error for error in result["errors"])
    assert any("reuse the same Nsight Compute artifact" in error for error in result["errors"])


def test_rejects_metric_rows_outside_native_control_matrix(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    metrics_path = tmp_path / "profiler_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["rows"][0]["model"] = "off-matrix-granite"
    metrics_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    analysis = analyze(payload)
    (tmp_path / "profiler_analysis_gate.json").write_text(
        json.dumps(analysis, indent=2) + "\n",
        encoding="utf-8",
    )

    result = check_run_artifacts(tmp_path, require_native_artifacts=False)

    assert result["status"] == "FAIL"
    assert any("not allowed by native_control_matrix" in error for error in result["errors"])


def test_rejects_metric_control_segment_outside_native_control_matrix(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    metrics_path = tmp_path / "profiler_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["rows"][0]["control_model_or_segment"] = "unregistered_boundary_window"
    metrics_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    analysis = analyze(payload)
    (tmp_path / "profiler_analysis_gate.json").write_text(
        json.dumps(analysis, indent=2) + "\n",
        encoding="utf-8",
    )

    result = check_run_artifacts(tmp_path, require_native_artifacts=False)

    assert result["status"] == "FAIL"
    assert any("control_model_or_segment is not allowed" in error for error in result["errors"])


def test_rejects_metric_boundary_direction_outside_native_control_matrix(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    metrics_path = tmp_path / "profiler_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["rows"][0]["boundary_direction"] = "ssm_to_attention_only"
    metrics_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    analysis = analyze(payload)
    (tmp_path / "profiler_analysis_gate.json").write_text(
        json.dumps(analysis, indent=2) + "\n",
        encoding="utf-8",
    )

    result = check_run_artifacts(tmp_path, require_native_artifacts=False)

    assert result["status"] == "FAIL"
    assert any("boundary_direction is not allowed" in error for error in result["errors"])


def test_rejects_metric_request_shape_outside_native_control_matrix(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    metrics_path = tmp_path / "profiler_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["rows"][0]["batch_shape"]["decode_tokens"] = 128
    metrics_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    analysis = analyze(payload)
    (tmp_path / "profiler_analysis_gate.json").write_text(
        json.dumps(analysis, indent=2) + "\n",
        encoding="utf-8",
    )

    result = check_run_artifacts(tmp_path, require_native_artifacts=False)

    assert result["status"] == "FAIL"
    assert any("request_shape does not match" in error for error in result["errors"])


def test_rejects_cuda_graph_state_outside_native_control_matrix(tmp_path: Path) -> None:
    _write_complete_run(tmp_path)
    metrics_path = tmp_path / "profiler_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["rows"][0]["cuda_graph_enabled"] = False
    metrics_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    analysis = analyze(payload)
    (tmp_path / "profiler_analysis_gate.json").write_text(
        json.dumps(analysis, indent=2) + "\n",
        encoding="utf-8",
    )

    result = check_run_artifacts(tmp_path, require_full_matrix=True, require_native_artifacts=False)

    assert result["status"] == "FAIL"
    assert any("request_shape does not match" in error for error in result["errors"])


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
