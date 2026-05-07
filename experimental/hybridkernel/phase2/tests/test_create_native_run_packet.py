from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from experimental.hybridkernel.phase2.analyze_profiler_metrics import analyze, _write_markdown
from experimental.hybridkernel.phase2.check_profiler_run_artifacts import (
    READOUT_MARKERS,
    SKELETON_TODO_MARKER,
    check_run_artifacts,
    _validate_cross_family_replacement,
)
from experimental.hybridkernel.phase2.create_native_run_packet import create_run_packet


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_profiler_artifact(path: Path) -> None:
    path.write_bytes((b"\x93NSIGHT\x00\xffnative-binary-export\x00" * 128)[:4096])


def _write_client_replay_log(run_dir: Path, *, model: str, run_id: str) -> None:
    (run_dir / f"logs/client_replay_{run_id}.log").write_text(
        json.dumps(
            {
                "model": model,
                "run_id": run_id,
                "dry_run": False,
                "token_counts_required": True,
                "token_count_source": "test_tokenizer",
                "requests": [
                    {
                        "status": "ok",
                        "batch_size": 1,
                        "prompt_token_counts": [128],
                        "prompt_token_count_total": 128,
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


def _write_snapshot_manifest(run_dir: Path, name: str) -> tuple[str, str]:
    path = run_dir / f"metadata/{name}_snapshot_manifest.json"
    path.write_text(
        json.dumps(
            {
                "files": [
                    {"path": "config.json", "sha256": "sha256:" + ("1" * 64)},
                    {"path": "tokenizer.json", "sha256": "sha256:" + ("2" * 64)},
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return f"metadata/{name}_snapshot_manifest.json", _sha256(path)


def _write_reduction_manifest(run_dir: Path, payload: dict[str, object]) -> None:
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


def test_create_native_run_packet_writes_required_skeleton(tmp_path: Path) -> None:
    run_dir = create_run_packet(
        output_dir=tmp_path / "packet",
        model="ibm-granite/granite-4.0-h-tiny",
    )

    expected_files = [
        "README.md",
        "metadata/environment.txt",
        "metadata/environment.json",
        "metadata/profile_scope.json",
        "metadata/architecture_map.json",
        "metadata/model_provenance.json",
        "metadata/native_control_matrix.json",
        "metadata/reduction_input_manifest.json",
        "metadata/reduction_worksheet.tsv",
        "logs/README.md",
        "nsys/README.md",
        "ncu/README.md",
        "readout.md",
        "profiler_metrics.json",
        "profiler_analysis_gate.json",
        "profiler_analysis_gate.md",
    ]
    for relative in expected_files:
        assert (run_dir / relative).is_file(), relative
    assert "--require-full-matrix" in (run_dir / "README.md").read_text()

    profile_scope = json.loads((run_dir / "metadata/profile_scope.json").read_text())
    assert profile_scope["profiled_process"] == "vllm_server"
    assert profile_scope["nsys_profiled_process"] == "vllm_server"
    assert profile_scope["ncu_profiled_process"] == "vllm_server"
    assert "vllm.entrypoints.openai.api_server" in profile_scope["vllm_command"]
    assert {scope["model"] for scope in profile_scope["model_scopes"]} == {
        "ibm-granite/granite-4.0-h-tiny",
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
    }
    assert {
        tuple(scope["row_roles"]) for scope in profile_scope["model_scopes"]
    } == {("primary_hybrid", "same_family_control"), ("cross_family_falsification",)}

    metrics = json.loads((run_dir / "profiler_metrics.json").read_text())
    reduction_manifest = json.loads(
        (run_dir / "metadata/reduction_input_manifest.json").read_text()
    )
    assert len(metrics["rows"]) == 9
    assert len(reduction_manifest["rows"]) == 9
    assert reduction_manifest["manifest_version"] == "hybridkernel_reduction_inputs_v1"
    assert {
        (row["run_id"], row["row_role"], row["model"])
        for row in reduction_manifest["rows"]
    } == {
        (row["run_id"], row["row_role"], row["model"])
        for row in metrics["rows"]
    }
    assert [row["run_id"] for row in metrics["rows"]] == [
        "primary-repeat-0",
        "primary-repeat-1",
        "primary-repeat-2",
        "same-family-control-repeat-0",
        "same-family-control-repeat-1",
        "same-family-control-repeat-2",
        "cross-family-falsification-repeat-0",
        "cross-family-falsification-repeat-1",
        "cross-family-falsification-repeat-2",
    ]
    assert [row["row_role"] for row in metrics["rows"]].count("primary_hybrid") == 3
    assert [row["row_role"] for row in metrics["rows"]].count("same_family_control") == 3
    assert [row["row_role"] for row in metrics["rows"]].count("cross_family_falsification") == 3
    assert {row["boundary_direction"] for row in metrics["rows"] if row["row_role"] == "primary_hybrid"} == {
        "mixed_attention_ssm"
    }
    assert {row["boundary_direction"] for row in metrics["rows"] if row["row_role"] == "same_family_control"} == {
        "non_boundary_same_family"
    }
    assert {row["boundary_direction"] for row in metrics["rows"] if row["row_role"] == "cross_family_falsification"} == {
        "linear_attention_gated_delta_boundary"
    }
    assert {tuple(row["batch_shape"].values()) for row in metrics["rows"]} == {
        (1, 128, 64, 16)
    }
    assert {row["model"] for row in metrics["rows"] if row["row_role"] == "same_family_control"} == {
        "ibm-granite/granite-4.0-h-tiny"
    }
    assert {row["model"] for row in metrics["rows"] if row["row_role"] == "cross_family_falsification"} == {
        "Qwen/Qwen3-Next-80B-A3B-Instruct"
    }
    for row in metrics["rows"]:
        assert "ncu_launch_selection" in row
        assert row["ncu_launch_selection"]["kernel_regex"] is None
        assert "control_window_ids" in row
    architecture_map = json.loads((run_dir / "metadata/architecture_map.json").read_text())
    assert "ibm-granite/granite-4.0-h-tiny" in {row["model"] for row in architecture_map}
    control_matrix = json.loads((run_dir / "metadata/native_control_matrix.json").read_text())
    assert control_matrix["decision"] == "CONTROL_MATRIX_READY_NOT_NATIVE_EVIDENCE"
    assert {row["row_role"] for row in control_matrix["rows"]} == {
        "primary_hybrid",
        "same_family_control",
        "cross_family_falsification",
    }
    model_provenance = json.loads((run_dir / "metadata/model_provenance.json").read_text())
    assert model_provenance["provenance_version"] == "hybridkernel_model_provenance_v1"
    assert {row["model_id"] for row in model_provenance["models"]} == {
        "ibm-granite/granite-4.0-h-tiny",
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
    }
    assert all("snapshot_manifest_path" in row for row in model_provenance["models"])
    environment = json.loads((run_dir / "metadata/environment.json").read_text())
    assert environment["environment_version"] == "hybridkernel_environment_v1"
    assert SKELETON_TODO_MARKER in (run_dir / "metadata/reduction_worksheet.tsv").read_text()


def test_skeleton_is_not_mistaken_for_complete_native_evidence(tmp_path: Path) -> None:
    run_dir = create_run_packet(output_dir=tmp_path / "packet")

    result = check_run_artifacts(run_dir)

    assert result["status"] == "FAIL"
    assert any("Nsight Systems" in error for error in result["errors"])
    assert any("Nsight Compute" in error for error in result["errors"])
    assert any("no valid native rows" in error for error in result["errors"])
    assert any(SKELETON_TODO_MARKER in (run_dir / path).read_text() for path in [
        "metadata/environment.txt",
        "metadata/environment.json",
        "metadata/model_provenance.json",
        "metadata/reduction_input_manifest.json",
        "metadata/reduction_worksheet.tsv",
        "readout.md",
        "profiler_metrics.json",
    ])
    assert any("skeleton TODO markers" in error for error in result["errors"])


def test_create_native_run_packet_refuses_non_empty_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "packet"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("operator data\n", encoding="utf-8")

    try:
        create_run_packet(output_dir=output_dir)
    except FileExistsError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("expected non-empty run directory to be rejected")


def test_create_native_run_packet_refuses_too_few_runs(tmp_path: Path) -> None:
    try:
        create_run_packet(output_dir=tmp_path / "packet", min_runs=2)
    except ValueError as exc:
        assert "at least 3" in str(exc)
    else:
        raise AssertionError("expected min_runs below review gate to be rejected")


def test_generated_packet_can_be_filled_into_complete_promotable_shape(tmp_path: Path) -> None:
    run_dir = create_run_packet(
        output_dir=tmp_path / "packet",
        model="ibm-granite/granite-4.0-h-tiny",
    )
    model = "ibm-granite/granite-4.0-h-tiny"
    same_family = model
    cross_family = "Qwen/Qwen3-Next-80B-A3B-Instruct"
    (run_dir / "metadata/environment.txt").write_text(
        "nvidia-smi\nnsys version\nncu version\npython -VV\n", encoding="utf-8"
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
    qwen_manifest_path, qwen_manifest_sha = _write_snapshot_manifest(run_dir, "qwen")
    (run_dir / "metadata/model_provenance.json").write_text(
        json.dumps(
            {
                "provenance_version": "hybridkernel_model_provenance_v1",
                "models": [
                    {
                        "model_id": model,
                        "served_model_id": model,
                        "model_revision": "test-granite-commit",
                        "tokenizer_revision": "test-granite-tokenizer-commit",
                        "cache_source": "synthetic test fixture",
                        "snapshot_manifest_path": granite_manifest_path,
                        "snapshot_manifest_sha256": granite_manifest_sha,
                        "local_files_only": True,
                        "trust_remote_code": True,
                    },
                    {
                        "model_id": cross_family,
                        "served_model_id": cross_family,
                        "model_revision": "test-qwen-commit",
                        "tokenizer_revision": "test-qwen-tokenizer-commit",
                        "cache_source": "synthetic test fixture",
                        "snapshot_manifest_path": qwen_manifest_path,
                        "snapshot_manifest_sha256": qwen_manifest_sha,
                        "local_files_only": True,
                        "trust_remote_code": True,
                    },
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "metadata/reduction_worksheet.tsv").write_text(
        "run_id\trow_role\tmodel\tnotes\nsynthetic\tprimary_hybrid\t"
        f"{model}\tfilled synthetic worksheet\n",
        encoding="utf-8",
    )
    (run_dir / "logs/nsys_server_b1.log").write_text(
        "nsys vllm server cuda profiler log\n", encoding="utf-8"
    )
    readout_rows = "\n".join(f"| {marker} | synthetic evidence | no |" for marker in READOUT_MARKERS)
    (run_dir / "readout.md").write_text(
        "| Question | Evidence | Decision |\n|---|---|---|\n" + readout_rows + "\n",
        encoding="utf-8",
    )

    rows = []
    specs = [
        (
            "primary",
            model,
            "primary_hybrid",
            "same_family_matched_segment",
            "granite_hybrid_attention_ssm_boundary_windows",
            "mixed_attention_ssm",
            8.0,
            2.0,
        ),
        (
            "same",
            same_family,
            "same_family_control",
            "same_model_non_boundary_segment_control",
            "granite_same_model_non_boundary_ssm_to_ssm_or_attention_internal_windows",
            "non_boundary_same_family",
            2.0,
            2.0,
        ),
        (
            "cross",
            cross_family,
            "cross_family_falsification",
            "cross_family_hybrid_control",
            "qwen3_next_hybrid_boundary_windows",
            "linear_attention_gated_delta_boundary",
            2.0,
            2.0,
        ),
    ]
    for spec_index, (
        label,
        row_model,
        role,
        family,
        control_segment,
        boundary_direction,
        boundary_ms,
        matched_ms,
    ) in enumerate(specs):
        for repeat in range(3):
            row_id = f"{label}-{repeat}"
            _write_client_replay_log(run_dir, model=row_model, run_id=row_id)
            nsys_path = run_dir / f"nsys/{row_id}.nsys-rep"
            ncu_path = run_dir / f"ncu/{row_id}.ncu-rep"
            _write_profiler_artifact(nsys_path)
            _write_profiler_artifact(ncu_path)
            rows.append(
                {
                    "model": row_model,
                    "run_id": row_id,
                    "total_step_ms": 100.0,
                    "attention_ssm_boundary_ms": boundary_ms,
                    "matched_non_boundary_ms": matched_ms,
                    "recoverable_fraction": 0.60,
                    "dtype": "bfloat16",
                    "profiled_process": "vllm_server",
                    "trace_scope": "server-side CUDA kernels, not client-only HTTP replay",
                    "cuda_graph_enabled": True,
                    "batch_shape": {
                        "batch_size": 1,
                        "prefill_tokens": 128,
                        "decode_tokens": 64,
                        "requests": 16,
                    },
                    "control_model_or_segment": control_segment,
                    "row_role": role,
                    "control_family": family,
                    "boundary_direction": boundary_direction,
                    "nsys_artifact": f"nsys/{row_id}.nsys-rep",
                    "nsys_artifact_sha256": _sha256(nsys_path),
                    "ncu_artifact": f"ncu/{row_id}.ncu-rep",
                    "ncu_artifact_sha256": _sha256(ncu_path),
                    "kernel_names": [f"synthetic_{label}_kernel"],
                    "boundary_indices": [spec_index] if role != "same_family_control" else [],
                    "control_window_ids": (
                        [f"granite-non-boundary-window-{repeat}"]
                        if role == "same_family_control"
                        else []
                    ),
                    "time_window_ms": {
                        "start": float(spec_index * 10 + repeat),
                        "end": float(spec_index * 10 + repeat) + 1.0,
                    },
                    "ncu_launch_selection": {
                        "kernel_regex": f"synthetic_{label}_kernel",
                        "launch_skip": repeat,
                        "launch_count": 1,
                        "source_nsys_artifact": f"nsys/{row_id}.nsys-rep",
                        "source_time_window_ms": {
                            "start": float(spec_index * 10 + repeat),
                            "end": float(spec_index * 10 + repeat) + 1.0,
                        },
                        "derivation_notes": "Selected from the matching synthetic Nsight Systems window.",
                    },
                    "recoverable_fraction_basis": "Synthetic integration test uses fixed 60% recovery for parser plumbing.",
                    "reduction_command": "python -m pytest experimental/hybridkernel/phase2/tests/test_create_native_run_packet.py",
                    "reduction_notes": "Synthetic generated-packet integration row.",
                }
            )
    metrics_payload = {"description": "Synthetic generated-packet integration test.", "rows": rows}
    analysis = analyze(metrics_payload)
    (run_dir / "profiler_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_reduction_manifest(run_dir, metrics_payload)
    (run_dir / "profiler_analysis_gate.json").write_text(
        json.dumps(analysis, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown(analysis, run_dir / "profiler_analysis_gate.md")

    result = check_run_artifacts(run_dir)

    assert result["status"] == "PASS"
    assert result["metrics_status"].startswith("PROMOTE")
    assert result["metrics_rows"] == 9


def test_generated_packet_rejects_replacement_without_preregistered_metadata(tmp_path: Path) -> None:
    run_dir = create_run_packet(
        output_dir=tmp_path / "packet",
        model="ibm-granite/granite-4.0-h-tiny",
    )
    errors: list[str] = []

    _validate_cross_family_replacement(
        run_dir=run_dir,
        control_specs={},
        raw_rows=[
            {
                "row_role": "cross_family_falsification",
                "model": "replacement/hybrid-small",
            }
        ],
        errors=errors,
    )

    assert any("cross-family replacement requires" in error for error in errors)


def test_checker_cli_rejects_schema_escape_with_full_matrix(tmp_path: Path) -> None:
    run_dir = create_run_packet(output_dir=tmp_path / "packet")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experimental.hybridkernel.phase2.check_profiler_run_artifacts",
            "--run-dir",
            str(run_dir),
            "--allow-missing-native-artifacts",
            "--require-full-matrix",
        ],
        cwd=Path(__file__).resolve().parents[4],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "schema-test-only" in result.stderr
