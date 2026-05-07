from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experimental.hybridkernel.phase2.analyze_profiler_metrics import analyze, _write_markdown
from experimental.hybridkernel.phase2.check_profiler_run_artifacts import (
    READOUT_MARKERS,
    SKELETON_TODO_MARKER,
    check_run_artifacts,
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


def test_create_native_run_packet_writes_required_skeleton(tmp_path: Path) -> None:
    run_dir = create_run_packet(
        output_dir=tmp_path / "packet",
        model="ibm-granite/granite-4.0-h-tiny",
    )

    expected_files = [
        "README.md",
        "metadata/environment.txt",
        "metadata/profile_scope.json",
        "metadata/architecture_map.json",
        "metadata/native_control_matrix.json",
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

    profile_scope = json.loads((run_dir / "metadata/profile_scope.json").read_text())
    assert profile_scope["profiled_process"] == "vllm_server"
    assert profile_scope["nsys_profiled_process"] == "vllm_server"
    assert profile_scope["ncu_profiled_process"] == "vllm_server"
    assert "vllm.entrypoints.openai.api_server" in profile_scope["vllm_command"]
    assert {scope["model"] for scope in profile_scope["model_scopes"]} == {
        "ibm-granite/granite-4.0-h-tiny",
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
    }

    metrics = json.loads((run_dir / "profiler_metrics.json").read_text())
    assert len(metrics["rows"]) == 9
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
    assert {row["model"] for row in metrics["rows"] if row["row_role"] == "same_family_control"} == {
        "ibm-granite/granite-4.0-h-tiny"
    }
    assert {row["model"] for row in metrics["rows"] if row["row_role"] == "cross_family_falsification"} == {
        "Qwen/Qwen3-Next-80B-A3B-Instruct"
    }
    architecture_map = json.loads((run_dir / "metadata/architecture_map.json").read_text())
    assert "ibm-granite/granite-4.0-h-tiny" in {row["model"] for row in architecture_map}
    control_matrix = json.loads((run_dir / "metadata/native_control_matrix.json").read_text())
    assert control_matrix["decision"] == "CONTROL_MATRIX_READY_NOT_NATIVE_EVIDENCE"
    assert {row["row_role"] for row in control_matrix["rows"]} == {
        "primary_hybrid",
        "same_family_control",
        "cross_family_falsification",
    }


def test_skeleton_is_not_mistaken_for_complete_native_evidence(tmp_path: Path) -> None:
    run_dir = create_run_packet(output_dir=tmp_path / "packet")

    result = check_run_artifacts(run_dir)

    assert result["status"] == "FAIL"
    assert any("Nsight Systems" in error for error in result["errors"])
    assert any("Nsight Compute" in error for error in result["errors"])
    assert any("no valid native rows" in error for error in result["errors"])
    assert any(SKELETON_TODO_MARKER in (run_dir / path).read_text() for path in [
        "metadata/environment.txt",
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
    (run_dir / "logs/nsys_server_b1.log").write_text(
        "nsys vllm server cuda profiler log\n", encoding="utf-8"
    )
    for log_index, replay_model in enumerate((model, same_family, cross_family)):
        (run_dir / f"logs/client_replay_b1_{log_index}.log").write_text(
            json.dumps(
                {
                    "model": replay_model,
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
    readout_rows = "\n".join(f"| {marker} | synthetic evidence | no |" for marker in READOUT_MARKERS)
    (run_dir / "readout.md").write_text(
        "| Question | Evidence | Decision |\n|---|---|---|\n" + readout_rows + "\n",
        encoding="utf-8",
    )

    rows = []
    specs = [
        ("primary", model, "primary_hybrid", "same_family_matched_segment", 8.0, 2.0),
        ("same", same_family, "same_family_control", "same_model_non_boundary_segment_control", 2.0, 2.0),
        ("cross", cross_family, "cross_family_falsification", "cross_family_hybrid_control", 2.0, 2.0),
    ]
    for spec_index, (label, row_model, role, family, boundary_ms, matched_ms) in enumerate(specs):
        for repeat in range(3):
            row_id = f"{label}-{repeat}"
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
                    "control_model_or_segment": family,
                    "row_role": role,
                    "control_family": family,
                    "boundary_direction": "mixed_attention_ssm",
                    "nsys_artifact": f"nsys/{row_id}.nsys-rep",
                    "nsys_artifact_sha256": _sha256(nsys_path),
                    "ncu_artifact": f"ncu/{row_id}.ncu-rep",
                    "ncu_artifact_sha256": _sha256(ncu_path),
                    "kernel_names": [f"synthetic_{label}_kernel"],
                    "boundary_indices": [spec_index] if role == "primary_hybrid" else [],
                    "time_window_ms": {
                        "start": float(spec_index * 10 + repeat),
                        "end": float(spec_index * 10 + repeat) + 1.0,
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
    (run_dir / "profiler_analysis_gate.json").write_text(
        json.dumps(analysis, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown(analysis, run_dir / "profiler_analysis_gate.md")

    result = check_run_artifacts(run_dir)

    assert result["status"] == "PASS"
    assert result["metrics_status"].startswith("PROMOTE")
    assert result["metrics_rows"] == 9
