from __future__ import annotations

import json
from pathlib import Path

from experimental.hybridkernel.phase2.check_profiler_run_artifacts import (
    SKELETON_TODO_MARKER,
    check_run_artifacts,
)
from experimental.hybridkernel.phase2.create_native_run_packet import create_run_packet


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

    metrics = json.loads((run_dir / "profiler_metrics.json").read_text())
    assert len(metrics["rows"]) == 3
    assert [row["run_id"] for row in metrics["rows"]] == [
        "repeat-0",
        "repeat-1",
        "repeat-2",
    ]


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
