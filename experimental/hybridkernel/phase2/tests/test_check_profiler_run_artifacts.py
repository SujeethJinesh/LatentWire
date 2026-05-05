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
