from __future__ import annotations

import csv
import json
from pathlib import Path

from experimental.sinkaware.phase2.check_native_gpu_packet import (
    REQUIRED_ROW_IDS,
    check_native_gpu_packet,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _complete_packet(run_dir: Path, latency_runs: int = 3) -> None:
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "gpu": "NVIDIA H100",
                "driver": "550.54",
                "cuda": "12.4",
                "pytorch": "2.6.0+cu124",
                "triton": "3.2.0",
                "model": "distilgpt2",
                "dtype": "bfloat16",
                "sequence_shapes": [{"batch_size": 1, "sequence_length": 96}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    quality_rows = [
        {
            "row": row_id,
            "model": "distilgpt2",
            "sequence_length": 96,
            "batch_size": 1,
            "layer": "mean",
            "output_rel_l2": 0.01,
            "sink_mass_mae": 0.002,
            "attention_l1": 0.003,
        }
        for row_id in REQUIRED_ROW_IDS
    ]
    quality_by_head_rows = [
        {
            "row": row_id,
            "model": "distilgpt2",
            "sequence_length": 96,
            "batch_size": 1,
            "layer": 0,
            "head": 0,
            "output_rel_l2": 0.01,
            "sink_mass_mae": 0.002,
            "attention_l1": 0.003,
        }
        for row_id in REQUIRED_ROW_IDS
    ]
    latency_rows = [
        {
            "row": row_id,
            "model": "distilgpt2",
            "sequence_length": 96,
            "batch_size": 1,
            "run_id": run_id,
            "latency_ms": 1.0 + run_id * 0.01,
        }
        for row_id in REQUIRED_ROW_IDS
        for run_id in range(latency_runs)
    ]
    ncu_rows = [
        {
            "row": row_id,
            "model": "distilgpt2",
            "sequence_length": 96,
            "batch_size": 1,
            "dram_bytes": 1000000,
            "l2_bytes": 500000,
            "achieved_occupancy": 0.5,
            "registers_per_thread": 64,
        }
        for row_id in REQUIRED_ROW_IDS
    ]
    _write_csv(run_dir / "quality_drift.csv", quality_rows)
    _write_csv(run_dir / "quality_drift_by_head.csv", quality_by_head_rows)
    _write_csv(run_dir / "latency.csv", latency_rows)
    _write_csv(run_dir / "ncu_summary.csv", ncu_rows)
    (run_dir / "decision.md").write_text(
        "# SinkAware Native GPU Decision\n\n"
        "Decision: PROMOTE only if repeated native GPU evidence clears the "
        "rank-2 quality threshold and speed/HBM threshold. This packet records "
        "the measured quality drift and native memory evidence.\n",
        encoding="utf-8",
    )


def test_complete_native_gpu_packet_passes(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    _complete_packet(packet)

    result = check_native_gpu_packet(packet)

    assert result["status"] == "PASS"
    assert result["csv_summaries"]["latency.csv"]["row_run_counts"] == {
        row_id: 3 for row_id in REQUIRED_ROW_IDS
    }


def test_rejects_missing_required_artifact(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    _complete_packet(packet)
    (packet / "ncu_summary.csv").unlink()

    result = check_native_gpu_packet(packet)

    assert result["status"] == "FAIL"
    assert any("ncu_summary.csv" in error for error in result["errors"])


def test_rejects_partial_row_coverage(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    _complete_packet(packet)
    rows = [
        row
        for row in csv.DictReader((packet / "quality_drift.csv").open(encoding="utf-8"))
        if row["row"] != "rank2_sink_logit_predictor"
    ]
    _write_csv(packet / "quality_drift.csv", rows)

    result = check_native_gpu_packet(packet)

    assert result["status"] == "FAIL"
    assert any("quality_drift.csv missing required row" in error for error in result["errors"])


def test_requires_repeated_latency_runs_per_row(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    _complete_packet(packet, latency_runs=2)

    result = check_native_gpu_packet(packet)

    assert result["status"] == "FAIL"
    assert any("latency.csv row exact_attention has 2 distinct" in error for error in result["errors"])


def test_requires_latency_repeats_for_same_shape(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    _complete_packet(packet)
    rows = list(csv.DictReader((packet / "latency.csv").open(encoding="utf-8")))
    for row in rows:
        row["sequence_length"] = str(int(row["sequence_length"]) + int(row["run_id"]))
    _write_csv(packet / "latency.csv", rows)

    result = check_native_gpu_packet(packet)

    assert result["status"] == "FAIL"
    assert any(
        "model/sequence_length/batch_size group" in error for error in result["errors"]
    )


def test_rejects_any_under_repeated_latency_shape_even_with_one_complete_shape(
    tmp_path: Path,
) -> None:
    packet = tmp_path / "packet"
    _complete_packet(packet)
    metadata = json.loads((packet / "metadata.json").read_text(encoding="utf-8"))
    metadata["sequence_shapes"].append({"batch_size": 1, "sequence_length": 128})
    (packet / "metadata.json").write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    for filename in ["quality_drift.csv", "quality_drift_by_head.csv", "ncu_summary.csv"]:
        rows = list(csv.DictReader((packet / filename).open(encoding="utf-8")))
        extra = dict(rows[0])
        extra["sequence_length"] = "128"
        rows.append(extra)
        _write_csv(packet / filename, rows)

    latency_rows = list(csv.DictReader((packet / "latency.csv").open(encoding="utf-8")))
    extra_latency = dict(latency_rows[0])
    extra_latency["sequence_length"] = "128"
    extra_latency["run_id"] = "extra-only"
    latency_rows.append(extra_latency)
    _write_csv(packet / "latency.csv", latency_rows)

    result = check_native_gpu_packet(packet)

    assert result["status"] == "FAIL"
    assert any(
        "row exact_attention model=distilgpt2 sequence_length=128 batch_size=1 has 1 distinct"
        in error
        for error in result["errors"]
    )


def test_requires_same_shapes_across_quality_latency_and_ncu(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    _complete_packet(packet)
    quality_rows = list(csv.DictReader((packet / "quality_drift.csv").open(encoding="utf-8")))
    for row in quality_rows:
        row["sequence_length"] = "64"
    _write_csv(packet / "quality_drift.csv", quality_rows)
    ncu_rows = list(csv.DictReader((packet / "ncu_summary.csv").open(encoding="utf-8")))
    for row in ncu_rows:
        row["sequence_length"] = "128"
    _write_csv(packet / "ncu_summary.csv", ncu_rows)

    result = check_native_gpu_packet(packet)

    assert result["status"] == "FAIL"
    assert any(
        "row/model/sequence_length/batch_size groups do not match" in error
        for error in result["errors"]
    )


def test_requires_metadata_sequence_shapes_to_match_measured_csv_shapes(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    _complete_packet(packet)
    metadata = json.loads((packet / "metadata.json").read_text(encoding="utf-8"))
    metadata["sequence_shapes"] = [{"batch_size": 99, "sequence_length": 999}]
    (packet / "metadata.json").write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    result = check_native_gpu_packet(packet)

    assert result["status"] == "FAIL"
    assert any(
        "metadata.json sequence_shapes do not match" in error
        for error in result["errors"]
    )


def test_requires_metadata_model_to_match_measured_csv_models(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    _complete_packet(packet)
    metadata = json.loads((packet / "metadata.json").read_text(encoding="utf-8"))
    metadata["model"] = "wrong-model"
    (packet / "metadata.json").write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    result = check_native_gpu_packet(packet)

    assert result["status"] == "FAIL"
    assert any("metadata.json model does not match" in error for error in result["errors"])


def test_rejects_invalid_metadata_native_scope(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    _complete_packet(packet)
    metadata = json.loads((packet / "metadata.json").read_text(encoding="utf-8"))
    metadata["gpu"] = "Mac MPS"
    metadata["cuda"] = "unavailable"
    del metadata["sequence_shapes"]
    (packet / "metadata.json").write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    result = check_native_gpu_packet(packet)

    assert result["status"] == "FAIL"
    assert any("metadata.json gpu" in error for error in result["errors"])
    assert any("metadata.json cuda" in error for error in result["errors"])
    assert any("sequence_shapes" in error for error in result["errors"])


def test_rejects_not_available_cuda_metadata(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    _complete_packet(packet)
    metadata = json.loads((packet / "metadata.json").read_text(encoding="utf-8"))
    metadata["cuda"] = "not available"
    (packet / "metadata.json").write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    result = check_native_gpu_packet(packet)

    assert result["status"] == "FAIL"
    assert any("metadata.json cuda" in error for error in result["errors"])


def test_rejects_non_cuda_backend_metadata(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    _complete_packet(packet)
    metadata = json.loads((packet / "metadata.json").read_text(encoding="utf-8"))
    metadata["cuda"] = "hip/rocm 6.0"
    (packet / "metadata.json").write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    result = check_native_gpu_packet(packet)

    assert result["status"] == "FAIL"
    assert any("metadata.json cuda" in error for error in result["errors"])


def test_rejects_non_nvidia_gpu_metadata(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    _complete_packet(packet)
    metadata = json.loads((packet / "metadata.json").read_text(encoding="utf-8"))
    metadata["gpu"] = "AMD MI300X"
    (packet / "metadata.json").write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    result = check_native_gpu_packet(packet)

    assert result["status"] == "FAIL"
    assert any("metadata.json gpu must describe an NVIDIA GPU" in error for error in result["errors"])


def test_rejects_placeholder_decision_packet(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    _complete_packet(packet)
    (packet / "decision.md").write_text("TODO_NATIVE_SINKAWARE_FILL\n", encoding="utf-8")

    result = check_native_gpu_packet(packet)

    assert result["status"] == "FAIL"
    assert any("placeholder" in error for error in result["errors"])
    assert any("promote or kill" in error for error in result["errors"])


def test_rejects_non_numeric_metric_cells(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    _complete_packet(packet)
    rows = list(csv.DictReader((packet / "latency.csv").open(encoding="utf-8")))
    rows[0]["latency_ms"] = "fast"
    _write_csv(packet / "latency.csv", rows)

    result = check_native_gpu_packet(packet)

    assert result["status"] == "FAIL"
    assert any("non-numeric latency_ms" in error for error in result["errors"])
