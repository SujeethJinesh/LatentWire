from __future__ import annotations

import json
import sys

from scripts import build_source_private_latest_model_matrix as matrix


def test_latest_matrix_contains_qwen35_and_moe_candidates() -> None:
    rows = matrix._model_matrix()
    model_ids = {row.model for row in rows}

    assert "Qwen/Qwen3.5-0.8B" in model_ids
    assert "Qwen/Qwen3.5-2B" in model_ids
    assert "Qwen/Qwen3.6-35B-A3B" in model_ids
    assert "Qwen/Qwen3.6-35B-A3B-FP8" in model_ids
    assert any(row.architecture == "sparse MoE" for row in rows)


def test_latest_matrix_commands_are_existing_runner_commands() -> None:
    row = next(row for row in matrix._model_matrix() if row.model == "Qwen/Qwen3.5-0.8B")
    command = row.command("bench.jsonl", "out")

    assert "scripts/run_source_private_hidden_repair_packet_llm.py" in command
    assert "--model Qwen/Qwen3.5-0.8B" in command
    assert "--prompt-mode trace_no_hint" in command


def test_latest_matrix_main_writes_manifest(tmp_path, monkeypatch) -> None:
    out = tmp_path / "matrix"
    monkeypatch.setattr(sys, "argv", [
        "build_source_private_latest_model_matrix.py",
        "--benchmark-jsonl",
        "bench.jsonl",
        "--output-dir",
        str(out),
    ])
    matrix.main()

    payload = json.loads((out / "latest_model_matrix.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))

    assert payload["models"]
    assert manifest["summary"]["moe_count"] >= 2
    assert "qwen3_5" in payload["compatibility_note"]
