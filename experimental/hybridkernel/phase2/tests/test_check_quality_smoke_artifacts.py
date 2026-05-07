from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experimental.hybridkernel.phase2.check_quality_smoke_artifacts import check_quality_smoke


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_packet(tmp_path: Path) -> Path:
    prompts = tmp_path / "prompts.jsonl"
    stock = tmp_path / "stock_outputs.jsonl"
    prototype = tmp_path / "prototype_outputs.jsonl"
    prompts.write_text("\n".join(f'{{"id": {idx}}}' for idx in range(12)) + "\n", encoding="utf-8")
    stock.write_text("\n".join(f'{{"id": {idx}, "answer": "A"}}' for idx in range(12)) + "\n", encoding="utf-8")
    prototype.write_text(
        "\n".join(f'{{"id": {idx}, "answer": "A"}}' for idx in range(12)) + "\n",
        encoding="utf-8",
    )
    payload = {
        "quality_smoke_version": "hybridkernel_quality_smoke_v1",
        "prompt_file": "prompts.jsonl",
        "prompt_file_sha256": _sha256(prompts),
        "rows": [
            {
                "model": "ibm-granite/granite-4.0-h-tiny",
                "stock_mode": "stock_vllm",
                "prototype_mode": "hybridkernel_prototype",
                "prompt_count": 12,
                "normalized_answer_mismatch_count": 0,
                "accuracy_delta_prototype_minus_stock": 0.0,
                "mean_output_length_drift_fraction": 0.02,
                "stock_outputs_path": "stock_outputs.jsonl",
                "stock_outputs_sha256": _sha256(stock),
                "prototype_outputs_path": "prototype_outputs.jsonl",
                "prototype_outputs_sha256": _sha256(prototype),
            }
        ],
    }
    path = tmp_path / "quality_smoke.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def test_accepts_complete_quality_smoke_packet(tmp_path: Path) -> None:
    path = _write_packet(tmp_path)

    result = check_quality_smoke(path, repo_root=tmp_path)

    assert result["status"] == "PASS"
    assert result["errors"] == []


def test_rejects_answer_or_length_regressions(tmp_path: Path) -> None:
    path = _write_packet(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rows"][0]["normalized_answer_mismatch_count"] = 1
    payload["rows"][0]["accuracy_delta_prototype_minus_stock"] = -0.02
    payload["rows"][0]["mean_output_length_drift_fraction"] = 0.15
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = check_quality_smoke(path, repo_root=tmp_path)

    assert result["status"] == "FAIL"
    assert any("normalized_answer_mismatch_count" in error for error in result["errors"])
    assert any("accuracy drop" in error for error in result["errors"])
    assert any("output length drift" in error for error in result["errors"])
