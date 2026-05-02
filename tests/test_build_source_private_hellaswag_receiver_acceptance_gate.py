from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import build_source_private_hellaswag_receiver_acceptance_gate as gate


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _score_row(prediction: int) -> list[float]:
    return [1.0 if index == prediction else 0.0 for index in range(4)]


def _fixture(tmp_path, *, mismatch: bool = False):
    answers = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    packet = [0, 1, 2, 0, 0, 0, 2, 0, 1, 1, 1, 3]
    qwen_mean = [0, 1, 2, 3, 1, 1, 3, 3, 0, 2, 2, 1]
    qwen_hybrid = [0, 1, 2, 3, 1, 1, 0, 3, 0, 2, 2, 0]
    tiny_rows = [
        {
            "row_id": str(index),
            "answer_index": answer,
            "selected_prediction": packet[index],
            "selected_margin": 0.1 + 0.03 * index,
            "wrong_example_hidden_prediction": (packet[index] + 1) % 4,
            "candidate_roll_hidden_prediction": (packet[index] + 2) % 4,
            "zero_hidden_prediction": 0,
            "source_label_prediction": 0,
        }
        for index, answer in enumerate(answers)
    ]
    qwen_rows = [
        {
            "row_id": str(index + (1 if mismatch else 0)),
            "answer_index": answer,
            "mean_zscore_prediction": qwen_mean[index],
            "hybrid_vote_on_score_agreement_prediction": qwen_hybrid[index],
        }
        for index, answer in enumerate(answers)
    ]
    tiny_packet_jsonl = tmp_path / "tiny.jsonl"
    qwen_packet_jsonl = tmp_path / "qwen.jsonl"
    tiny_artifact = tmp_path / "tiny_artifact.json"
    _write_jsonl(tiny_packet_jsonl, tiny_rows)
    _write_jsonl(qwen_packet_jsonl, qwen_rows)
    _write_json(tiny_artifact, {"ok": True})

    hidden_path = tmp_path / "hidden.npz"
    hidden = np.zeros((len(answers), 4, 1, 6), dtype=np.float32)
    for row_index, answer in enumerate(answers):
        hidden[row_index, :, 0, :] = 0.05 * row_index
        hidden[row_index, answer, 0, 0] = 1.0
        hidden[row_index, qwen_hybrid[row_index], 0, 1] = 1.0
        hidden[row_index, packet[row_index], 0, 2] = 1.0
    np.savez(hidden_path, features=hidden)
    score_cache = tmp_path / "scores.json"
    _write_json(
        score_cache,
        {
            "row_count": len(answers),
            "row_ids": [str(index) for index in range(len(answers))],
            "source_predictions": qwen_hybrid,
            "source_scores": [_score_row(prediction) for prediction in qwen_hybrid],
        },
    )
    qwen_global = tmp_path / "qwen_global.json"
    _write_json(
        qwen_global,
        {
            "eval_slices": [
                {
                    "name": "unit",
                    "start": 0,
                    "end": len(answers),
                    "rows": len(answers),
                    "score_cache": str(score_cache),
                    "hidden_cache": str(hidden_path),
                }
            ]
        },
    )
    return tiny_packet_jsonl, tiny_artifact, qwen_packet_jsonl, qwen_global


def test_receiver_acceptance_gate_builds(tmp_path) -> None:
    tiny_packet_jsonl, tiny_artifact, qwen_packet_jsonl, qwen_global = _fixture(tmp_path)
    payload = gate.build_gate(
        output_dir=tmp_path / "out",
        tiny_packet_jsonl=tiny_packet_jsonl,
        tiny_artifact=tiny_artifact,
        qwen_packet_jsonl=qwen_packet_jsonl,
        qwen_global_artifact=qwen_global,
        train_prefixes=(8,),
        ridges=(1.0,),
        k_values=(3,),
        bootstrap_samples=5,
    )

    assert payload["headline"]["row_count"] == 12
    assert payload["headline"]["default_train_prefix_rows"] == 8
    assert payload["frontier_rows"]
    assert payload["systems_packet_sideband"]["raw_payload_bytes_per_request"] == 2
    assert (tmp_path / "out" / "hellaswag_receiver_acceptance_gate.json").exists()
    assert (tmp_path / "out" / "hellaswag_receiver_acceptance_gate.md").exists()
    assert (tmp_path / "out" / "manifest.json").exists()


def test_receiver_acceptance_gate_rejects_misaligned_rows(tmp_path) -> None:
    tiny_packet_jsonl, tiny_artifact, qwen_packet_jsonl, qwen_global = _fixture(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match="not aligned"):
        gate.build_gate(
            output_dir=tmp_path / "out",
            tiny_packet_jsonl=tiny_packet_jsonl,
            tiny_artifact=tiny_artifact,
            qwen_packet_jsonl=qwen_packet_jsonl,
            qwen_global_artifact=qwen_global,
            train_prefixes=(8,),
            ridges=(1.0,),
            k_values=(3,),
            bootstrap_samples=5,
        )
