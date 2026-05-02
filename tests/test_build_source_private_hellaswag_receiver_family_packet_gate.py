from __future__ import annotations

import json

import pytest

from scripts import build_source_private_hellaswag_receiver_family_packet_gate as gate


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(tmp_path, *, mismatch: bool = False):
    rows = [
        {
            "row_id": str(index),
            "answer_index": index % 4,
            "selected_prediction": [0, 1, 2, 0, 0, 1, 3, 3][index],
            "selected_margin": 0.25 + 0.1 * index,
            "wrong_example_hidden_prediction": [1, 0, 1, 2, 1, 0, 2, 2][index],
            "candidate_roll_hidden_prediction": [1, 2, 3, 1, 1, 2, 0, 0][index],
            "zero_hidden_prediction": [1, 1, 1, 1, 0, 0, 0, 0][index],
            "source_label_prediction": [1, 1, 1, 1, 0, 0, 0, 0][index],
        }
        for index in range(8)
    ]
    packet_jsonl = tmp_path / "source_packets.jsonl"
    _write_jsonl(packet_jsonl, rows)
    source_artifact = tmp_path / "source_artifact.json"
    _write_json(source_artifact, {"ok": True})

    row_ids = [str(index + (1 if mismatch else 0)) for index in range(8)]
    score_cache = tmp_path / "target_scores.json"
    _write_json(
        score_cache,
        {
            "row_count": 8,
            "row_ids": row_ids,
            "source_predictions": [1, 1, 0, 3, 0, 0, 2, 1],
            "source_scores": [
                [0.1, 0.3, 0.0, -0.1],
                [0.0, 0.4, 0.1, -0.1],
                [0.5, 0.1, 0.2, 0.0],
                [0.0, 0.1, 0.2, 0.6],
                [0.7, 0.1, 0.2, 0.0],
                [0.6, 0.2, 0.1, 0.0],
                [0.1, 0.0, 0.5, 0.2],
                [0.2, 0.5, 0.1, 0.0],
            ],
        },
    )
    target_artifact = tmp_path / "target_global.json"
    _write_json(
        target_artifact,
        {
            "eval_slices": [
                {
                    "name": "tiny",
                    "start": 0,
                    "end": 8,
                    "rows": 8,
                    "score_cache": str(score_cache),
                }
            ]
        },
    )
    return packet_jsonl, target_artifact, source_artifact


def test_receiver_family_packet_gate_builds(tmp_path):
    packet_jsonl, target_artifact, source_artifact = _fixture(tmp_path)
    payload = gate.build_gate(
        output_dir=tmp_path / "out",
        source_packet_jsonl=packet_jsonl,
        target_global_artifact=target_artifact,
        source_packet_artifact=source_artifact,
        train_prefix_rows=4,
        bootstrap_samples=50,
        ridges=(1.0, 10.0),
    )

    assert payload["headline"]["row_count"] == 8
    assert payload["headline"]["train_rows"] == 4
    assert payload["headline"]["eval_rows"] == 4
    assert payload["headline"]["selected_receiver_kind"] in {
        "candidate_ridge_receiver",
        "target_margin_accept_packet",
    }
    assert payload["baseline_rows"][0]["name"] == "target_only"
    assert (tmp_path / "out" / "hellaswag_receiver_family_packet_gate.json").exists()
    assert (tmp_path / "out" / "manifest.json").exists()


def test_receiver_family_packet_gate_rejects_misaligned_rows(tmp_path):
    packet_jsonl, target_artifact, source_artifact = _fixture(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match="not aligned"):
        gate.build_gate(
            output_dir=tmp_path / "out",
            source_packet_jsonl=packet_jsonl,
            target_global_artifact=target_artifact,
            source_packet_artifact=source_artifact,
            train_prefix_rows=4,
            bootstrap_samples=10,
            ridges=(1.0,),
        )
