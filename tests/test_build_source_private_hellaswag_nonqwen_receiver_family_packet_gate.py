from __future__ import annotations

import hashlib
import json

from scripts import build_source_private_hellaswag_nonqwen_receiver_family_packet_gate as gate


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _content_id(question: str, choices: list[str]) -> str:
    payload = json.dumps({"context": question, "endings": choices}, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _content_digest(rows) -> str:
    return hashlib.sha256("\n".join(row["content_id"] for row in rows).encode("utf-8")).hexdigest()


def test_phi_receiver_family_packet_gate_builds_from_cached_scores(tmp_path):
    eval_rows = []
    packet_rows = []
    for index in range(10):
        question = f"context {index}"
        choices = [f"ending {index}-{choice}" for choice in range(4)]
        answer = index % 4
        cid = _content_id(question, choices)
        eval_rows.append(
            {
                "id": str(index),
                "content_id": cid,
                "question": question,
                "choices": choices,
                "choice_labels": ["A", "B", "C", "D"],
                "answer_index": answer,
                "answer_label": ["A", "B", "C", "D"][answer],
            }
        )
        packet_rows.append(
            {
                "row_id": str(index),
                "answer_index": answer,
                "selected_prediction": answer if index % 3 else (answer + 1) % 4,
                "selected_margin": 0.1 + 0.01 * index,
                "wrong_example_hidden_prediction": (answer + 2) % 4,
                "candidate_roll_hidden_prediction": (answer + 1) % 4,
                "zero_hidden_prediction": 0,
                "source_label_prediction": answer if index % 4 else (answer + 1) % 4,
            }
        )
    eval_path = tmp_path / "hellaswag_validation.jsonl"
    packet_path = tmp_path / "packets.jsonl"
    source_artifact = tmp_path / "source_artifact.json"
    score_cache = tmp_path / "phi_scores.json"
    _write_jsonl(eval_path, eval_rows)
    _write_jsonl(packet_path, packet_rows)
    _write_json(source_artifact, {"ok": True})
    _write_json(
        score_cache,
        {
            "created_utc": "2026-05-02T00:00:00+00:00",
            "row_count": len(eval_rows),
            "row_ids": [row["id"] for row in eval_rows],
            "content_digest": _content_digest(eval_rows),
            "source_scores": [
                [0.5 if choice == ((index + 1) % 4) else 0.0 for choice in range(4)]
                for index in range(len(eval_rows))
            ],
            "source_predictions": [(index + 1) % 4 for index in range(len(eval_rows))],
            "source_model": {
                "kind": "fixture_phi_scores",
                "cache_hit": True,
                "device": "cpu",
            },
        },
    )

    payload = gate.build_gate(
        output_dir=tmp_path / "out",
        eval_full_path=eval_path,
        source_packet_jsonl=packet_path,
        source_packet_artifact=source_artifact,
        slice_start=0,
        slice_rows=10,
        train_prefix_rows=4,
        bootstrap_samples=25,
        target_score_cache=score_cache,
        target_family="FixtureTarget",
        target_lm_model="fixture",
        target_lm_device="cpu",
        target_lm_dtype="float32",
    )

    assert payload["headline"]["row_count"] == 10
    assert payload["headline"]["target_family"] == "FixtureTarget"
    assert payload["headline"]["target_score_cache_hit"] is True
    assert payload["source_packet"]["raw_payload_bytes"] == 2
    assert payload["source_packet"]["framed_record_bytes"] == 5
    assert payload["source_packet"]["exposes_source_kv"] is False
    assert (tmp_path / "out" / "tinyllama_source_packet_slice_augmented.jsonl").exists()
    assert (tmp_path / "out" / "receiver_gate" / "hellaswag_receiver_family_packet_gate.json").exists()
    augmented = (tmp_path / "out" / "tinyllama_source_packet_slice_augmented.jsonl").read_text(
        encoding="utf-8"
    )
    assert "row_shuffle_packet" in augmented
    assert "target_derived_packet" in augmented
