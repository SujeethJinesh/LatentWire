from __future__ import annotations

import json

from scripts import build_source_private_hellaswag_nonqwen_anchor_acceptor_gate as gate


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _slice(root, *, name: str, start: int, row_count: int):
    slice_dir = root / name
    packet_rows = []
    row_ids = []
    source_scores = []
    target_scores = []
    for offset in range(row_count):
        row_id = str(start + offset)
        answer = offset % 4
        packet = answer if offset % 2 else (answer + 1) % 4
        target = answer if offset % 3 == 0 else (answer + 2) % 4
        row_ids.append(row_id)
        packet_rows.append(
            {
                "row_id": row_id,
                "answer_index": answer,
                "selected_prediction": packet,
                "selected_margin": 0.1,
                "source_label_prediction": packet,
                "score_only_bagged_prediction": packet,
                "wrong_example_hidden_prediction": (answer + 2) % 4,
                "candidate_roll_hidden_prediction": (packet + 1) % 4,
                "zero_hidden_prediction": 0,
                "row_shuffle_packet": (packet + 2) % 4,
                "random_same_byte_packet": (offset + 1) % 4,
                "target_derived_packet": target,
                "candidate_derangement_packet": (answer + 1) % 4,
            }
        )
        source_scores.append([2.0 if choice == packet else 0.0 for choice in range(4)])
        target_scores.append([2.0 if choice == target else 0.0 for choice in range(4)])
    _write_jsonl(slice_dir / "tinyllama_source_packet_slice_augmented.jsonl", packet_rows)
    _write_json(
        slice_dir / "target_score_cache.json",
        {
            "row_count": row_count,
            "row_ids": row_ids,
            "source_scores": target_scores,
            "source_predictions": [
                max(range(4), key=lambda choice, row=row: target_scores[row][choice])
                for row in range(row_count)
            ],
        },
    )
    _write_json(
        slice_dir / "hellaswag_nonqwen_receiver_family_packet_gate.json",
        {
            "gate": "source_private_hellaswag_nonqwen_receiver_family_packet_gate",
            "headline": {"slice_start": start, "slice_end_exclusive": start + row_count},
        },
    )
    return slice_dir, row_ids, source_scores


def test_anchor_acceptor_gate_builds_packet_preserving_slices(tmp_path):
    left_dir, left_ids, left_scores = _slice(tmp_path, name="left", start=100, row_count=24)
    right_dir, right_ids, right_scores = _slice(tmp_path, name="right", start=200, row_count=24)
    source_score_cache = tmp_path / "source_eval_score_cache.json"
    all_scores = left_scores + right_scores
    _write_json(
        source_score_cache,
        {
            "row_count": len(all_scores),
            "row_ids": left_ids + right_ids,
            "source_scores": all_scores,
            "source_predictions": [
                max(range(4), key=lambda choice, row=row: row[choice])
                for row in all_scores
            ],
        },
    )

    payload = gate.build_gate(
        output_dir=tmp_path / "out",
        slice_dirs=(left_dir, right_dir),
        source_score_cache=source_score_cache,
        train_prefix_rows=16,
        bootstrap_samples=10,
        ridges=(1.0,),
        bins=(2,),
        anchors=(4,),
        run_date="2026-05-03",
    )

    assert payload["gate"] == "source_private_hellaswag_nonqwen_anchor_acceptor_gate"
    assert payload["headline"]["slice_count"] == 2
    assert payload["packet_contract"]["forbidden_source_fields"][-1] == "source_logits"
    assert payload["headline"]["max_selected_raw_packet_bytes"] <= 1
    assert (tmp_path / "out" / "hellaswag_nonqwen_anchor_acceptor_gate.json").exists()
    assert (tmp_path / "out" / "slice_rows.csv").exists()
