from __future__ import annotations

import json

from scripts import build_source_private_hellaswag_nonqwen_score_simplex_receiver_gate as gate


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _slice(root, *, name: str, start: int, row_count: int) -> tuple[object, list[str], list[list[float]]]:
    slice_dir = root / name
    rows = []
    row_ids = []
    source_scores = []
    target_scores = []
    for offset in range(row_count):
        row_id = str(start + offset)
        answer = offset % 4
        packet = answer if offset % 2 else (answer + 1) % 4
        target = answer if offset % 3 == 0 else (answer + 2) % 4
        row_ids.append(row_id)
        rows.append(
            {
                "row_id": row_id,
                "answer_index": answer,
                "selected_prediction": packet,
                "selected_margin": 0.1 + 0.01 * offset,
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
        source_scores.append([1.0 if choice == packet else 0.0 for choice in range(4)])
        target_scores.append([1.0 if choice == target else 0.0 for choice in range(4)])
    _write_jsonl(slice_dir / "tinyllama_source_packet_slice_augmented.jsonl", rows)
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


def test_nonqwen_score_simplex_receiver_gate_builds_from_cached_slices(tmp_path):
    left_dir, left_ids, left_source_scores = _slice(tmp_path, name="left", start=100, row_count=10)
    right_dir, right_ids, right_source_scores = _slice(tmp_path, name="right", start=200, row_count=10)
    source_score_cache = tmp_path / "source_eval_score_cache.json"
    _write_json(
        source_score_cache,
        {
            "row_count": 20,
            "row_ids": left_ids + right_ids,
            "source_scores": left_source_scores + right_source_scores,
            "source_predictions": [
                int(max(range(4), key=lambda choice, row=row: row[choice]))
                for row in left_source_scores + right_source_scores
            ],
        },
    )

    payload = gate.build_gate(
        output_dir=tmp_path / "out",
        slice_dirs=(left_dir, right_dir),
        source_score_cache=source_score_cache,
        train_prefix_rows=4,
        bootstrap_samples=10,
        ridges=(1.0,),
        run_date="2026-05-03",
    )

    assert payload["gate"] == "source_private_hellaswag_nonqwen_score_simplex_receiver_gate"
    assert payload["headline"]["slice_count"] == 2
    assert payload["headline"]["total_eval_rows"] == 12
    assert any(
        row["name"] == "source_score_argmax" for row in payload["slice_payloads"][0]["baseline_rows"]
    )
    controls = {row["name"] for row in payload["slice_payloads"][0]["control_rows"]}
    assert {"basis_sign_flip_source", "basis_permute_source", "source_score_row_shuffle"} <= controls
    assert (tmp_path / "out" / "hellaswag_nonqwen_score_simplex_receiver_gate.json").exists()
    assert (tmp_path / "out" / "slice_rows.csv").exists()
