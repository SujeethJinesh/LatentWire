from __future__ import annotations

import json

import numpy as np

from scripts import build_source_private_hellaswag_qwen_to_phi_protected_rival_packet_gate as gate


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _base_row(index: int) -> dict:
    answer = index % 4
    hybrid = answer if index % 3 else (answer + 1) % 4
    target = answer if index % 5 == 0 else (answer + 2) % 4
    row = {
        "row_id": f"row-{index}",
        "answer_index": answer,
        "selected_prediction": hybrid,
        "hidden_mean_prediction": hybrid,
        "score_mean_prediction": hybrid if index % 2 else (hybrid + 1) % 4,
        "vote_prediction": hybrid,
        "score_vote_prediction": hybrid,
        "selected_margin": float((index % 7) / 3.0),
        "source_label_prediction": (hybrid + 1) % 4,
        "source_rank_only_bagged_prediction": (hybrid + 2) % 4,
        "score_only_bagged_prediction": (hybrid + 3) % 4,
        "trained_label_prediction": (hybrid + 1) % 4,
        "wrong_example_hidden_prediction": (hybrid + 2) % 4,
        "zero_hidden_prediction": 0,
        "candidate_roll_hidden_prediction": (hybrid + 1) % 4,
        "score_channel_roll_hidden_prediction": (hybrid + 2) % 4,
        "_target_for_cache": target,
    }
    row["qwen_hybrid_prediction"] = hybrid
    return row


def test_protected_rival_packet_decodes_default_and_rival():
    row = _base_row(0)
    row["qwen_hybrid_prediction"] = 1
    row["qwen_source_score_prediction"] = 2
    row["qwen_source_scores"] = [0.1, 1.0, 1.5, -0.2]
    code = gate._encode_packet_code(row, "code8_hybrid_rival")
    decoded = gate._decode_packet_code(code, "code8_hybrid_rival")
    assert decoded["ids"]["hybrid"] == 1
    assert decoded["ids"]["rival"] == 2
    rolled = gate._encode_packet_code(row, "code8_hybrid_rival", source_transform="candidate_roll_code")
    assert rolled != code


def test_pair_oracle_keeps_fixed_hybrid_default():
    rows = [_base_row(index) for index in range(4)]
    for row in rows:
        row["qwen_source_scores"] = [0.0, 0.0, 0.0, 0.0]
        row["qwen_source_scores"][(row["answer_index"] + 1) % 4] = 2.0
        row["qwen_source_scores"][row["answer_index"]] = 1.5
        row["qwen_source_score_prediction"] = int(np.argmax(row["qwen_source_scores"]))
    fixed = np.asarray([row["qwen_hybrid_prediction"] for row in rows], dtype=np.int64)
    oracle_predictions = gate._pair_oracle(rows, mode="code8_hybrid_rival")
    answers = np.asarray([row["answer_index"] for row in rows], dtype=np.int64)
    assert np.all((oracle_predictions == answers) | (oracle_predictions == fixed))


def test_build_gate_writes_protected_rival_artifacts(tmp_path):
    qwen_path = tmp_path / "qwen.jsonl"
    phi_path = tmp_path / "phi.json"
    score_path = tmp_path / "qwen_scores.json"
    rows = [_base_row(index) for index in range(12)]
    _write_jsonl(qwen_path, [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows])
    phi_path.write_text(
        json.dumps(
            {
                "row_count": len(rows),
                "row_ids": [row["row_id"] for row in rows],
                "source_predictions": [row["_target_for_cache"] for row in rows],
                "source_scores": [
                    [float(candidate == row["_target_for_cache"]) for candidate in range(4)]
                    for row in rows
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    score_path.write_text(
        json.dumps(
            {
                "row_count": len(rows),
                "row_ids": [row["row_id"] for row in rows],
                "source_model": {"name": "synthetic"},
                "source_predictions": [int(row["qwen_hybrid_prediction"]) for row in rows],
                "source_scores": [
                    [
                        float(candidate == row["qwen_hybrid_prediction"])
                        + 0.2 * float(candidate == ((row["qwen_hybrid_prediction"] + 1) % 4))
                        + 0.01 * index
                        for candidate in range(4)
                    ]
                    for index, row in enumerate(rows)
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    payload = gate.build_gate(
        output_dir=tmp_path / "out",
        slices=(
            {
                "slice_start": 0,
                "slice_end_exclusive": len(rows),
                "qwen_predictions": qwen_path,
                "phi_target_score_cache": phi_path,
            },
        ),
        source_score_cache=score_path,
        fit_rows_per_slice=4,
        select_rows_per_slice=4,
        bootstrap_samples=50,
        run_date="2026-05-04",
    )
    assert payload["headline"]["eval_rows"] == 4
    assert payload["packet_contract"]["source_text_exposed"] is False
    assert payload["source_score_metadata"]["source_score_cache_rows"] == len(rows)
    method_names = {row["method"] for row in payload["method_rows"]}
    assert "selected_protected_pair_decoder" in method_names
    assert "source_score_row_shuffle_before_encoding_pair_decoder_control" in method_names
    assert (tmp_path / "out" / "hellaswag_qwen_to_phi_protected_rival_packet_gate.json").exists()


def test_fit_select_diagnostic_is_not_promotable(tmp_path):
    row = gate._method_row(
        name="fit_select_pair_decoder_diagnostic",
        rows=[{"answer_index": 0}],
        predictions=np.asarray([0]),
        fixed_hybrid=np.asarray([1]),
        candidate_only=np.asarray([1]),
        target_only=np.asarray([0]),
        bootstrap_samples=20,
        raw_payload_bytes=1,
        framed_record_bytes=4,
        details={"not_promotable": True, "uses_select_labels_for_training": True},
    )
    details = json.loads(row["details"])
    assert details["not_promotable"] is True
