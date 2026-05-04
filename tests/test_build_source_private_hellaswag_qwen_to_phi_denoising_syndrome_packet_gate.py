from __future__ import annotations

import json

from scripts.build_source_private_hellaswag_qwen_to_phi_denoising_syndrome_packet_gate import (
    _encode_source_code,
    _hybrid_prediction,
    _packet_bytes,
    build_gate,
)


def _qwen_row(row_id: int, *, answer: int, selected: int, hidden: int, score: int, vote: int) -> dict[str, object]:
    return {
        "answer_index": answer,
        "candidate_roll_hidden_prediction": (hidden + 1) % 4,
        "hidden_mean_prediction": hidden,
        "row_id": str(row_id),
        "score_channel_roll_hidden_prediction": (hidden + 2) % 4,
        "score_mean_prediction": score,
        "score_only_bagged_prediction": score,
        "score_vote_prediction": vote,
        "selected_margin": 0.4 + 0.2 * (row_id % 5),
        "selected_prediction": selected,
        "source_label_prediction": selected,
        "source_rank_only_bagged_prediction": selected,
        "trained_label_prediction": row_id % 4,
        "vote_prediction": vote,
        "wrong_example_hidden_prediction": (hidden + 3) % 4,
        "zero_hidden_prediction": 0,
    }


def _write_slice(tmp_path, *, name: str, start_row_id: int) -> dict[str, object]:
    qwen_path = tmp_path / f"{name}_qwen.jsonl"
    phi_path = tmp_path / f"{name}_phi.json"
    qwen_rows = []
    phi_predictions = []
    phi_scores = []
    row_ids = []
    for offset in range(8):
        row_id = start_row_id + offset
        answer = offset % 4
        selected = answer if offset % 3 != 0 else (answer + 1) % 4
        hidden = selected if offset % 2 == 0 else answer
        score = hidden if offset % 4 != 1 else (hidden + 1) % 4
        vote = answer
        qwen_rows.append(
            _qwen_row(row_id, answer=answer, selected=selected, hidden=hidden, score=score, vote=vote)
        )
        row_ids.append(str(row_id))
        phi_prediction = answer if offset % 4 == 1 else (answer + 2) % 4
        phi_predictions.append(phi_prediction)
        scores = [-2.0, -2.5, -3.0, -3.5]
        scores[phi_prediction] = -1.0
        phi_scores.append(scores)
    qwen_path.write_text("\n".join(json.dumps(row) for row in qwen_rows) + "\n", encoding="utf-8")
    phi_path.write_text(
        json.dumps(
            {
                "row_count": len(qwen_rows),
                "row_ids": row_ids,
                "source_predictions": phi_predictions,
                "source_scores": phi_scores,
                "source_model": {"kind": "synthetic_phi_score_cache"},
            }
        ),
        encoding="utf-8",
    )
    return {
        "slice_start": start_row_id,
        "slice_end_exclusive": start_row_id + len(qwen_rows),
        "qwen_predictions": str(qwen_path),
        "phi_target_score_cache": str(phi_path),
    }


def test_helpers_encode_packet_fields() -> None:
    row = _qwen_row(10, answer=1, selected=2, hidden=3, score=3, vote=1)
    row["qwen_hybrid_prediction"] = _hybrid_prediction(row)

    assert row["qwen_hybrid_prediction"] == 1
    assert _packet_bytes("code8_hybrid_selected_margin") == (1, 4)
    assert _packet_bytes("code16_policy_margin") == (2, 5)
    assert _encode_source_code(row, "code8_hybrid_selected_margin") < 256
    assert _encode_source_code(row, "code16_policy_margin") < 2**16


def test_denoising_syndrome_gate_writes_controls(tmp_path) -> None:
    slices = (
        _write_slice(tmp_path, name="slice_a", start_row_id=1000),
        _write_slice(tmp_path, name="slice_b", start_row_id=2000),
    )

    payload = build_gate(
        output_dir=tmp_path / "out",
        slices=slices,
        fit_rows_per_slice=2,
        select_rows_per_slice=2,
        bootstrap_samples=50,
        run_date="2026-05-04",
    )

    assert payload["headline"]["fit_rows"] == 4
    assert payload["headline"]["select_rows"] == 4
    assert payload["headline"]["eval_rows"] == 8
    assert payload["headline"]["framed_record_bytes"] in {4, 5}
    assert payload["packet_contract"]["source_text_exposed"] is False
    assert payload["packet_contract"]["raw_scores_or_logits_transmitted"] is False

    rows = {row["method"]: row for row in payload["method_rows"]}
    for method in [
        "denoising_syndrome_packet",
        "fixed_hybrid_vote_on_score_agreement",
        "qwen_candidate_only",
        "phi_target_only",
        "target_or_hybrid_oracle",
        "source_row_shuffle_control",
        "code_value_permutation_control",
        "candidate_roll_code_control",
        "random_same_byte_control",
        "target_derived_code_control",
        "zero_byte_target_ridge_control",
        "label_permutation_decoder_control",
    ]:
        assert method in rows

    out = tmp_path / "out"
    assert (out / "hellaswag_qwen_to_phi_denoising_syndrome_packet_gate.json").exists()
    assert (out / "hellaswag_qwen_to_phi_denoising_syndrome_packet_gate.md").exists()
    assert (out / "method_rows.csv").exists()
    assert (out / "config_rows.csv").exists()
    assert (out / "slice_rows.csv").exists()
    assert (out / "manifest.json").exists()
