from __future__ import annotations

import json

from scripts import build_source_private_hellaswag_score_packet_headroom as headroom


def _row(row_id: str, answer_index: int) -> dict[str, object]:
    labels = ["A", "B", "C", "D"]
    return {
        "id": row_id,
        "content_id": row_id,
        "question": f"Context {row_id}",
        "choices": [
            f"{row_id} ending A",
            f"{row_id} ending B",
            f"{row_id} ending C",
            f"{row_id} ending D",
        ],
        "choice_labels": labels,
        "answer_index": answer_index,
        "answer_label": labels[answer_index],
        "source_name": "unit",
    }


def test_hellaswag_score_packet_headroom_can_promote_margin_decoder(tmp_path, monkeypatch) -> None:
    rows = [
        _row("cal_low_margin_second_is_right", 1),
        _row("heldout_low_margin_second_is_right", 1),
        _row("cal_high_margin_top_is_right", 0),
        _row("heldout_high_margin_top_is_right", 0),
    ]
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

    def fake_source_scores(**kwargs):
        source_scores = [
            [0.51, 0.50, 0.00, -0.10],
            [0.52, 0.51, 0.00, -0.10],
            [1.00, 0.10, 0.00, -0.10],
            [1.10, 0.10, 0.00, -0.10],
        ]
        source_predictions = [0, 0, 0, 0]
        return source_scores, source_predictions, {"kind": "unit", "prompt_mode": "continuation"}, None

    monkeypatch.setattr(headroom, "_source_scores", fake_source_scores)

    payload = headroom.build_headroom(
        output_dir=tmp_path / "out",
        eval_path=eval_path,
        score_cache=None,
        source_lm_model="unit",
        source_lm_device="auto_cpu",
        source_lm_dtype="float32",
        source_lm_max_length=128,
        source_lm_normalization="mean",
        source_lm_prompt_mode="continuation",
        local_files_only=True,
        run_date="2026-05-01",
    )

    assert payload["pass_gate"] is True
    assert payload["headline"]["source_label_text_heldout_accuracy"] == 0.5
    assert payload["headline"]["best_rank_bin_packet_heldout_accuracy"] == 1.0
    assert payload["headline"]["best_rank_bin_packet_minus_source_label_text_heldout"] == 0.5
    assert (tmp_path / "out" / "hellaswag_score_packet_headroom.json").exists()


def test_hellaswag_score_cache_rejects_wrong_row_order(tmp_path) -> None:
    rows = [_row("first", 0), _row("second", 1)]
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

    cache_path = tmp_path / "score_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "row_count": 2,
                "row_ids": ["second", "first"],
                "content_digest": headroom._content_digest([]),
                "source_scores": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                "source_predictions": [0, 1],
                "source_model": {"kind": "unit"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        headroom.build_headroom(
            output_dir=tmp_path / "out",
            eval_path=eval_path,
            score_cache=cache_path,
            source_lm_model="unit",
            source_lm_device="auto_cpu",
            source_lm_dtype="float32",
            source_lm_max_length=128,
            source_lm_normalization="mean",
            source_lm_prompt_mode="continuation",
            local_files_only=True,
            run_date="2026-05-01",
        )
    except ValueError as exc:
        assert "row id order" in str(exc)
    else:
        raise AssertionError("expected stale score cache to be rejected")
