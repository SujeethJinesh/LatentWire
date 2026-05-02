from __future__ import annotations

import json

from scripts import build_source_private_hellaswag_score_packet_headroom as headroom
from scripts import build_source_private_hellaswag_train_source_score_repair_probe as probe
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate


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


def _write_rows(path, rows) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_rank_decoder_learns_margin_conditioned_top2_flip() -> None:
    rows = [
        arc_gate.ArcRow(
            row_id="low_margin_flip",
            content_id="low_margin_flip",
            question="Context low",
            choices=("A", "B", "C", "D"),
            choice_labels=("A", "B", "C", "D"),
            answer_index=1,
            answer_label="B",
            source_name="unit",
        ),
        arc_gate.ArcRow(
            row_id="high_margin_trust",
            content_id="high_margin_trust",
            question="Context high",
            choices=("A", "B", "C", "D"),
            choice_labels=("A", "B", "C", "D"),
            answer_index=0,
            answer_label="A",
            source_name="unit",
        ),
    ]
    scores = [[0.51, 0.50, 0.0, -0.1], [1.0, 0.1, 0.0, -0.1]]
    feature_fns = probe._make_feature_fns(scores)
    feature_fn, _ = feature_fns["top2_margin_2bin"]
    decoder = probe._fit_rank_decoder(rows=rows, scores=scores, feature_fn=feature_fn, max_rank=2)

    predictions = probe._predict_rank_decoder(scores, decoder, feature_fn)

    assert predictions == [1, 0]


def test_train_source_score_repair_probe_writes_artifacts(tmp_path) -> None:
    train_rows = [
        _row("train_low_0", 1),
        _row("train_low_1", 1),
        _row("train_high_0", 0),
        _row("train_high_1", 0),
        _row("train_low_2", 1),
        _row("train_high_2", 0),
    ]
    train_scores = [
        [0.51, 0.50, 0.0, -0.1],
        [0.52, 0.51, 0.0, -0.1],
        [1.0, 0.1, 0.0, -0.1],
        [1.1, 0.1, 0.0, -0.1],
        [0.53, 0.52, 0.0, -0.1],
        [1.2, 0.1, 0.0, -0.1],
    ]
    eval_rows = [_row("eval_low", 1), _row("eval_high", 0)]
    eval_scores = [[0.515, 0.505, 0.0, -0.1], [1.05, 0.1, 0.0, -0.1]]
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_rows(train_path, train_rows)
    _write_rows(eval_path, eval_rows)
    loaded_train_rows = arc_gate._load_rows(train_path)
    loaded_eval_rows = arc_gate._load_rows(eval_path)

    train_cache = tmp_path / "train_score_cache.json"
    eval_cache = tmp_path / "eval_score_cache.json"
    headroom._write_score_cache(
        train_cache,
        rows=loaded_train_rows,
        source_scores=train_scores,
        source_predictions=[0, 0, 0, 0, 0, 0],
        source_model={"kind": "unit"},
    )
    headroom._write_score_cache(
        eval_cache,
        rows=loaded_eval_rows,
        source_scores=eval_scores,
        source_predictions=[0, 0],
        source_model={"kind": "unit"},
    )

    payload = probe.build_probe(
        output_dir=tmp_path / "out",
        train_path=train_path,
        eval_path=eval_path,
        eval_score_cache=eval_cache,
        train_score_cache=train_cache,
        train_score_rows=6,
        selection_seed=0,
        dev_fraction=0.25,
        source_lm_model="unit",
        source_lm_device="auto_cpu",
        source_lm_dtype="float32",
        source_lm_max_length=128,
        source_lm_normalization="mean",
        source_lm_prompt_mode="continuation",
        local_files_only=True,
        run_date="2026-05-01",
    )

    assert payload["gate"] == "source_private_hellaswag_train_source_score_repair_probe"
    assert payload["scored_train_rows"] == 6
    assert "trained_choice_bias_label_copy" in payload["policy_readouts"]
    assert payload["headline"]["source_top2_oracle_accuracy"] == 1.0
    assert (tmp_path / "out" / "hellaswag_train_source_score_repair_probe.json").exists()
    assert (tmp_path / "out" / "policy_readouts.jsonl").exists()
