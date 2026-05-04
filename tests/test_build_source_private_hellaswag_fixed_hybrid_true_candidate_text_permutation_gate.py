from __future__ import annotations

import json

from scripts import build_source_private_hellaswag_fixed_hybrid_true_candidate_text_permutation_gate as gate


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _eval_row(row_id: int, answer: int) -> dict:
    return {
        "answer_index": answer,
        "answer_label": "ABCD"[answer],
        "choice_labels": ["A", "B", "C", "D"],
        "choices": [f"choice-{row_id}-{index}" for index in range(4)],
        "content_id": f"content-{row_id}",
        "id": str(row_id),
        "question": f"context {row_id}",
        "row_index": row_id,
        "source_name": "synthetic/hellaswag",
    }


def _prediction(row_id: str, answer: int, prediction: int) -> dict:
    controls = {
        "selected_prediction": prediction,
        "hidden_mean_prediction": prediction,
        "score_mean_prediction": prediction,
        "vote_prediction": prediction,
        "source_label_prediction": prediction,
        "source_rank_only_bagged_prediction": prediction,
        "score_only_bagged_prediction": prediction,
        "score_vote_prediction": prediction,
        "trained_label_prediction": prediction,
        "wrong_example_hidden_prediction": (prediction + 1) % 4,
        "zero_hidden_prediction": prediction,
        "candidate_roll_hidden_prediction": (prediction + 1) % 4,
        "score_channel_roll_hidden_prediction": (prediction + 1) % 4,
    }
    return {
        "row_id": str(row_id),
        "answer_index": int(answer),
        "selected_margin": 0.1,
        **{key: int(value) for key, value in controls.items()},
    }


def test_fixed_hybrid_true_candidate_text_permutation_prepare_and_evaluate(tmp_path):
    eval_rows = [_eval_row(index, index % 4) for index in range(8)]
    eval_path = tmp_path / "eval.jsonl"
    _write_jsonl(eval_path, eval_rows)

    prepare = gate.prepare_permuted_eval(
        eval_full_path=eval_path,
        output_dir=tmp_path / "out",
        eval_slice_start=0,
        eval_rows=len(eval_rows),
        permutation_mode="fixed8_nonidentity",
        run_date="2026-05-03",
    )
    permuted_eval = [json.loads(line) for line in (tmp_path / "out" / "hellaswag_validation_permuted_rows_0_8.jsonl").read_text().splitlines()]
    assert prepare["all_nonidentity_permutations"] is True
    assert all(row["choices"] != eval_rows[index]["choices"] for index, row in enumerate(permuted_eval))

    original_predictions = []
    permuted_predictions = []
    for original, permuted in zip(eval_rows, permuted_eval, strict=True):
        canonical_prediction = int(original["answer_index"])
        display_prediction = int(permuted["permutation_canonical_to_display"][canonical_prediction])
        original_predictions.append(_prediction(original["id"], original["answer_index"], canonical_prediction))
        permuted_predictions.append(_prediction(permuted["id"], permuted["answer_index"], display_prediction))
    original_path = tmp_path / "original.jsonl"
    permuted_path = tmp_path / "permuted.jsonl"
    _write_jsonl(original_path, original_predictions)
    _write_jsonl(permuted_path, permuted_predictions)
    run_json = tmp_path / "hidden_run.json"
    _write_json(
        run_json,
        {
            "eval_cache_metadata": {
                "score_cache_hit": False,
                "hidden_cache_hit": False,
            }
        },
    )

    payload = gate.evaluate_gate(
        output_dir=tmp_path / "out",
        original_predictions=original_path,
        permuted_predictions=permuted_path,
        permuted_eval_path=tmp_path / "out" / "hellaswag_validation_permuted_rows_0_8.jsonl",
        permuted_run_json=run_json,
        eval_rows=len(eval_rows),
        bootstrap_samples=30,
        run_date="2026-05-03",
    )

    headline = payload["headline"]
    assert payload["pass_gate"] is True
    assert payload["smoke_pass_gate"] is True
    assert payload["promotion_pass_gate"] is False
    assert headline["original_fixed_hybrid_accuracy"] == 1.0
    assert headline["remapped_fixed_hybrid_accuracy"] == 1.0
    assert headline["fixed_hybrid_canonical_consistency_rate"] == 1.0
    assert headline["unremapped_fixed_hybrid_accuracy"] < 1.0
    assert headline["score_cache_hit"] is False
    assert headline["hidden_cache_hit"] is False
    assert (tmp_path / "out" / "comparison_rows.csv").exists()
    assert (tmp_path / "out" / "component_rows.csv").exists()
