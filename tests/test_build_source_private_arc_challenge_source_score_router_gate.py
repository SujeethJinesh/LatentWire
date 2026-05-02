from __future__ import annotations

import csv
import json

from scripts import build_source_private_arc_challenge_source_score_router_gate as gate


def _write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _arc_row(content_id: str, answer_index: int) -> dict:
    return {
        "id": content_id,
        "content_id": content_id,
        "question": f"Question {content_id}?",
        "choices": [f"{content_id} A", f"{content_id} B", f"{content_id} C"],
        "choice_labels": ["A", "B", "C"],
        "answer_index": answer_index,
        "answer_label": ["A", "B", "C"][answer_index],
    }


def _prediction(
    *,
    split: str,
    seed: int,
    content_id: str,
    condition: str,
    source_index: int,
    correct: bool,
) -> dict:
    return {
        "split": split,
        "seed": seed,
        "content_id": content_id,
        "row_id": content_id,
        "condition": condition,
        "answer_index": 0,
        "prediction_index": source_index,
        "correct": correct,
        "payload_bytes": 12,
        "latency_ms": 0.1,
        "metadata": {
            "source_selected_index": source_index,
            "scores": [1.0 if index == source_index else 0.0 for index in range(3)],
            "best_score": 1.0,
            "packet_code_l2": 1.0,
        },
    }


def _score_cache_row(content_id: str, source_index: int, *, margin: float, family: str) -> dict:
    scores = [0.0, 0.0, 0.0]
    scores[source_index] = margin
    return {
        "row_id": content_id,
        "content_id": content_id,
        "source_family": family,
        "source_model": "toy",
        "source_score_mode": "lm_choice_loglikelihood",
        "source_lm_prompt_mode": "qa",
        "source_lm_normalization": "mean",
        "source_scores": scores,
        "source_selected_index": source_index,
        "source_selected_choice_sha256": "toy",
        "source_visible_fields": ["question", "choices"],
        "forbidden_source_fields": ["answer", "answerKey", "answer_index", "answer_label", "gold"],
        "best_score": margin,
        "margin": margin,
        "neg_entropy": -0.1 / max(margin, 0.1),
        "score_std": margin / 3.0,
    }


def _parent_cache_row(content_id: str, source_index: int, *, family: str) -> dict:
    return {
        "row_id": content_id,
        "content_id": content_id,
        "source_family": family,
        "source_selected_index": source_index,
        "source_selected_choice_sha256": "toy",
        "source_visible_fields": ["question", "choices"],
        "forbidden_source_fields": ["answer", "answerKey", "answer_index", "answer_label", "gold"],
    }


def test_source_score_router_gate_uses_score_caches_and_writes_artifacts(tmp_path) -> None:
    parent = tmp_path / "parent"
    output = tmp_path / "out"
    parent.mkdir()
    validation_rows = [_arc_row(f"validation-{index}", index % 3) for index in range(4)]
    test_rows = [_arc_row(f"test-{index}", index % 3) for index in range(4)]
    validation_path = tmp_path / "validation.jsonl"
    test_path = tmp_path / "test.jsonl"
    _write_jsonl(validation_path, validation_rows)
    _write_jsonl(test_path, test_rows)
    qwen_validation_parent = parent / "qwen_validation.jsonl"
    qwen_test_parent = parent / "qwen_test.jsonl"
    tiny_validation_parent = parent / "tinyllama_validation" / "source_prediction_cache.jsonl"
    tiny_test_parent = parent / "tinyllama_test" / "source_prediction_cache.jsonl"

    prediction_rows = []
    agreement_rows = []
    for split, rows in (("validation", validation_rows), ("test", test_rows)):
        for row_index, row in enumerate(rows):
            alt_correct = row_index % 2 == 0
            qwen_correct = not alt_correct
            prediction_rows.extend(
                [
                    _prediction(
                        split=split,
                        seed=47,
                        content_id=row["content_id"],
                        condition=gate.ALT_CONDITION,
                        source_index=0,
                        correct=alt_correct,
                    ),
                    _prediction(
                        split=split,
                        seed=47,
                        content_id=row["content_id"],
                        condition=gate.QWEN_CONDITION,
                        source_index=1,
                        correct=qwen_correct,
                    ),
                ]
            )
            agreement_rows.append(
                {
                    "split": split,
                    "row_id": row["id"],
                    "content_id": row["content_id"],
                    "alt_source_selected_index": "0",
                    "qwen_source_selected_index": "1",
                    "agree": "False",
                    "answer_index": str(row["answer_index"]),
                    "alt_source_correct": str(alt_correct),
                    "qwen_source_correct": str(qwen_correct),
                }
            )
    _write_jsonl(parent / "qwen_disagreement_predictions.jsonl", prediction_rows)
    with (parent / "source_cache_agreement.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(agreement_rows[0]))
        writer.writeheader()
        writer.writerows(agreement_rows)
    _write_jsonl(qwen_validation_parent, [_parent_cache_row(row["content_id"], 1, family="qwen") for row in validation_rows])
    _write_jsonl(qwen_test_parent, [_parent_cache_row(row["content_id"], 1, family="qwen") for row in test_rows])
    _write_jsonl(tiny_validation_parent, [_parent_cache_row(row["content_id"], 0, family="tiny") for row in validation_rows])
    _write_jsonl(tiny_test_parent, [_parent_cache_row(row["content_id"], 0, family="tiny") for row in test_rows])
    (parent / "source_family_cache_falsification.json").write_text(
        json.dumps(
            {
                "pass_gate": False,
                "source_cache_audit": {
                    "qwen_validation_cache": str(qwen_validation_parent),
                    "qwen_test_cache": str(qwen_test_parent),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    tiny_validation_scores = tmp_path / "tiny_validation_scores.jsonl"
    tiny_test_scores = tmp_path / "tiny_test_scores.jsonl"
    qwen_validation_scores = tmp_path / "qwen_validation_scores.jsonl"
    qwen_test_scores = tmp_path / "qwen_test_scores.jsonl"
    for path, rows, family, index in (
        (tiny_validation_scores, validation_rows, "tiny", 0),
        (tiny_test_scores, test_rows, "tiny", 0),
        (qwen_validation_scores, validation_rows, "qwen", 1),
        (qwen_test_scores, test_rows, "qwen", 1),
    ):
        _write_jsonl(
            path,
            [
                _score_cache_row(
                    row["content_id"],
                    index,
                    margin=2.0 if row_index % 2 == 0 and family == "tiny" else 0.5,
                    family=family,
                )
                for row_index, row in enumerate(rows)
            ],
        )

    payload = gate.build_source_score_router_gate(
        parent_dir=parent,
        output_dir=output,
        validation_path=validation_path,
        test_path=test_path,
        tiny_validation_score_cache=tiny_validation_scores,
        tiny_test_score_cache=tiny_test_scores,
        qwen_validation_score_cache=qwen_validation_scores,
        qwen_test_score_cache=qwen_test_scores,
        materialize_score_caches=False,
        force_rematerialize=False,
        tiny_source_model=tmp_path / "tiny",
        qwen_source_model=tmp_path / "qwen",
        source_lm_device="cpu",
        source_lm_dtype="float32",
        tiny_source_lm_max_length=16,
        qwen_source_lm_max_length=16,
        source_lm_normalization="mean",
        source_lm_prompt_mode="qa",
        local_files_only=True,
        bootstrap_samples=20,
        min_lookup_count=1,
        min_gap_over_qwen=-1.0,
        source_confidence_sidecar_bytes=1,
    )

    assert payload["gate"] == "source_private_arc_challenge_source_score_router_gate"
    assert payload["source_score_cache_audit"]["tiny_test"]["selected_indices_match_parent_cache"] is True
    assert payload["selected_rule"]["metric"] in [*gate.SCALAR_METRICS, gate.SOURCE_INDEX_LOOKUP]
    assert payload["selected_rule_test_summary"]["aggregate"]["n"] == len(test_rows)
    assert (output / "source_score_router_gate.json").exists()
    assert (output / "source_score_router_rule_metrics.csv").exists()
    assert (output / "selected_source_score_router_predictions.jsonl").exists()
