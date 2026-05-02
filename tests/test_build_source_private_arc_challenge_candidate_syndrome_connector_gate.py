from __future__ import annotations

import json

from scripts import build_source_private_arc_challenge_candidate_syndrome_connector_gate as gate


def _write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _prediction(
    *,
    split: str,
    seed: int,
    content_id: str,
    condition: str,
    answer_index: int,
    prediction_index: int,
    scores: list[float],
) -> dict:
    return {
        "split": split,
        "seed": seed,
        "content_id": content_id,
        "row_id": content_id,
        "condition": condition,
        "answer_index": answer_index,
        "prediction_index": prediction_index,
        "correct": prediction_index == answer_index,
        "metadata": {
            "source_selected_index": prediction_index,
            "scores": scores,
            "best_score": max(scores),
            "packet_code_l2": 1.0,
        },
    }


def _score_cache_row(content_id: str, scores: list[float], selected_index: int) -> dict:
    ordered = sorted(scores, reverse=True)
    margin = ordered[0] - ordered[1]
    return {
        "row_id": content_id,
        "content_id": content_id,
        "source_family": "toy",
        "source_model": "toy",
        "source_scores": scores,
        "source_selected_index": selected_index,
        "source_selected_choice_sha256": "toy",
        "source_visible_fields": ["question", "choices"],
        "forbidden_source_fields": ["answer", "answerKey", "answer_index", "answer_label", "gold"],
        "best_score": max(scores),
        "margin": margin,
        "neg_entropy": -0.5,
        "score_std": 1.0,
    }


def test_candidate_syndrome_connector_gate_writes_artifacts(tmp_path) -> None:
    parent = tmp_path / "parent"
    score_router = tmp_path / "score_router"
    output = tmp_path / "out"
    prediction_rows = []
    for split, n in (("validation", 8), ("test", 8)):
        for row_index in range(n):
            answer_index = row_index % 3
            content_id = f"{split}-{row_index}"
            for seed in (47, 53):
                tiny_scores = [0.0, 0.0, 0.0]
                tiny_scores[answer_index] = 3.0
                qwen_scores = [1.0, 0.0, 0.0]
                prediction_rows.extend(
                    [
                        _prediction(
                            split=split,
                            seed=seed,
                            content_id=content_id,
                            condition=gate.ALT_CONDITION,
                            answer_index=answer_index,
                            prediction_index=answer_index,
                            scores=tiny_scores,
                        ),
                        _prediction(
                            split=split,
                            seed=seed,
                            content_id=content_id,
                            condition=gate.QWEN_CONDITION,
                            answer_index=answer_index,
                            prediction_index=0,
                            scores=qwen_scores,
                        ),
                    ]
                )
    _write_jsonl(parent / "qwen_disagreement_predictions.jsonl", prediction_rows)
    (score_router / "source_score_router_gate.json").parent.mkdir(parents=True, exist_ok=True)
    (score_router / "source_score_router_gate.json").write_text("{}", encoding="utf-8")
    for split in ("validation", "test"):
        tiny_rows = []
        qwen_rows = []
        for row_index in range(8):
            answer_index = row_index % 3
            content_id = f"{split}-{row_index}"
            tiny_scores = [0.0, 0.0, 0.0]
            tiny_scores[answer_index] = 3.0
            qwen_scores = [1.0, 0.0, 0.0]
            tiny_rows.append(_score_cache_row(content_id, tiny_scores, answer_index))
            qwen_rows.append(_score_cache_row(content_id, qwen_scores, 0))
        _write_jsonl(score_router / "source_score_caches" / f"tinyllama_{split}_source_scores.jsonl", tiny_rows)
        _write_jsonl(score_router / "source_score_caches" / f"qwen_{split}_source_scores.jsonl", qwen_rows)

    payload = gate.build_candidate_syndrome_connector_gate(
        parent_dir=parent,
        score_router_dir=score_router,
        output_dir=output,
        bootstrap_samples=20,
        cv_folds=2,
        l2_grid=(0.1, 1.0),
        min_accuracy=0.5,
    )

    assert payload["gate"] == "source_private_arc_challenge_candidate_syndrome_connector_gate"
    assert payload["test_disagreement_rows"] == 8
    assert len(payload["view_rows"]) == 3
    assert payload["selected_primary_view"]["view"] in gate.PRIMARY_VIEWS
    assert (output / "candidate_syndrome_connector_gate.json").exists()
    assert (output / "candidate_syndrome_connector_rows.csv").exists()
    assert (output / "candidate_syndrome_connector_predictions.jsonl").exists()
